#!/usr/bin/env python3
# ============================================
# 파일명: src/run_top_experiment.py
# 설명: 상위 전략 비교 실험 (Walk-Forward Sliding Window)
#
# 비교 대상 (5가지):
#   1. LR-001          : learning_rate=0.01 (느린 학습)
#   2. Sampling50      : subsample=0.5, colsample_bytree=0.5 (강한 배깅)
#   3. Combined        : LR-001 + Sampling50 (느린 학습 + 강한 배깅)
#   4. Base(Hybrid)    : 기존 하이브리드 전략 (기본 파라미터)
#   5. SPY             : Buy & Hold 벤치마크
#
# 사용법:
#   python run_top_experiment.py
#   python run_top_experiment.py \
#     --train_start 2015-01-01 \
#     --train_end 2020-01-01 \
#     --backtest_end 2024-12-31
# ============================================

import sys
import os
import argparse
from datetime import datetime
from copy import deepcopy

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

sys.path.insert(0, os.path.dirname(__file__))

from data import get_backtest_data, get_sp500_list
from ai_data import create_features, get_feature_columns
from hybrid_strategy import HybridStrategy
from ai_backtest import run_ai_backtest, calculate_ai_metrics
from strategy import prepare_price_data, CustomStrategy, filter_tuesday
from ai_strategy import AIStrategy, XGB_PARAMS
from run_model_experiment import (
    generate_walk_forward_folds,
    stitch_portfolios,
    run_single_experiment,
)


# ============================================
# [1] 실험 설정 (5가지)
# ============================================

def get_top_experiments():
    """상위 전략 비교 실험 목록"""
    base = XGB_PARAMS.copy()

    experiments = [
        {
            'name': 'LR-001',
            'desc': 'lr=0.01 (느린 학습)',
            'params': {**base, 'learning_rate': 0.01},
        },
        {
            'name': 'Sampling50',
            'desc': 'subsample=0.5, colsample=0.5 (강한 배깅)',
            'params': {**base, 'subsample': 0.5, 'colsample_bytree': 0.5},
        },
        {
            'name': 'Combined',
            'desc': 'lr=0.01 + subsample=0.5, colsample=0.5',
            'params': {
                **base,
                'learning_rate': 0.01,
                'subsample': 0.5,
                'colsample_bytree': 0.5,
            },
        },
        {
            'name': 'Base',
            'desc': f"기존 하이브리드 (depth={base['max_depth']}, lr={base['learning_rate']})",
            'params': base,
        },
    ]
    return experiments


# ============================================
# [2] SPY Buy & Hold 포트폴리오 생성
# ============================================

def create_spy_portfolio(all_raw, test_start, test_end, initial_capital):
    """
    SPY Buy & Hold 전략의 포트폴리오 시계열 생성.

    Args:
        all_raw: 전체 원시 데이터
        test_start: 테스트 시작일
        test_end: 테스트 종료일
        initial_capital: 초기 자본금

    Returns:
        dict: {'portfolio': DataFrame, 'trades': DataFrame, 'metrics': dict}
    """
    spy = all_raw[
        (all_raw['symbol'] == 'SPY') &
        (all_raw['date'] > pd.Timestamp(test_start)) &
        (all_raw['date'] <= pd.Timestamp(test_end))
    ].sort_values('date').copy()

    if spy.empty:
        return None

    spy = spy.reset_index(drop=True)
    initial_price = spy.iloc[0]['close']
    shares = initial_capital / initial_price

    portfolio = pd.DataFrame({
        'date': spy['date'],
        'value': spy['close'] * shares,
        'cash': 0.0,
        'stock_value': spy['close'] * shares,
    })

    return portfolio


# ============================================
# [3] 결과 출력
# ============================================

def print_results_table(results, args, backtest_end, folds):
    """실험 결과 텍스트 테이블 출력"""
    print()
    print("=" * 95)
    print("   Top Strategy Comparison (Walk-Forward)")
    print("=" * 95)
    print(f"학습 초기   : {args.train_start} ~ {args.train_end}")
    print(f"백테스트    : {args.train_end} ~ {backtest_end}")
    print(f"WF 스텝     : {args.wf_step_months}개월  /  폴드 수: {len(folds)}개")
    print(f"초기자본    : ${args.initial_capital:,.2f}")
    print(f"가중치      : Momentum {args.momentum_weight*100:.0f}% / AI {args.ai_weight*100:.0f}%")
    print(f"수수료/슬립 : {args.commission*100:.2f}% / {args.slippage*100:.2f}%")
    print()

    header = (f"{'실험명':<14} {'설명':<42} {'수익률':>8} {'CAGR':>7} "
              f"{'샤프':>6} {'MDD':>7} {'Alpha':>7} {'거래':>5} {'승률':>6}")
    print(header)
    print("-" * 105)

    for r in results:
        m = r['metrics']
        trades_str = f"{m['total_trades']:>5}" if m['total_trades'] > 0 else "  B&H"
        win_str = f"{m['win_rate']*100:>5.1f}%" if m['total_trades'] > 0 else "    -"
        print(
            f"{r['name']:<14} "
            f"{r['desc']:<42} "
            f"{m['total_return']*100:>+7.1f}% "
            f"{m['cagr']*100:>6.1f}% "
            f"{m['sharpe_ratio']:>6.2f} "
            f"{m['mdd']*100:>6.1f}% "
            f"{m.get('alpha', 0)*100:>+6.1f}% "
            f"{trades_str} "
            f"{win_str}"
        )

    print("=" * 95)

    # SPY 제외한 전략들 중 베스트
    strategies = [r for r in results if r['name'] != 'SPY']
    if strategies:
        best_cagr = max(strategies, key=lambda r: r['metrics']['cagr'])
        best_sharpe = max(strategies, key=lambda r: r['metrics']['sharpe_ratio'])
        best_mdd = min(strategies, key=lambda r: r['metrics']['mdd'])

        print()
        print(f"  CAGR 최고  : {best_cagr['name']} ({best_cagr['metrics']['cagr']*100:.1f}%)")
        print(f"  Sharpe 최고: {best_sharpe['name']} ({best_sharpe['metrics']['sharpe_ratio']:.2f})")
        print(f"  MDD 최소   : {best_mdd['name']} ({best_mdd['metrics']['mdd']*100:.1f}%)")
        print()


# ============================================
# [4] 결과 CSV 저장
# ============================================

def save_results_csv(results, csv_path):
    """실험 결과를 CSV로 저장"""
    rows = []
    for r in results:
        m = r['metrics']
        rows.append({
            'experiment': r['name'],
            'description': r['desc'],
            'total_return': round(m['total_return'] * 100, 2),
            'cagr': round(m['cagr'] * 100, 2),
            'sharpe_ratio': round(m['sharpe_ratio'], 3),
            'mdd': round(m['mdd'] * 100, 2),
            'volatility': round(m.get('volatility', 0) * 100, 2),
            'alpha': round(m.get('alpha', 0) * 100, 2),
            'spy_return': round(m.get('spy_return', 0) * 100, 2),
            'total_trades': m['total_trades'],
            'win_rate': round(m['win_rate'] * 100, 2),
            'stop_loss_count': m.get('stop_loss_count', 0),
            'total_commission': round(m.get('total_commission', 0), 2),
        })

    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"Results CSV saved: {csv_path}")


# ============================================
# [5] 비교 차트 생성
# ============================================

def plot_results(results, output_path, args, backtest_end, folds):
    """4-subplot 비교 차트"""
    fig, axes = plt.subplots(4, 1, figsize=(14, 20))
    title = (
        f"Top Strategy Comparison: Walk-Forward (Step={args.wf_step_months}M, Folds={len(folds)})\n"
        f"Train Init: {args.train_start}~{args.train_end}  "
        f"| Backtest: {args.train_end}~{backtest_end}  "
        f"| Momentum {args.momentum_weight*100:.0f}% / AI {args.ai_weight*100:.0f}%"
    )
    fig.suptitle(title, fontsize=12, fontweight='bold')

    names = [r['name'] for r in results]
    # 전략별 색상: LR-001=파랑, Sampling50=초록, Combined=빨강, Base=회색, SPY=검정
    color_map = {
        'LR-001': '#1f77b4',
        'Sampling50': '#2ca02c',
        'Combined': '#d62728',
        'Base': '#7f7f7f',
        'SPY': '#000000',
    }
    colors = [color_map.get(r['name'], '#999999') for r in results]

    fold_boundaries = [pd.Timestamp(f['test_start']) for f in folds[1:]]

    # ----- Subplot 1: 누적 수익률 -----
    ax1 = axes[0]
    for i, r in enumerate(results):
        port = r['portfolio'].copy()
        port['date'] = pd.to_datetime(port['date'])
        port['normalized'] = port['value'] / port['value'].iloc[0] * 100
        ls = '--' if r['name'] == 'SPY' else '-'
        lw = 2.5 if r['name'] in ('Combined', 'SPY') else 1.5
        ax1.plot(port['date'], port['normalized'],
                 label=r['name'], linewidth=lw, color=colors[i],
                 linestyle=ls, alpha=0.85)

    for boundary in fold_boundaries:
        ax1.axvline(x=boundary, color='gray', linewidth=0.8, linestyle=':', alpha=0.7)

    ax1.set_title(f'Cumulative Return (base=100) | WF 폴드 경계 ({len(folds)}개)', fontsize=11)
    ax1.set_ylabel('Value (base=100)')
    ax1.legend(loc='upper left', fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))

    # ----- Subplot 2: CAGR -----
    ax2 = axes[1]
    cagrs = [r['metrics']['cagr'] * 100 for r in results]
    bar_colors = [color_map.get(r['name'], 'steelblue') for r in results]
    bars = ax2.bar(names, cagrs, color=bar_colors, alpha=0.8)
    ax2.axhline(y=0, color='black', linewidth=0.8)
    for bar, v in zip(bars, cagrs):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2,
                 f'{v:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')
    ax2.set_title('CAGR (%)', fontsize=11)
    ax2.set_ylabel('CAGR (%)')
    ax2.grid(True, alpha=0.3, axis='y')

    # ----- Subplot 3: 샤프 비율 -----
    ax3 = axes[2]
    sharpes = [r['metrics']['sharpe_ratio'] for r in results]
    bars3 = ax3.bar(names, sharpes, color=bar_colors, alpha=0.8)
    ax3.axhline(y=1.0, color='green', linewidth=1, linestyle='--', label='Sharpe=1.0')
    ax3.axhline(y=0, color='black', linewidth=0.8)
    for bar, v in zip(bars3, sharpes):
        ax3.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                 f'{v:.2f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    ax3.set_title('Sharpe Ratio', fontsize=11)
    ax3.set_ylabel('Sharpe Ratio')
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3, axis='y')

    # ----- Subplot 4: MDD -----
    ax4 = axes[3]
    mdds = [r['metrics']['mdd'] * 100 for r in results]
    bars4 = ax4.bar(names, mdds, color=bar_colors, alpha=0.7)
    ax4.axhline(y=0, color='black', linewidth=0.8)
    for bar, v in zip(bars4, mdds):
        ax4.text(bar.get_x() + bar.get_width() / 2, bar.get_height() - 0.3,
                 f'{v:.1f}%', ha='center', va='top', fontsize=9,
                 fontweight='bold', color='white')
    ax4.set_title('Max Drawdown (%)', fontsize=11)
    ax4.set_ylabel('MDD (%)')
    ax4.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nChart saved: {output_path}")


# ============================================
# [6] CLI 파싱
# ============================================

def parse_args():
    parser = argparse.ArgumentParser(description='상위 전략 비교 실험 (Walk-Forward)')
    parser.add_argument('--train_start', type=str, default='2015-01-01')
    parser.add_argument('--train_end', type=str, default='2020-01-01')
    parser.add_argument('--backtest_end', type=str, default='')
    parser.add_argument('--wf_step_months', type=int, default=3)
    parser.add_argument('--initial_capital', type=float, default=10000)
    parser.add_argument('--momentum_weight', type=float, default=0.35)
    parser.add_argument('--ai_weight', type=float, default=0.65)
    parser.add_argument('--commission', type=float, default=0.001)
    parser.add_argument('--slippage', type=float, default=0.001)
    parser.add_argument('--output', type=str, default='top_experiment_result.png')
    parser.add_argument('--results_csv', type=str, default='top_experiment_results.csv')
    return parser.parse_args()


# ============================================
# [7] 메인
# ============================================

def main():
    args = parse_args()
    backtest_end = args.backtest_end if args.backtest_end else datetime.now().strftime('%Y-%m-%d')

    total_weight = args.momentum_weight + args.ai_weight
    if abs(total_weight - 1.0) > 0.01:
        print(f"경고: momentum_weight({args.momentum_weight}) + "
              f"ai_weight({args.ai_weight}) = {total_weight:.2f}")

    folds = generate_walk_forward_folds(
        args.train_start, args.train_end, backtest_end, args.wf_step_months
    )

    experiments = get_top_experiments()

    print("=" * 60)
    print("[Top Strategy Comparison (Walk-Forward)]")
    print(f"학습 초기   : {args.train_start} ~ {args.train_end}")
    print(f"백테스트    : {args.train_end} ~ {backtest_end}")
    print(f"WF 스텝     : {args.wf_step_months}개월  /  폴드 수: {len(folds)}개")
    print(f"가중치      : Momentum {args.momentum_weight*100:.0f}% / AI {args.ai_weight*100:.0f}%")
    print(f"비교 전략   : {', '.join(e['name'] for e in experiments)} + SPY")
    print("=" * 60)

    if not folds:
        print("생성된 폴드가 없습니다.")
        return

    for i, f in enumerate(folds):
        print(f"  Fold {i+1:2d}: Train {f['train_start']}~{f['train_end']}  "
              f"| Test {f['test_start']}~{f['test_end']}")
    print()

    # ----- [1] 종목 목록 -----
    print("[1] S&P 500 종목 목록 로딩...")
    try:
        sp500 = get_sp500_list()
        symbols = sp500['symbol'].tolist() + ['SPY']
    except Exception:
        from config import SP500_BACKUP
        symbols = SP500_BACKUP + ['SPY']
    print(f"종목 수: {len(symbols)}")

    # ----- [2] 데이터 다운로드 -----
    print(f"\n[2] 전체 데이터 다운로드: {args.train_start} ~ {backtest_end}")
    all_raw = get_backtest_data(symbols, start_date=args.train_start, end_date=backtest_end)
    all_raw['date'] = pd.to_datetime(all_raw['date'])
    print(f"전체 데이터 rows: {len(all_raw):,}")

    # ----- [3] 피처 생성 -----
    print("\n[3] 피처 생성 중...")
    all_features = create_features(all_raw)
    all_features['date'] = pd.to_datetime(all_features['date'])
    feature_cols = get_feature_columns(all_features)
    print(f"피처 수: {len(feature_cols)}  /  전체 샘플: {len(all_features):,}")

    # ----- [4] Walk-Forward 폴드별 실행 -----
    exp_fold_data = {
        exp['name']: {'desc': exp['desc'], 'folds': []}
        for exp in experiments
    }
    # SPY도 폴드별 포트폴리오 수집
    spy_fold_portfolios = []

    for fold_idx, fold in enumerate(folds):
        fold_train_start = pd.Timestamp(fold['train_start'])
        fold_train_end = pd.Timestamp(fold['train_end'])
        fold_test_end = pd.Timestamp(fold['test_end'])

        print(f"\n{'='*60}")
        print(f"[Fold {fold_idx+1}/{len(folds)}]  "
              f"Train: {fold['train_start']}~{fold['train_end']}  "
              f"| Test: {fold['test_start']}~{fold['test_end']}")
        print(f"{'='*60}")

        train_features_fold = all_features[
            (all_features['date'] >= fold_train_start) &
            (all_features['date'] <= fold_train_end)
        ]
        test_features_fold = all_features[
            (all_features['date'] > fold_train_end) &
            (all_features['date'] <= fold_test_end)
        ]
        test_raw_fold = all_raw[
            (all_raw['date'] > fold_train_end) &
            (all_raw['date'] <= fold_test_end)
        ]

        if len(test_features_fold) == 0:
            print(f"  경고: 테스트 데이터 없음, Fold {fold_idx+1} 스킵")
            continue

        print(f"학습 샘플: {len(train_features_fold):,}  /  테스트 샘플: {len(test_features_fold):,}")

        momentum_lookback = fold_train_end - pd.DateOffset(months=7)
        price_raw_fold = all_raw[
            (all_raw['date'] >= momentum_lookback) &
            (all_raw['date'] <= fold_test_end)
        ]
        price_df_fold = prepare_price_data(price_raw_fold)

        # SPY Buy & Hold (폴드별)
        spy_port = create_spy_portfolio(
            all_raw, fold['train_end'], fold['test_end'], args.initial_capital
        )
        if spy_port is not None:
            spy_fold_portfolios.append(spy_port)

        # 4개 전략 실행
        for exp in experiments:
            print(f"\n  [{exp['name']}] {exp['desc']}")
            try:
                r = run_single_experiment(
                    exp, train_features_fold, test_features_fold, feature_cols,
                    price_df_fold, test_raw_fold, args
                )
                exp_fold_data[exp['name']]['folds'].append({
                    'portfolio': r['portfolio'],
                    'trades': r['trades'],
                })
                m = r['metrics']
                print(f"    → 수익률: {m['total_return']*100:+.1f}%  "
                      f"CAGR: {m['cagr']*100:.1f}%  "
                      f"Sharpe: {m['sharpe_ratio']:.2f}  "
                      f"MDD: {m['mdd']*100:.1f}%")
            except Exception as e:
                print(f"    ✗ 실패: {e}")

    # ----- [5] 결과 집계 -----
    print(f"\n\n[집계] Walk-Forward 결과 통합 중...")

    spy_raw = all_raw[
        (all_raw['symbol'] == 'SPY') &
        (all_raw['date'] > pd.Timestamp(args.train_end))
    ][['date', 'close']].rename(columns={'close': 'spy_close'})

    aggregated_results = []

    for exp in experiments:
        data = exp_fold_data[exp['name']]
        fold_list = data['folds']

        if not fold_list:
            print(f"  {exp['name']}: 유효 폴드 없음, 스킵")
            continue

        stitched_portfolio = stitch_portfolios(
            [f['portfolio'] for f in fold_list], args.initial_capital
        )
        trade_dfs = [f['trades'] for f in fold_list if not f['trades'].empty]
        all_trades = pd.concat(trade_dfs, ignore_index=True) if trade_dfs else pd.DataFrame()

        metrics = calculate_ai_metrics(
            stitched_portfolio, all_trades, spy_raw,
            args.initial_capital, args.slippage
        )

        aggregated_results.append({
            'name': exp['name'],
            'desc': data['desc'],
            'metrics': metrics,
            'portfolio': stitched_portfolio,
        })

        print(f"  {exp['name']}: "
              f"수익률={metrics['total_return']*100:+.1f}%  "
              f"CAGR={metrics['cagr']*100:.1f}%  "
              f"Sharpe={metrics['sharpe_ratio']:.2f}")

    # SPY Buy & Hold 집계
    if spy_fold_portfolios:
        stitched_spy = stitch_portfolios(spy_fold_portfolios, args.initial_capital)
        spy_metrics = calculate_ai_metrics(
            stitched_spy, pd.DataFrame(), spy_raw,
            args.initial_capital, 0
        )
        spy_metrics['alpha'] = 0.0

        aggregated_results.append({
            'name': 'SPY',
            'desc': 'S&P 500 Buy & Hold (벤치마크)',
            'metrics': spy_metrics,
            'portfolio': stitched_spy,
        })

        print(f"  SPY: "
              f"수익률={spy_metrics['total_return']*100:+.1f}%  "
              f"CAGR={spy_metrics['cagr']*100:.1f}%  "
              f"Sharpe={spy_metrics['sharpe_ratio']:.2f}")

    if not aggregated_results:
        print("집계된 결과가 없습니다.")
        return

    # ----- [6] 출력 -----
    print_results_table(aggregated_results, args, backtest_end, folds)
    save_results_csv(aggregated_results, args.results_csv)
    plot_results(aggregated_results, args.output, args, backtest_end, folds)


if __name__ == '__main__':
    main()
