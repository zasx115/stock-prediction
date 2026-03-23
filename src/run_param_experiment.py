#!/usr/bin/env python3
# ============================================
# 파일명: src/run_param_experiment.py
# 설명: 핵심 파라미터 비교 실험 (Walk-Forward Sliding Window)
#
# 3가지 실험 카테고리:
#   1. 라벨 정의 (TARGET_DAYS × TARGET_RETURN)
#   2. Walk-Forward 재훈련 주기 (step_months)
#   3. 하이브리드 가중치 (momentum_weight / ai_weight)
#
# 사용법:
#   python run_param_experiment.py --experiment label
#   python run_param_experiment.py --experiment walkforward
#   python run_param_experiment.py --experiment hybrid_weight
#   python run_param_experiment.py --experiment all
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
from run_top_experiment import create_spy_portfolio


# ============================================
# [1] 실험 정의
# ============================================

def get_label_experiments():
    """라벨 정의 실험: TARGET_DAYS × TARGET_RETURN 조합"""
    return [
        {'name': '5D-3%',  'target_days': 5,  'target_return': 0.03, 'desc': '5일 +3% (쉬운 라벨)'},
        {'name': '5D-5%',  'target_days': 5,  'target_return': 0.05, 'desc': '5일 +5% (현재 기본값)'},
        {'name': '10D-5%', 'target_days': 10, 'target_return': 0.05, 'desc': '10일 +5% (중기)'},
        {'name': '10D-7%', 'target_days': 10, 'target_return': 0.07, 'desc': '10일 +7% (중기 급등)'},
        {'name': '20D-5%', 'target_days': 20, 'target_return': 0.05, 'desc': '20일 +5% (장기 안정)'},
    ]


def get_walkforward_experiments():
    """Walk-Forward 재훈련 주기 실험"""
    return [
        {'name': 'WF-1M', 'step_months': 1, 'desc': '매월 재훈련'},
        {'name': 'WF-3M', 'step_months': 3, 'desc': '분기별 재훈련 (기본값)'},
        {'name': 'WF-6M', 'step_months': 6, 'desc': '반기별 재훈련'},
    ]


def get_hybrid_weight_experiments():
    """하이브리드 가중치 실험"""
    return [
        {'name': 'Pure-AI',   'momentum_weight': 0.0,  'ai_weight': 1.0,  'desc': 'AI 100%'},
        {'name': 'AI-Heavy',  'momentum_weight': 0.2,  'ai_weight': 0.8,  'desc': 'AI 80% + 모멘텀 20%'},
        {'name': 'Base',      'momentum_weight': 0.35, 'ai_weight': 0.65, 'desc': 'AI 65% + 모멘텀 35% (기본값)'},
        {'name': 'Balanced',  'momentum_weight': 0.5,  'ai_weight': 0.5,  'desc': 'AI 50% + 모멘텀 50%'},
        {'name': 'Mom-Heavy', 'momentum_weight': 0.65, 'ai_weight': 0.35, 'desc': '모멘텀 65% + AI 35%'},
    ]


# ============================================
# [2] 라벨 실험 실행
# ============================================

def run_label_experiment(args, all_raw, backtest_end):
    """라벨 정의 변경 실험"""
    experiments = get_label_experiments()
    folds = generate_walk_forward_folds(
        args.train_start, args.train_end, backtest_end, args.wf_step_months
    )

    print("=" * 70)
    print("[실험 1] 라벨 정의 비교 (TARGET_DAYS × TARGET_RETURN)")
    print("=" * 70)
    print(f"학습 초기 : {args.train_start} ~ {args.train_end}")
    print(f"백테스트  : {args.train_end} ~ {backtest_end}")
    print(f"WF 스텝   : {args.wf_step_months}개월 / 폴드 수: {len(folds)}개")
    print(f"비교 대상 : {', '.join(e['name'] for e in experiments)} + SPY")
    print()

    for i, f in enumerate(folds):
        print(f"  Fold {i+1:2d}: Train {f['train_start']}~{f['train_end']} | Test {f['test_start']}~{f['test_end']}")
    print()

    # 종목 목록
    try:
        sp500 = get_sp500_list()
        symbols = sp500['symbol'].tolist() + ['SPY']
    except Exception:
        from config import SP500_BACKUP
        symbols = SP500_BACKUP + ['SPY']

    # 각 라벨 정의별로 피처 생성 → 별도 파이프라인 필요
    aggregated_results = []

    for exp in experiments:
        print(f"\n{'='*60}")
        print(f"[{exp['name']}] {exp['desc']} (target_days={exp['target_days']}, target_return={exp['target_return']})")
        print(f"{'='*60}")

        # 라벨 파라미터를 변경하여 피처 재생성
        all_features = create_features(
            all_raw,
            target_return=exp['target_return'],
            target_days=exp['target_days'],
        )
        all_features['date'] = pd.to_datetime(all_features['date'])
        feature_cols = get_feature_columns(all_features)

        exp_fold_portfolios = []
        exp_fold_trades = []

        for fold_idx, fold in enumerate(folds):
            fold_train_start = pd.Timestamp(fold['train_start'])
            fold_train_end = pd.Timestamp(fold['train_end'])
            fold_test_end = pd.Timestamp(fold['test_end'])

            train_features = all_features[
                (all_features['date'] >= fold_train_start) &
                (all_features['date'] <= fold_train_end)
            ]
            test_features = all_features[
                (all_features['date'] > fold_train_end) &
                (all_features['date'] <= fold_test_end)
            ]
            test_raw = all_raw[
                (all_raw['date'] > fold_train_end) &
                (all_raw['date'] <= fold_test_end)
            ]

            if len(test_features) == 0:
                continue

            momentum_lookback = fold_train_end - pd.DateOffset(months=7)
            price_raw = all_raw[
                (all_raw['date'] >= momentum_lookback) &
                (all_raw['date'] <= fold_test_end)
            ]
            price_df = prepare_price_data(price_raw)

            exp_dict = {
                'name': exp['name'],
                'desc': exp['desc'],
                'params': XGB_PARAMS.copy(),
            }

            try:
                r = run_single_experiment(
                    exp_dict, train_features, test_features, feature_cols,
                    price_df, test_raw, args
                )
                exp_fold_portfolios.append(r['portfolio'])
                exp_fold_trades.append(r['trades'])
                m = r['metrics']
                print(f"  Fold {fold_idx+1}: 수익률={m['total_return']*100:+.1f}% Sharpe={m['sharpe_ratio']:.2f}")
            except Exception as e:
                print(f"  Fold {fold_idx+1}: 실패 - {e}")

        if not exp_fold_portfolios:
            continue

        stitched = stitch_portfolios(exp_fold_portfolios, args.initial_capital)
        trade_dfs = [t for t in exp_fold_trades if not t.empty]
        all_trades = pd.concat(trade_dfs, ignore_index=True) if trade_dfs else pd.DataFrame()

        spy_raw = all_raw[
            (all_raw['symbol'] == 'SPY') &
            (all_raw['date'] > pd.Timestamp(args.train_end))
        ][['date', 'close']].rename(columns={'close': 'spy_close'})

        metrics = calculate_ai_metrics(
            stitched, all_trades, spy_raw,
            args.initial_capital, args.slippage
        )

        aggregated_results.append({
            'name': exp['name'],
            'desc': exp['desc'],
            'metrics': metrics,
            'portfolio': stitched,
        })

    # SPY 추가
    spy_result = _add_spy_result(all_raw, folds, args)
    if spy_result:
        aggregated_results.append(spy_result)

    return aggregated_results, folds


# ============================================
# [3] Walk-Forward 주기 실험 실행
# ============================================

def run_walkforward_experiment(args, all_raw, all_features, feature_cols, backtest_end):
    """Walk-Forward 재훈련 주기 비교 실험"""
    experiments = get_walkforward_experiments()

    print("=" * 70)
    print("[실험 2] Walk-Forward 재훈련 주기 비교")
    print("=" * 70)
    print(f"학습 초기 : {args.train_start} ~ {args.train_end}")
    print(f"백테스트  : {args.train_end} ~ {backtest_end}")
    print(f"비교 대상 : {', '.join(e['name'] for e in experiments)} + SPY")
    print()

    aggregated_results = []
    all_folds = {}

    for exp in experiments:
        step = exp['step_months']
        folds = generate_walk_forward_folds(
            args.train_start, args.train_end, backtest_end, step
        )
        all_folds[exp['name']] = folds

        print(f"\n{'='*60}")
        print(f"[{exp['name']}] {exp['desc']} (step={step}M, 폴드={len(folds)}개)")
        print(f"{'='*60}")

        exp_fold_portfolios = []
        exp_fold_trades = []

        for fold_idx, fold in enumerate(folds):
            fold_train_start = pd.Timestamp(fold['train_start'])
            fold_train_end = pd.Timestamp(fold['train_end'])
            fold_test_end = pd.Timestamp(fold['test_end'])

            train_features = all_features[
                (all_features['date'] >= fold_train_start) &
                (all_features['date'] <= fold_train_end)
            ]
            test_features = all_features[
                (all_features['date'] > fold_train_end) &
                (all_features['date'] <= fold_test_end)
            ]
            test_raw = all_raw[
                (all_raw['date'] > fold_train_end) &
                (all_raw['date'] <= fold_test_end)
            ]

            if len(test_features) == 0:
                continue

            momentum_lookback = fold_train_end - pd.DateOffset(months=7)
            price_raw = all_raw[
                (all_raw['date'] >= momentum_lookback) &
                (all_raw['date'] <= fold_test_end)
            ]
            price_df = prepare_price_data(price_raw)

            exp_dict = {
                'name': exp['name'],
                'desc': exp['desc'],
                'params': XGB_PARAMS.copy(),
            }

            try:
                r = run_single_experiment(
                    exp_dict, train_features, test_features, feature_cols,
                    price_df, test_raw, args
                )
                exp_fold_portfolios.append(r['portfolio'])
                exp_fold_trades.append(r['trades'])
                m = r['metrics']
                print(f"  Fold {fold_idx+1}: 수익률={m['total_return']*100:+.1f}% Sharpe={m['sharpe_ratio']:.2f}")
            except Exception as e:
                print(f"  Fold {fold_idx+1}: 실패 - {e}")

        if not exp_fold_portfolios:
            continue

        stitched = stitch_portfolios(exp_fold_portfolios, args.initial_capital)
        trade_dfs = [t for t in exp_fold_trades if not t.empty]
        all_trades = pd.concat(trade_dfs, ignore_index=True) if trade_dfs else pd.DataFrame()

        spy_raw = all_raw[
            (all_raw['symbol'] == 'SPY') &
            (all_raw['date'] > pd.Timestamp(args.train_end))
        ][['date', 'close']].rename(columns={'close': 'spy_close'})

        metrics = calculate_ai_metrics(
            stitched, all_trades, spy_raw,
            args.initial_capital, args.slippage
        )

        aggregated_results.append({
            'name': exp['name'],
            'desc': exp['desc'],
            'metrics': metrics,
            'portfolio': stitched,
        })

    # SPY (3M 폴드 기준)
    folds_3m = all_folds.get('WF-3M', generate_walk_forward_folds(
        args.train_start, args.train_end, backtest_end, 3
    ))
    spy_result = _add_spy_result(all_raw, folds_3m, args)
    if spy_result:
        aggregated_results.append(spy_result)

    return aggregated_results, folds_3m


# ============================================
# [4] 하이브리드 가중치 실험 실행
# ============================================

def run_hybrid_weight_experiment(args, all_raw, all_features, feature_cols, backtest_end):
    """하이브리드 가중치 비교 실험"""
    experiments = get_hybrid_weight_experiments()
    folds = generate_walk_forward_folds(
        args.train_start, args.train_end, backtest_end, args.wf_step_months
    )

    print("=" * 70)
    print("[실험 3] 하이브리드 가중치 비교 (모멘텀 vs AI)")
    print("=" * 70)
    print(f"학습 초기 : {args.train_start} ~ {args.train_end}")
    print(f"백테스트  : {args.train_end} ~ {backtest_end}")
    print(f"WF 스텝   : {args.wf_step_months}개월 / 폴드 수: {len(folds)}개")
    print(f"비교 대상 : {', '.join(e['name'] for e in experiments)} + SPY")
    print()

    aggregated_results = []

    for exp in experiments:
        print(f"\n{'='*60}")
        print(f"[{exp['name']}] {exp['desc']}")
        print(f"{'='*60}")

        # 이 실험에서는 momentum_weight / ai_weight를 변경
        exp_args = argparse.Namespace(**vars(args))
        exp_args.momentum_weight = exp['momentum_weight']
        exp_args.ai_weight = exp['ai_weight']

        exp_fold_portfolios = []
        exp_fold_trades = []

        for fold_idx, fold in enumerate(folds):
            fold_train_start = pd.Timestamp(fold['train_start'])
            fold_train_end = pd.Timestamp(fold['train_end'])
            fold_test_end = pd.Timestamp(fold['test_end'])

            train_features = all_features[
                (all_features['date'] >= fold_train_start) &
                (all_features['date'] <= fold_train_end)
            ]
            test_features = all_features[
                (all_features['date'] > fold_train_end) &
                (all_features['date'] <= fold_test_end)
            ]
            test_raw = all_raw[
                (all_raw['date'] > fold_train_end) &
                (all_raw['date'] <= fold_test_end)
            ]

            if len(test_features) == 0:
                continue

            momentum_lookback = fold_train_end - pd.DateOffset(months=7)
            price_raw = all_raw[
                (all_raw['date'] >= momentum_lookback) &
                (all_raw['date'] <= fold_test_end)
            ]
            price_df = prepare_price_data(price_raw)

            exp_dict = {
                'name': exp['name'],
                'desc': exp['desc'],
                'params': XGB_PARAMS.copy(),
            }

            try:
                r = run_single_experiment(
                    exp_dict, train_features, test_features, feature_cols,
                    price_df, test_raw, exp_args
                )
                exp_fold_portfolios.append(r['portfolio'])
                exp_fold_trades.append(r['trades'])
                m = r['metrics']
                print(f"  Fold {fold_idx+1}: 수익률={m['total_return']*100:+.1f}% Sharpe={m['sharpe_ratio']:.2f}")
            except Exception as e:
                print(f"  Fold {fold_idx+1}: 실패 - {e}")

        if not exp_fold_portfolios:
            continue

        stitched = stitch_portfolios(exp_fold_portfolios, args.initial_capital)
        trade_dfs = [t for t in exp_fold_trades if not t.empty]
        all_trades = pd.concat(trade_dfs, ignore_index=True) if trade_dfs else pd.DataFrame()

        spy_raw = all_raw[
            (all_raw['symbol'] == 'SPY') &
            (all_raw['date'] > pd.Timestamp(args.train_end))
        ][['date', 'close']].rename(columns={'close': 'spy_close'})

        metrics = calculate_ai_metrics(
            stitched, all_trades, spy_raw,
            args.initial_capital, args.slippage
        )

        aggregated_results.append({
            'name': exp['name'],
            'desc': exp['desc'],
            'metrics': metrics,
            'portfolio': stitched,
        })

    # SPY
    spy_result = _add_spy_result(all_raw, folds, args)
    if spy_result:
        aggregated_results.append(spy_result)

    return aggregated_results, folds


# ============================================
# [5] SPY 벤치마크 헬퍼
# ============================================

def _add_spy_result(all_raw, folds, args):
    """SPY Buy & Hold 결과를 생성"""
    spy_fold_portfolios = []
    for fold in folds:
        spy_port = create_spy_portfolio(
            all_raw, fold['train_end'], fold['test_end'], args.initial_capital
        )
        if spy_port is not None:
            spy_fold_portfolios.append(spy_port)

    if not spy_fold_portfolios:
        return None

    stitched_spy = stitch_portfolios(spy_fold_portfolios, args.initial_capital)

    spy_raw = all_raw[
        (all_raw['symbol'] == 'SPY') &
        (all_raw['date'] > pd.Timestamp(args.train_end))
    ][['date', 'close']].rename(columns={'close': 'spy_close'})

    spy_metrics = calculate_ai_metrics(
        stitched_spy, pd.DataFrame(), spy_raw,
        args.initial_capital, 0
    )
    spy_metrics['alpha'] = 0.0

    return {
        'name': 'SPY',
        'desc': 'S&P 500 Buy & Hold (벤치마크)',
        'metrics': spy_metrics,
        'portfolio': stitched_spy,
    }


# ============================================
# [6] 결과 출력 / 저장 / 차트
# ============================================

def print_results_table(title, results, args, backtest_end, folds):
    """결과 테이블 출력"""
    print()
    print("=" * 105)
    print(f"   {title}")
    print("=" * 105)
    print(f"학습 초기 : {args.train_start} ~ {args.train_end}")
    print(f"백테스트  : {args.train_end} ~ {backtest_end}")
    print(f"초기자본  : ${args.initial_capital:,.2f}")
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

    print("=" * 105)

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


def save_results_csv(results, csv_path, experiment_type):
    """결과 CSV 저장"""
    rows = []
    for r in results:
        m = r['metrics']
        rows.append({
            'experiment_type': experiment_type,
            'experiment': r['name'],
            'description': r['desc'],
            'total_return': round(m['total_return'] * 100, 2),
            'cagr': round(m['cagr'] * 100, 2),
            'sharpe_ratio': round(m['sharpe_ratio'], 3),
            'mdd': round(m['mdd'] * 100, 2),
            'volatility': round(m.get('volatility', 0) * 100, 2),
            'alpha': round(m.get('alpha', 0) * 100, 2),
            'total_trades': m['total_trades'],
            'win_rate': round(m['win_rate'] * 100, 2),
        })

    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"CSV saved: {csv_path}")


def plot_results(results, output_path, title):
    """4-subplot 비교 차트 생성"""
    fig, axes = plt.subplots(4, 1, figsize=(14, 20))
    fig.suptitle(title, fontsize=12, fontweight='bold')

    names = [r['name'] for r in results]
    n = len(results)
    cmap = plt.cm.get_cmap('tab10', max(n, 10))
    colors = [cmap(i) for i in range(n)]
    # SPY는 항상 검정
    for i, r in enumerate(results):
        if r['name'] == 'SPY':
            colors[i] = '#000000'

    # Subplot 1: 누적 수익률
    ax1 = axes[0]
    for i, r in enumerate(results):
        port = r['portfolio'].copy()
        port['date'] = pd.to_datetime(port['date'])
        port['normalized'] = port['value'] / port['value'].iloc[0] * 100
        ls = '--' if r['name'] == 'SPY' else '-'
        lw = 2.5 if r['name'] == 'SPY' else 1.5
        ax1.plot(port['date'], port['normalized'],
                 label=r['name'], linewidth=lw, color=colors[i],
                 linestyle=ls, alpha=0.85)
    ax1.set_title('Cumulative Return (base=100)', fontsize=11)
    ax1.set_ylabel('Value (base=100)')
    ax1.legend(loc='upper left', fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))

    # Subplot 2: CAGR
    ax2 = axes[1]
    cagrs = [r['metrics']['cagr'] * 100 for r in results]
    bars = ax2.bar(names, cagrs, color=colors, alpha=0.8)
    ax2.axhline(y=0, color='black', linewidth=0.8)
    for bar, v in zip(bars, cagrs):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2,
                 f'{v:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')
    ax2.set_title('CAGR (%)', fontsize=11)
    ax2.set_ylabel('CAGR (%)')
    ax2.grid(True, alpha=0.3, axis='y')

    # Subplot 3: Sharpe
    ax3 = axes[2]
    sharpes = [r['metrics']['sharpe_ratio'] for r in results]
    bars3 = ax3.bar(names, sharpes, color=colors, alpha=0.8)
    ax3.axhline(y=1.0, color='green', linewidth=1, linestyle='--', label='Sharpe=1.0')
    ax3.axhline(y=0, color='black', linewidth=0.8)
    for bar, v in zip(bars3, sharpes):
        ax3.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                 f'{v:.2f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    ax3.set_title('Sharpe Ratio', fontsize=11)
    ax3.set_ylabel('Sharpe Ratio')
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3, axis='y')

    # Subplot 4: MDD
    ax4 = axes[3]
    mdds = [r['metrics']['mdd'] * 100 for r in results]
    bars4 = ax4.bar(names, mdds, color=colors, alpha=0.7)
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
    print(f"Chart saved: {output_path}")


# ============================================
# [7] CLI
# ============================================

def parse_args():
    parser = argparse.ArgumentParser(description='핵심 파라미터 비교 실험')
    parser.add_argument('--experiment', type=str, required=True,
                        choices=['label', 'walkforward', 'hybrid_weight', 'all'],
                        help='실험 유형: label, walkforward, hybrid_weight, all')
    parser.add_argument('--train_start', type=str, default='2013-07-01')
    parser.add_argument('--train_end', type=str, default='2018-07-01')
    parser.add_argument('--backtest_end', type=str, default='')
    parser.add_argument('--wf_step_months', type=int, default=3)
    parser.add_argument('--initial_capital', type=float, default=10000)
    parser.add_argument('--momentum_weight', type=float, default=0.35)
    parser.add_argument('--ai_weight', type=float, default=0.65)
    parser.add_argument('--commission', type=float, default=0.001)
    parser.add_argument('--slippage', type=float, default=0.001)
    parser.add_argument('--output_dir', type=str, default='..')
    return parser.parse_args()


# ============================================
# [8] 메인
# ============================================

def main():
    args = parse_args()
    backtest_end = args.backtest_end if args.backtest_end else datetime.now().strftime('%Y-%m-%d')
    experiment_type = args.experiment

    experiments_to_run = []
    if experiment_type == 'all':
        experiments_to_run = ['label', 'walkforward', 'hybrid_weight']
    else:
        experiments_to_run = [experiment_type]

    # ----- 종목 목록 -----
    print("[1] S&P 500 종목 목록 로딩...")
    try:
        sp500 = get_sp500_list()
        symbols = sp500['symbol'].tolist() + ['SPY']
    except Exception:
        from config import SP500_BACKUP
        symbols = SP500_BACKUP + ['SPY']
    print(f"종목 수: {len(symbols)}")

    # ----- 데이터 다운로드 -----
    print(f"\n[2] 전체 데이터 다운로드: {args.train_start} ~ {backtest_end}")
    all_raw = get_backtest_data(symbols, start_date=args.train_start, end_date=backtest_end)
    all_raw['date'] = pd.to_datetime(all_raw['date'])
    print(f"전체 데이터 rows: {len(all_raw):,}")

    # 기본 피처 생성 (label/walkforward/hybrid_weight 중 label이 아닌 실험용)
    all_features = None
    feature_cols = None
    if any(e in experiments_to_run for e in ['walkforward', 'hybrid_weight']):
        print("\n[3] 피처 생성 (기본 라벨)...")
        all_features = create_features(all_raw)
        all_features['date'] = pd.to_datetime(all_features['date'])
        feature_cols = get_feature_columns(all_features)
        print(f"피처 수: {len(feature_cols)} / 전체 샘플: {len(all_features):,}")

    # ----- 실험 실행 -----
    for exp_type in experiments_to_run:
        if exp_type == 'label':
            results, folds = run_label_experiment(args, all_raw, backtest_end)
            title = 'Label Definition Comparison'
            prefix = 'label'

        elif exp_type == 'walkforward':
            results, folds = run_walkforward_experiment(
                args, all_raw, all_features, feature_cols, backtest_end
            )
            title = 'Walk-Forward Step Comparison'
            prefix = 'walkforward'

        elif exp_type == 'hybrid_weight':
            results, folds = run_hybrid_weight_experiment(
                args, all_raw, all_features, feature_cols, backtest_end
            )
            title = 'Hybrid Weight Comparison'
            prefix = 'hybrid_weight'

        if not results:
            print(f"\n{exp_type}: 결과 없음, 스킵")
            continue

        # 출력
        print_results_table(title, results, args, backtest_end, folds)

        # CSV 저장
        csv_path = os.path.join(args.output_dir, f'param_exp_{prefix}_results.csv')
        save_results_csv(results, csv_path, exp_type)

        # 차트 저장
        chart_path = os.path.join(args.output_dir, f'param_exp_{prefix}_chart.png')
        full_title = (
            f"{title}\n"
            f"Train: {args.train_start}~{args.train_end} | "
            f"Backtest: {args.train_end}~{backtest_end}"
        )
        plot_results(results, chart_path, full_title)


if __name__ == '__main__':
    main()
