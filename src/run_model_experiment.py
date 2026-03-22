#!/usr/bin/env python3
# ============================================
# 파일명: src/run_model_experiment.py
# 설명: XGBoost 하이퍼파라미터 실험 비교 백테스트
#
# 역할 요약:
#   다양한 XGBoost 파라미터 조합으로 HybridStrategy를 학습하고
#   각 실험의 백테스트 성과를 비교하여 최적 파라미터를 탐색.
#
# 실험 목록 (9가지):
#   0. Base       : 현재 기본값 (max_depth=4, lr=0.03, n_est=1000, spw=3)
#   1. Depth-3    : max_depth=3  (더 얕은 트리 → 과적합 감소)
#   2. Depth-6    : max_depth=6  (더 깊은 트리 → 표현력 증가)
#   3. LR-001     : learning_rate=0.01  (느린 학습)
#   4. LR-005     : learning_rate=0.05  (빠른 학습)
#   5. SPW-2      : scale_pos_weight=2  (클래스 불균형 보정 약화)
#   6. SPW-5      : scale_pos_weight=5  (클래스 불균형 보정 강화)
#   7. Sampling50 : subsample=0.5, colsample_bytree=0.5  (강한 배깅)
#   8. Est-500    : n_estimators=500  (트리 수 절반)
#
# 실행 흐름:
#   1. argparse로 CLI 인자 파싱
#   2. S&P 500 + SPY 데이터 다운로드 (학습/백테스트)
#   3. 피처 생성 (create_features)
#   4. 각 실험별 HybridStrategy 학습 → run_ai_backtest 시뮬레이션
#   5. 결과 DataFrame 생성 → CSV 저장 + 텍스트 출력
#   6. 4-subplot 비교 차트 생성 및 PNG 저장
#
# 사용법:
#   python run_model_experiment.py
#   python run_model_experiment.py \
#     --train_start 2015-01-01 \
#     --train_end 2020-01-01 \
#     --backtest_start 2020-01-01 \
#     --backtest_end 2024-12-31 \
#     --initial_capital 10000 \
#     --output experiment_result.png \
#     --results_csv experiment_results.csv
#
# 의존 관계:
#   ← data.py (get_backtest_data, get_sp500_list)
#   ← ai_data.py (create_features, get_feature_columns)
#   ← hybrid_strategy.py (HybridStrategy)
#   ← ai_backtest.py (run_ai_backtest)
#   ← strategy.py (prepare_price_data)
#   ← config.py (SP500_BACKUP)
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
from ai_backtest import run_ai_backtest
from strategy import prepare_price_data
from ai_strategy import XGB_PARAMS


# ============================================
# [1] 실험 설정 목록
# ============================================

def get_experiments():
    """
    비교할 XGBoost 파라미터 실험 목록 반환.

    Returns:
        list of dict: [{'name': str, 'params': dict}, ...]
    """
    base = XGB_PARAMS.copy()

    experiments = [
        {
            'name': 'Base',
            'desc': f"max_depth={base['max_depth']}, lr={base['learning_rate']}, "
                    f"n_est={base['n_estimators']}, spw={base['scale_pos_weight']}",
            'params': base,
        },
        {
            'name': 'Depth-3',
            'desc': 'max_depth=3 (얕은 트리)',
            'params': {**base, 'max_depth': 3},
        },
        {
            'name': 'Depth-6',
            'desc': 'max_depth=6 (깊은 트리)',
            'params': {**base, 'max_depth': 6},
        },
        {
            'name': 'LR-001',
            'desc': 'learning_rate=0.01 (느린 학습)',
            'params': {**base, 'learning_rate': 0.01},
        },
        {
            'name': 'LR-005',
            'desc': 'learning_rate=0.05 (빠른 학습)',
            'params': {**base, 'learning_rate': 0.05},
        },
        {
            'name': 'SPW-2',
            'desc': 'scale_pos_weight=2 (약한 불균형 보정)',
            'params': {**base, 'scale_pos_weight': 2},
        },
        {
            'name': 'SPW-5',
            'desc': 'scale_pos_weight=5 (강한 불균형 보정)',
            'params': {**base, 'scale_pos_weight': 5},
        },
        {
            'name': 'Sampling50',
            'desc': 'subsample=0.5, colsample=0.5 (강한 배깅)',
            'params': {**base, 'subsample': 0.5, 'colsample_bytree': 0.5},
        },
        {
            'name': 'Est-500',
            'desc': 'n_estimators=500 (트리 수 절반)',
            'params': {**base, 'n_estimators': 500},
        },
    ]
    return experiments


# ============================================
# [2] 단일 실험 실행
# ============================================

def run_single_experiment(exp, train_features, test_features, feature_cols,
                          price_df, test_raw, args):
    """
    하나의 실험(파라미터 조합)에 대해 HybridStrategy를 학습하고 백테스트를 실행.

    Args:
        exp (dict): {'name', 'desc', 'params'}
        train_features (pd.DataFrame): 학습용 피처 DataFrame
        test_features (pd.DataFrame): 테스트용 피처 DataFrame
        feature_cols (list): 피처 컬럼 목록
        price_df (pd.DataFrame): 모멘텀 전략용 가격 데이터 (피벗)
        test_raw (pd.DataFrame): 원시 테스트 데이터 (SPY 수익률 계산용)
        args: argparse Namespace

    Returns:
        dict: 실험 이름, 설명, 메트릭, portfolio_df
    """
    strategy = HybridStrategy(
        weight_momentum=args.momentum_weight,
        weight_ai=args.ai_weight,
        xgb_params=exp['params'],
    )
    strategy.prepare(train_features, price_df, feature_cols)

    result = run_ai_backtest(
        strategy=strategy,
        test_df=test_features,
        feature_cols=feature_cols,
        initial_capital=args.initial_capital,
        commission=args.commission,
        slippage=args.slippage,
    )

    metrics = result['metrics']
    portfolio_df = result['portfolio']

    # SPY 수익률 보정 (create_features가 SPY 제외하므로 test_raw에서 직접 계산)
    if 'SPY' in test_raw['symbol'].unique():
        spy = test_raw[test_raw['symbol'] == 'SPY'].sort_values('date')
        if len(spy) >= 2:
            spy_return = (spy.iloc[-1]['close'] - spy.iloc[0]['close']) / spy.iloc[0]['close']
            metrics['spy_return'] = spy_return
            metrics['alpha'] = metrics['total_return'] - spy_return

    return {
        'name': exp['name'],
        'desc': exp['desc'],
        'metrics': metrics,
        'portfolio': portfolio_df,
    }


# ============================================
# [3] 결과 출력
# ============================================

def print_results_table(results, args, backtest_end):
    """실험 결과 텍스트 테이블 출력"""
    print()
    print("=" * 75)
    print("   Model Experiment Results")
    print("=" * 75)
    print(f"학습기간    : {args.train_start} ~ {args.train_end}")
    print(f"백테스트    : {args.backtest_start} ~ {backtest_end}")
    print(f"초기자본    : ${args.initial_capital:,.2f}")
    print(f"가중치      : Momentum {args.momentum_weight*100:.0f}% / AI {args.ai_weight*100:.0f}%")
    print(f"수수료/슬립 : {args.commission*100:.2f}% / {args.slippage*100:.2f}%")
    print()

    header = f"{'실험명':<12} {'설명':<32} {'수익률':>8} {'CAGR':>7} {'샤프':>6} {'MDD':>7} {'Alpha':>7} {'거래':>5} {'승률':>6}"
    print(header)
    print("-" * 95)

    for r in results:
        m = r['metrics']
        calmar = abs(m['cagr'] / m['mdd']) if m.get('mdd', 0) != 0 else 0
        print(
            f"{r['name']:<12} "
            f"{r['desc']:<32} "
            f"{m['total_return']*100:>+7.1f}% "
            f"{m['cagr']*100:>6.1f}% "
            f"{m['sharpe_ratio']:>6.2f} "
            f"{m['mdd']*100:>6.1f}% "
            f"{m.get('alpha', 0)*100:>+6.1f}% "
            f"{m['total_trades']:>5} "
            f"{m['win_rate']*100:>5.1f}%"
        )

    print("=" * 75)

    # 최고 성과 실험 강조
    best_cagr = max(results, key=lambda r: r['metrics']['cagr'])
    best_sharpe = max(results, key=lambda r: r['metrics']['sharpe_ratio'])
    best_mdd = min(results, key=lambda r: r['metrics']['mdd'])

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

def plot_results(results, test_raw, output_path, args, backtest_end):
    """
    4-subplot 비교 차트 생성 및 저장.

    Subplot 1: 누적 수익률 곡선 (전략별 + SPY)
    Subplot 2: CAGR 막대 차트
    Subplot 3: 샤프 비율 막대 차트
    Subplot 4: MDD 막대 차트
    """
    fig, axes = plt.subplots(4, 1, figsize=(14, 20))
    title = (
        f"Model Experiment: XGBoost Hyperparameter Comparison\n"
        f"Train: {args.train_start}~{args.train_end}  "
        f"| Backtest: {args.backtest_start}~{backtest_end}  "
        f"| Momentum {args.momentum_weight*100:.0f}% / AI {args.ai_weight*100:.0f}%"
    )
    fig.suptitle(title, fontsize=12, fontweight='bold')

    names = [r['name'] for r in results]
    colors = plt.cm.tab10(np.linspace(0, 1, len(results)))

    # ----- Subplot 1: 누적 수익률 -----
    ax1 = axes[0]

    for i, r in enumerate(results):
        port = r['portfolio'].copy()
        port['date'] = pd.to_datetime(port['date'])
        port['normalized'] = port['value'] / port['value'].iloc[0] * 100
        ax1.plot(port['date'], port['normalized'],
                 label=r['name'], linewidth=1.5, color=colors[i], alpha=0.85)

    # SPY 기준선
    if 'SPY' in test_raw['symbol'].unique():
        spy = test_raw[test_raw['symbol'] == 'SPY'].sort_values('date').copy()
        spy['date'] = pd.to_datetime(spy['date'])
        spy['normalized'] = spy['close'] / spy['close'].iloc[0] * 100
        ax1.plot(spy['date'], spy['normalized'],
                 label='SPY', linewidth=2.5, linestyle='--',
                 color='black', alpha=0.6)

    ax1.set_title('Cumulative Return (base=100)', fontsize=11)
    ax1.set_ylabel('Value (base=100)')
    ax1.legend(loc='upper left', fontsize=7, ncol=3)
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))

    # ----- Subplot 2: CAGR -----
    ax2 = axes[1]
    cagrs = [r['metrics']['cagr'] * 100 for r in results]
    bar_colors = ['steelblue' if v >= 0 else 'crimson' for v in cagrs]
    bars = ax2.bar(names, cagrs, color=bar_colors, alpha=0.8)
    ax2.axhline(y=0, color='black', linewidth=0.8)

    # SPY CAGR 기준선 표시
    if results:
        spy_cagr = results[0]['metrics'].get('spy_return', 0)
        if spy_cagr != 0:
            # 간단히 총수익률에서 연환산 (정확하지 않으나 시각적 참고용)
            ax2.axhline(y=spy_cagr * 100, color='orange', linewidth=1.5,
                        linestyle='--', label=f'SPY Total Return {spy_cagr*100:.1f}%')
            ax2.legend(fontsize=8)

    for bar, v in zip(bars, cagrs):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2,
                 f'{v:.1f}%', ha='center', va='bottom', fontsize=8)
    ax2.set_title('CAGR (%)', fontsize=11)
    ax2.set_ylabel('CAGR (%)')
    ax2.grid(True, alpha=0.3, axis='y')
    plt.setp(ax2.get_xticklabels(), rotation=30, ha='right', fontsize=8)

    # ----- Subplot 3: 샤프 비율 -----
    ax3 = axes[2]
    sharpes = [r['metrics']['sharpe_ratio'] for r in results]
    bar_colors3 = ['steelblue' if v >= 1 else ('orange' if v >= 0 else 'crimson')
                   for v in sharpes]
    bars3 = ax3.bar(names, sharpes, color=bar_colors3, alpha=0.8)
    ax3.axhline(y=1.0, color='green', linewidth=1, linestyle='--', label='Sharpe=1.0')
    ax3.axhline(y=0, color='black', linewidth=0.8)

    for bar, v in zip(bars3, sharpes):
        ax3.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                 f'{v:.2f}', ha='center', va='bottom', fontsize=8)
    ax3.set_title('Sharpe Ratio', fontsize=11)
    ax3.set_ylabel('Sharpe Ratio')
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3, axis='y')
    plt.setp(ax3.get_xticklabels(), rotation=30, ha='right', fontsize=8)

    # ----- Subplot 4: MDD -----
    ax4 = axes[3]
    mdds = [r['metrics']['mdd'] * 100 for r in results]
    bars4 = ax4.bar(names, mdds, color='crimson', alpha=0.7)
    ax4.axhline(y=0, color='black', linewidth=0.8)

    for bar, v in zip(bars4, mdds):
        ax4.text(bar.get_x() + bar.get_width() / 2, bar.get_height() - 0.3,
                 f'{v:.1f}%', ha='center', va='top', fontsize=8, color='white')
    ax4.set_title('Max Drawdown (%)', fontsize=11)
    ax4.set_ylabel('MDD (%)')
    ax4.grid(True, alpha=0.3, axis='y')
    plt.setp(ax4.get_xticklabels(), rotation=30, ha='right', fontsize=8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nChart saved: {output_path}")


# ============================================
# [6] CLI 파싱
# ============================================

def parse_args():
    parser = argparse.ArgumentParser(description='XGBoost 하이퍼파라미터 실험 비교')
    parser.add_argument('--train_start', type=str, default='2015-01-01',
                        help='AI 학습 시작일 (YYYY-MM-DD)')
    parser.add_argument('--train_end', type=str, default='2020-01-01',
                        help='AI 학습 종료일 (YYYY-MM-DD)')
    parser.add_argument('--backtest_start', type=str, default='2020-01-01',
                        help='백테스트 시작일 (YYYY-MM-DD)')
    parser.add_argument('--backtest_end', type=str, default='',
                        help='백테스트 종료일 (YYYY-MM-DD, 빈값=오늘)')
    parser.add_argument('--initial_capital', type=float, default=10000,
                        help='초기 자본금 (USD)')
    parser.add_argument('--momentum_weight', type=float, default=0.35,
                        help='모멘텀 가중치 (기본 0.35)')
    parser.add_argument('--ai_weight', type=float, default=0.65,
                        help='AI 가중치 (기본 0.65)')
    parser.add_argument('--commission', type=float, default=0.001,
                        help='수수료율 (기본 0.001 = 0.1%%)')
    parser.add_argument('--slippage', type=float, default=0.001,
                        help='슬리피지율 (기본 0.001 = 0.1%%)')
    parser.add_argument('--output', type=str, default='experiment_result.png',
                        help='차트 저장 경로')
    parser.add_argument('--results_csv', type=str, default='experiment_results.csv',
                        help='결과 CSV 저장 경로')
    return parser.parse_args()


# ============================================
# [7] 메인
# ============================================

def main():
    args = parse_args()

    backtest_end = args.backtest_end if args.backtest_end else datetime.now().strftime('%Y-%m-%d')

    # 가중치 합 검증
    total_weight = args.momentum_weight + args.ai_weight
    if abs(total_weight - 1.0) > 0.01:
        print(f"경고: momentum_weight({args.momentum_weight}) + "
              f"ai_weight({args.ai_weight}) = {total_weight:.2f} (합계가 1.0이 아님)")

    print("=" * 60)
    print("[Model Experiment 설정]")
    print(f"학습기간   : {args.train_start} ~ {args.train_end}")
    print(f"백테스트   : {args.backtest_start} ~ {backtest_end}")
    print(f"가중치     : Momentum {args.momentum_weight*100:.0f}% / AI {args.ai_weight*100:.0f}%")
    print(f"실험 수    : {len(get_experiments())}개")
    print("=" * 60)

    # ----- [1] 종목 목록 -----
    print("\n[1] S&P 500 종목 목록 로딩...")
    try:
        sp500 = get_sp500_list()
        symbols = sp500['symbol'].tolist() + ['SPY']
    except Exception:
        from config import SP500_BACKUP
        symbols = SP500_BACKUP + ['SPY']
    print(f"종목 수: {len(symbols)}")

    # ----- [2] 학습 데이터 다운로드 -----
    print(f"\n[2] 학습 데이터 다운로드: {args.train_start} ~ {args.train_end}")
    train_raw = get_backtest_data(symbols, start_date=args.train_start, end_date=args.train_end)
    print(f"학습 데이터 rows: {len(train_raw):,}")

    # ----- [3] 백테스트 데이터 다운로드 -----
    print(f"\n[3] 백테스트 데이터 다운로드: {args.backtest_start} ~ {backtest_end}")
    test_raw = get_backtest_data(symbols, start_date=args.backtest_start, end_date=backtest_end)
    print(f"백테스트 데이터 rows: {len(test_raw):,}")

    # ----- [4] 피처 생성 -----
    print("\n[4] 피처 생성 중...")
    train_features = create_features(train_raw)
    test_features = create_features(test_raw)
    feature_cols = get_feature_columns(train_features)
    print(f"피처 수: {len(feature_cols)}")
    print(f"학습 샘플: {len(train_features):,}  /  테스트 샘플: {len(test_features):,}")

    # ----- [5] 모멘텀용 가격 데이터 -----
    price_df = prepare_price_data(test_raw)

    # ----- [6] 실험 실행 -----
    experiments = get_experiments()
    results = []

    for i, exp in enumerate(experiments):
        print(f"\n[5-{i}] 실험 '{exp['name']}' 실행 중... ({exp['desc']})")
        try:
            r = run_single_experiment(
                exp, train_features, test_features, feature_cols,
                price_df, test_raw, args
            )
            results.append(r)
            m = r['metrics']
            print(f"      → 수익률: {m['total_return']*100:+.1f}%  "
                  f"CAGR: {m['cagr']*100:.1f}%  "
                  f"Sharpe: {m['sharpe_ratio']:.2f}  "
                  f"MDD: {m['mdd']*100:.1f}%")
        except Exception as e:
            print(f"      ✗ 실패: {e}")

    if not results:
        print("실험 결과가 없습니다. 종료합니다.")
        return

    # ----- [7] 결과 출력 -----
    print_results_table(results, args, backtest_end)

    # ----- [8] CSV 저장 -----
    save_results_csv(results, args.results_csv)

    # ----- [9] 차트 저장 -----
    plot_results(results, test_raw, args.output, args, backtest_end)


if __name__ == '__main__':
    main()
