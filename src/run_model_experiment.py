#!/usr/bin/env python3
# ============================================
# 파일명: src/run_model_experiment.py
# 설명: XGBoost 하이퍼파라미터 실험 비교 백테스트 (Walk-Forward Sliding Window)
#
# 역할 요약:
#   다양한 XGBoost 파라미터 조합으로 HybridStrategy를 학습하고
#   Walk-Forward(Sliding Window) 방식으로 각 실험의 백테스트 성과를 비교.
#
# Walk-Forward 방식 (Sliding Window, 스텝 3개월):
#   Fold 1: Train [train_start ~ train_end]            → Test [train_end ~ +3M]
#   Fold 2: Train [train_start+3M ~ train_end+3M]      → Test [train_end+3M ~ +6M]
#   Fold 3: Train [train_start+6M ~ train_end+6M]      → Test [train_end+6M ~ +9M]
#   ... backtest_end 까지 반복
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
#   2. 전체 기간 데이터 한 번에 다운로드 (train_start ~ backtest_end)
#   3. 전체 피처 한 번에 생성 (create_features)
#   4. Walk-Forward 폴드 생성 (스텝: wf_step_months개월)
#   5. 폴드별 루프:
#      - 학습/테스트 데이터 슬라이싱
#      - 각 실험별 HybridStrategy 재학습 → run_ai_backtest 시뮬레이션
#      - 폴드 결과 수집
#   6. 실험별 포트폴리오 체이닝 → 전체 기간 메트릭 계산
#   7. 결과 DataFrame 생성 → CSV 저장 + 텍스트 출력
#   8. 4-subplot 비교 차트 생성 및 PNG 저장 (폴드 경계선 포함)
#
# 사용법:
#   python run_model_experiment.py
#   python run_model_experiment.py \
#     --train_start 2015-01-01 \
#     --train_end 2020-01-01 \
#     --backtest_start 2020-01-01 \
#     --backtest_end 2024-12-31 \
#     --wf_step_months 3 \
#     --initial_capital 10000 \
#     --output experiment_result.png \
#     --results_csv experiment_results.csv
#
# 의존 관계:
#   ← data.py (get_backtest_data, get_sp500_list)
#   ← ai_data.py (create_features, get_feature_columns)
#   ← hybrid_strategy.py (HybridStrategy)
#   ← ai_backtest.py (run_ai_backtest, calculate_ai_metrics)
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
from ai_backtest import run_ai_backtest, calculate_ai_metrics
from strategy import prepare_price_data, CustomStrategy, filter_tuesday
from ai_strategy import AIStrategy, XGB_PARAMS


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
# [2] Walk-Forward 폴드 생성
# ============================================

def generate_walk_forward_folds(train_start, train_end, backtest_end, step_months=3):
    """
    Sliding Window Walk-Forward 폴드 목록 생성.

    학습 윈도우 크기(train_end - train_start)를 고정하고,
    매 스텝마다 train_start와 train_end를 step_months씩 슬라이딩.
    테스트 윈도우는 항상 step_months개월.

    Args:
        train_start (str): 초기 학습 시작일 (YYYY-MM-DD)
        train_end (str): 초기 학습 종료일 = 첫 테스트 시작일
        backtest_end (str): 전체 백테스트 종료일
        step_months (int): 테스트 윈도우 크기 (개월)

    Returns:
        list of dict: [{'train_start', 'train_end', 'test_start', 'test_end'}, ...]
    """
    folds = []
    current_train_start = pd.Timestamp(train_start)
    test_start = pd.Timestamp(train_end)
    end = pd.Timestamp(backtest_end)

    while test_start < end:
        test_end = test_start + pd.DateOffset(months=step_months)
        if test_end > end:
            test_end = end

        folds.append({
            'train_start': current_train_start.strftime('%Y-%m-%d'),
            'train_end': test_start.strftime('%Y-%m-%d'),
            'test_start': test_start.strftime('%Y-%m-%d'),
            'test_end': test_end.strftime('%Y-%m-%d'),
        })

        current_train_start += pd.DateOffset(months=step_months)
        test_start = test_end

    return folds


# ============================================
# [3] 포트폴리오 체이닝 (Walk-Forward 결합)
# ============================================

def stitch_portfolios(portfolio_dfs, initial_capital):
    """
    여러 폴드의 포트폴리오를 하나의 연속 시계열로 연결.

    각 폴드는 initial_capital에서 시작하므로, 이전 폴드의 최종 가치를 기준으로 스케일링.

    Fold 1: [10000 → V1]           (배율 1.0)
    Fold 2: [10000 → V2] × V1/C   = [V1 → V2*(V1/C)]
    Fold 3: [10000 → V3] × V2/C   = [V2 → V3*(V2/C)]
    ...

    Args:
        portfolio_dfs (list of pd.DataFrame): 각 폴드의 portfolio DataFrame
        initial_capital (float): 각 폴드의 시작 자본 (동일)

    Returns:
        pd.DataFrame: 연결된 포트폴리오 시계열
    """
    if not portfolio_dfs:
        return pd.DataFrame()

    stitched = []
    prev_end_value = initial_capital

    for df in portfolio_dfs:
        df = df.copy()
        multiplier = prev_end_value / initial_capital

        for col in ['value', 'cash', 'stock_value']:
            if col in df.columns:
                df[col] = df[col] * multiplier

        stitched.append(df)
        prev_end_value = df['value'].iloc[-1]

    result = pd.concat(stitched, ignore_index=True)
    result['date'] = pd.to_datetime(result['date'])
    result = result.sort_values('date').reset_index(drop=True)
    return result


# ============================================
# [4] 단일 폴드 단일 실험 실행
# ============================================

def run_single_experiment(exp, train_features, test_features, feature_cols,
                          price_df, test_raw, args):
    """
    하나의 실험(파라미터 조합)에 대해 HybridStrategy를 학습하고 백테스트를 실행.

    hybrid_strategy.py는 수정하지 않고, AIStrategy를 직접 생성·주입하는 방식으로
    실험별 XGBoost 파라미터를 적용.

    Args:
        exp (dict): {'name', 'desc', 'params'}
        train_features (pd.DataFrame): 학습용 피처 DataFrame
        test_features (pd.DataFrame): 테스트용 피처 DataFrame
        feature_cols (list): 피처 컬럼 목록
        price_df (pd.DataFrame): 모멘텀 전략용 가격 데이터 (피벗)
        test_raw (pd.DataFrame): 원시 테스트 데이터 (SPY 수익률 계산용)
        args: argparse Namespace

    Returns:
        dict: 실험 이름, 설명, 메트릭, portfolio_df, trades_df
    """
    # [1] 실험별 파라미터로 AIStrategy 직접 생성 및 학습
    ai = AIStrategy(params=exp['params'])
    ai.train(train_features, feature_cols)

    # [2] HybridStrategy 초기화 (hybrid_strategy.py 수정 없이)
    strategy = HybridStrategy(
        weight_momentum=args.momentum_weight,
        weight_ai=args.ai_weight,
    )
    strategy.feature_cols = feature_cols

    # [3] 학습된 AIStrategy 주입
    strategy.ai_strategy = ai

    # [4] 모멘텀 전략 준비 (prepare()의 모멘텀 부분만 수행)
    strategy.momentum_strategy = CustomStrategy()
    tuesday_df = filter_tuesday(price_df)
    strategy.score_df, strategy.correlation_df, strategy.ret_1m = \
        strategy.momentum_strategy.prepare(price_df, tuesday_df)
    strategy.is_prepared = True

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
    trades_df = result['trades']

    # SPY 수익률 보정
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
        'trades': trades_df,
    }


# ============================================
# [5] 결과 출력
# ============================================

def print_results_table(results, args, backtest_end, folds):
    """실험 결과 텍스트 테이블 출력"""
    print()
    print("=" * 75)
    print("   Model Experiment Results (Walk-Forward)")
    print("=" * 75)
    print(f"학습 초기   : {args.train_start} ~ {args.train_end}")
    print(f"백테스트    : {args.train_end} ~ {backtest_end}")
    print(f"WF 스텝     : {args.wf_step_months}개월  /  폴드 수: {len(folds)}개")
    print(f"초기자본    : ${args.initial_capital:,.2f}")
    print(f"가중치      : Momentum {args.momentum_weight*100:.0f}% / AI {args.ai_weight*100:.0f}%")
    print(f"수수료/슬립 : {args.commission*100:.2f}% / {args.slippage*100:.2f}%")
    print()

    header = f"{'실험명':<12} {'설명':<32} {'수익률':>8} {'CAGR':>7} {'샤프':>6} {'MDD':>7} {'Alpha':>7} {'거래':>5} {'승률':>6}"
    print(header)
    print("-" * 95)

    for r in results:
        m = r['metrics']
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

    best_cagr = max(results, key=lambda r: r['metrics']['cagr'])
    best_sharpe = max(results, key=lambda r: r['metrics']['sharpe_ratio'])
    best_mdd = min(results, key=lambda r: r['metrics']['mdd'])

    print()
    print(f"  CAGR 최고  : {best_cagr['name']} ({best_cagr['metrics']['cagr']*100:.1f}%)")
    print(f"  Sharpe 최고: {best_sharpe['name']} ({best_sharpe['metrics']['sharpe_ratio']:.2f})")
    print(f"  MDD 최소   : {best_mdd['name']} ({best_mdd['metrics']['mdd']*100:.1f}%)")
    print()


# ============================================
# [6] 결과 CSV 저장
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
# [7] 비교 차트 생성
# ============================================

def plot_results(results, spy_raw, output_path, args, backtest_end, folds):
    """
    4-subplot 비교 차트 생성 및 저장 (폴드 경계선 포함).

    Subplot 1: 누적 수익률 곡선 (전략별 + SPY + 폴드 경계)
    Subplot 2: CAGR 막대 차트
    Subplot 3: 샤프 비율 막대 차트
    Subplot 4: MDD 막대 차트
    """
    fig, axes = plt.subplots(4, 1, figsize=(14, 20))
    title = (
        f"Model Experiment: Walk-Forward (Step={args.wf_step_months}M, Folds={len(folds)})\n"
        f"Train Init: {args.train_start}~{args.train_end}  "
        f"| Backtest: {args.train_end}~{backtest_end}  "
        f"| Momentum {args.momentum_weight*100:.0f}% / AI {args.ai_weight*100:.0f}%"
    )
    fig.suptitle(title, fontsize=12, fontweight='bold')

    names = [r['name'] for r in results]
    colors = plt.cm.tab10(np.linspace(0, 1, len(results)))

    # 폴드 경계 날짜 (두 번째 폴드부터 test_start)
    fold_boundaries = [pd.Timestamp(f['test_start']) for f in folds[1:]]

    # ----- Subplot 1: 누적 수익률 -----
    ax1 = axes[0]

    for i, r in enumerate(results):
        port = r['portfolio'].copy()
        port['date'] = pd.to_datetime(port['date'])
        port['normalized'] = port['value'] / port['value'].iloc[0] * 100
        ax1.plot(port['date'], port['normalized'],
                 label=r['name'], linewidth=1.5, color=colors[i], alpha=0.85)

    # SPY 기준선
    if 'SPY' in spy_raw['symbol'].unique():
        spy = spy_raw[spy_raw['symbol'] == 'SPY'].sort_values('date').copy()
        spy['date'] = pd.to_datetime(spy['date'])
        spy['normalized'] = spy['close'] / spy['close'].iloc[0] * 100
        ax1.plot(spy['date'], spy['normalized'],
                 label='SPY', linewidth=2.5, linestyle='--',
                 color='black', alpha=0.6)

    # 폴드 경계 수직선
    for boundary in fold_boundaries:
        ax1.axvline(x=boundary, color='gray', linewidth=0.8, linestyle=':', alpha=0.7)

    ax1.set_title(f'Cumulative Return (base=100) | 수직선: WF 폴드 경계 ({len(folds)}개)', fontsize=11)
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

    if results:
        spy_cagr = results[0]['metrics'].get('spy_return', 0)
        if spy_cagr != 0:
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
# [8] CLI 파싱
# ============================================

def parse_args():
    parser = argparse.ArgumentParser(description='XGBoost 하이퍼파라미터 실험 비교 (Walk-Forward)')
    parser.add_argument('--train_start', type=str, default='2015-01-01',
                        help='AI 학습 시작일 (YYYY-MM-DD)')
    parser.add_argument('--train_end', type=str, default='2020-01-01',
                        help='초기 학습 종료일 = 첫 번째 테스트 시작일 (YYYY-MM-DD)')
    parser.add_argument('--backtest_start', type=str, default='2020-01-01',
                        help='(미사용, train_end와 동일하게 처리됨. 하위 호환용)')
    parser.add_argument('--backtest_end', type=str, default='',
                        help='백테스트 종료일 (YYYY-MM-DD, 빈값=오늘)')
    parser.add_argument('--wf_step_months', type=int, default=3,
                        help='Walk-Forward 테스트 윈도우 크기 (개월, 기본 3)')
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
# [9] 메인
# ============================================

def main():
    args = parse_args()

    backtest_end = args.backtest_end if args.backtest_end else datetime.now().strftime('%Y-%m-%d')

    # 가중치 합 검증
    total_weight = args.momentum_weight + args.ai_weight
    if abs(total_weight - 1.0) > 0.01:
        print(f"경고: momentum_weight({args.momentum_weight}) + "
              f"ai_weight({args.ai_weight}) = {total_weight:.2f} (합계가 1.0이 아님)")

    # ----- Walk-Forward 폴드 생성 -----
    folds = generate_walk_forward_folds(
        args.train_start, args.train_end, backtest_end, args.wf_step_months
    )

    print("=" * 60)
    print("[Model Experiment (Walk-Forward) 설정]")
    print(f"학습 초기   : {args.train_start} ~ {args.train_end}")
    print(f"백테스트    : {args.train_end} ~ {backtest_end}")
    print(f"WF 스텝     : {args.wf_step_months}개월  /  폴드 수: {len(folds)}개")
    print(f"가중치      : Momentum {args.momentum_weight*100:.0f}% / AI {args.ai_weight*100:.0f}%")
    print(f"실험 수     : {len(get_experiments())}개")
    print("=" * 60)

    if not folds:
        print("생성된 폴드가 없습니다. train_end와 backtest_end를 확인하세요.")
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

    # ----- [2] 전체 기간 데이터 한 번에 다운로드 -----
    print(f"\n[2] 전체 데이터 다운로드: {args.train_start} ~ {backtest_end}")
    all_raw = get_backtest_data(symbols, start_date=args.train_start, end_date=backtest_end)
    all_raw['date'] = pd.to_datetime(all_raw['date'])
    print(f"전체 데이터 rows: {len(all_raw):,}")

    # ----- [3] 전체 피처 한 번에 생성 -----
    print("\n[3] 피처 생성 중...")
    all_features = create_features(all_raw)
    all_features['date'] = pd.to_datetime(all_features['date'])
    feature_cols = get_feature_columns(all_features)
    print(f"피처 수: {len(feature_cols)}  /  전체 샘플: {len(all_features):,}")

    # ----- [4] Walk-Forward 폴드별 실험 실행 -----
    experiments = get_experiments()

    # exp_name → {'desc': str, 'folds': [{'portfolio': df, 'trades': df}, ...]}
    exp_fold_data = {
        exp['name']: {'desc': exp['desc'], 'folds': []}
        for exp in experiments
    }

    for fold_idx, fold in enumerate(folds):
        fold_train_start = pd.Timestamp(fold['train_start'])
        fold_train_end = pd.Timestamp(fold['train_end'])
        fold_test_end = pd.Timestamp(fold['test_end'])

        print(f"\n{'='*60}")
        print(f"[Fold {fold_idx+1}/{len(folds)}]  "
              f"Train: {fold['train_start']}~{fold['train_end']}  "
              f"| Test: {fold['test_start']}~{fold['test_end']}")
        print(f"{'='*60}")

        # 폴드별 데이터 슬라이싱 (Sliding Window: train_start ~ train_end)
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

        # 모멘텀 점수 계산에 과거 7개월 데이터 필요 (pct_change(24) = 화요일 24주 ≈ 6개월)
        momentum_lookback = fold_train_end - pd.DateOffset(months=7)
        price_raw_fold = all_raw[
            (all_raw['date'] >= momentum_lookback) &
            (all_raw['date'] <= fold_test_end)
        ]
        price_df_fold = prepare_price_data(price_raw_fold)

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

    # ----- [5] 폴드 결과 집계 (포트폴리오 체이닝 + 전체 메트릭 계산) -----
    print(f"\n\n[집계] Walk-Forward 결과 통합 중...")

    # 전체 테스트 기간 SPY 데이터
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

        # 포트폴리오 체이닝
        stitched_portfolio = stitch_portfolios(
            [f['portfolio'] for f in fold_list],
            args.initial_capital
        )

        # 거래 내역 합산
        trade_dfs = [f['trades'] for f in fold_list if not f['trades'].empty]
        all_trades = pd.concat(trade_dfs, ignore_index=True) if trade_dfs else pd.DataFrame()

        # 전체 기간 메트릭 재계산
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

        print(f"  {exp['name']}: 체이닝 완료  "
              f"수익률={metrics['total_return']*100:+.1f}%  "
              f"CAGR={metrics['cagr']*100:.1f}%  "
              f"Sharpe={metrics['sharpe_ratio']:.2f}")

    if not aggregated_results:
        print("집계된 실험 결과가 없습니다. 종료합니다.")
        return

    # ----- [6] 결과 출력 -----
    print_results_table(aggregated_results, args, backtest_end, folds)

    # ----- [7] CSV 저장 -----
    save_results_csv(aggregated_results, args.results_csv)

    # ----- [8] 차트 저장 -----
    plot_results(aggregated_results, all_raw, args.output, args, backtest_end, folds)


if __name__ == '__main__':
    main()
