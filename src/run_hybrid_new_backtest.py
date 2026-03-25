#!/usr/bin/env python3
# ============================================
# 파일명: src/run_hybrid_new_backtest.py
# 설명: 하이브리드_New 전략 백테스트 CLI 실행기 (Walk-Forward Sliding Window)
#
# 하이브리드_New 최적 파라미터:
#   - XGBoost: max_depth=3, subsample=0.5, colsample_bytree=0.5
#   - 라벨: 5D-3% (TARGET_RETURN_OPTIMIZED=0.03)
#   - 가중치: 모멘텀 50% + AI 50%
#   - Walk-Forward: 6개월 슬라이딩 윈도우
#
# Walk-Forward 방식 (Sliding Window):
#   Fold 1: Train [train_start ~ train_end]            → Test [train_end ~ +6M]
#   Fold 2: Train [train_start+6M ~ train_end+6M]      → Test [train_end+6M ~ +12M]
#   ... backtest_end 까지 반복
#
# 사용법:
#   python run_hybrid_new_backtest.py
#   python run_hybrid_new_backtest.py \
#     --train_start 2015-01-01 \
#     --train_end 2020-01-01 \
#     --backtest_end 2024-12-31 \
#     --wf_step_months 6 \
#     --momentum_weight 0.50 \
#     --ai_weight 0.50
#
# 의존 관계:
#   ← data.py (get_backtest_data, get_sp500_list)
#   ← ai_data.py (create_features, get_feature_columns, TARGET_RETURN_OPTIMIZED)
#   ← hybrid_strategy.py (HybridStrategy)
#   ← ai_strategy.py (AIStrategy, XGB_PARAMS_OPTIMIZED)
#   ← ai_backtest.py (run_ai_backtest, calculate_ai_metrics)
#   ← strategy.py (prepare_price_data, CustomStrategy, filter_tuesday)
#   ← run_model_experiment.py (generate_walk_forward_folds, stitch_portfolios)
# ============================================

import sys
import os
import argparse
from datetime import datetime

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

sys.path.insert(0, os.path.dirname(__file__))

from data import get_backtest_data, get_sp500_list
from ai_data import create_features, get_feature_columns, TARGET_RETURN_OPTIMIZED
from hybrid_strategy import HybridStrategy
from ai_strategy import AIStrategy, XGB_PARAMS_OPTIMIZED
from ai_backtest import run_ai_backtest, calculate_ai_metrics
from strategy import prepare_price_data, CustomStrategy, filter_tuesday
from run_model_experiment import generate_walk_forward_folds, stitch_portfolios


def parse_args():
    parser = argparse.ArgumentParser(description='하이브리드_New 전략 백테스트 (Walk-Forward)')
    parser.add_argument('--train_start', type=str, default='2015-01-01',
                        help='AI 학습 시작일 (YYYY-MM-DD)')
    parser.add_argument('--train_end', type=str, default='2020-01-01',
                        help='AI 학습 종료일 = 첫 번째 테스트 시작일 (YYYY-MM-DD)')
    parser.add_argument('--backtest_start', type=str, default='2020-01-01',
                        help='(하위 호환용, train_end와 동일하게 처리)')
    parser.add_argument('--backtest_end', type=str, default='',
                        help='백테스트 종료일 (YYYY-MM-DD, 빈값=오늘)')
    parser.add_argument('--wf_step_months', type=int, default=6,
                        help='Walk-Forward 테스트 윈도우 크기 (개월, 기본 6)')
    parser.add_argument('--initial_capital', type=float, default=10000,
                        help='초기 자본금 (USD)')
    parser.add_argument('--momentum_weight', type=float, default=0.50,
                        help='모멘텀 가중치 (기본 0.50)')
    parser.add_argument('--ai_weight', type=float, default=0.50,
                        help='AI 가중치 (기본 0.50)')
    parser.add_argument('--commission', type=float, default=0.001,
                        help='수수료율 (기본 0.001 = 0.1%%)')
    parser.add_argument('--slippage', type=float, default=0.001,
                        help='슬리피지율 (기본 0.001 = 0.1%%)')
    parser.add_argument('--output', type=str, default='backtest_result_hybrid_new.png',
                        help='그래프 저장 경로')
    return parser.parse_args()


def print_results(metrics, args, backtest_end, folds):
    """텍스트 메트릭 출력"""
    calmar = abs(metrics['cagr'] / metrics['mdd']) if metrics['mdd'] != 0 else 0
    net_cost = metrics['total_commission'] + metrics.get('total_slippage', 0)

    print()
    print("=" * 55)
    print("   Hybrid_New Backtest Result (Walk-Forward)")
    print("=" * 55)
    print(f"[기간 설정]")
    print(f"학습 초기      : {args.train_start} ~ {args.train_end}")
    print(f"백테스트       : {args.train_end} ~ {backtest_end}")
    print(f"WF 스텝        : {args.wf_step_months}개월  /  폴드 수: {len(folds)}개")
    print(f"Initial Capital: ${args.initial_capital:,.2f}")
    print(f"Weights        : Momentum {args.momentum_weight*100:.0f}% / AI {args.ai_weight*100:.0f}%")
    print(f"Commission     : {args.commission*100:.2f}%")
    print(f"Slippage       : {args.slippage*100:.2f}%")
    print()
    print(f"[Hybrid_New 파라미터]")
    print(f"XGBoost        : max_depth=3, subsample=0.5, colsample=0.5")
    print(f"라벨           : 5D-{TARGET_RETURN_OPTIMIZED*100:.0f}%")
    print()
    print("[수익률]")
    print(f"Total Return   : {metrics['total_return']*100:+.2f}%")
    print(f"SPY Return     : {metrics['spy_return']*100:+.2f}%")
    print(f"Alpha          : {metrics['alpha']*100:+.2f}%")
    print(f"CAGR           : {metrics['cagr']*100:.2f}%")
    print()
    print("[리스크]")
    print(f"Sharpe Ratio   : {metrics['sharpe_ratio']:.2f}")
    print(f"Calmar Ratio   : {calmar:.2f}")
    print(f"Max Drawdown   : {metrics['mdd']*100:.2f}%")
    print(f"Volatility     : {metrics['volatility']*100:.2f}%")
    print()
    print("[거래]")
    print(f"Total Trades   : {metrics['total_trades']}")
    print(f"Win Rate       : {metrics['win_rate']*100:.2f}%")
    print(f"Stop Loss      : {metrics['stop_loss_count']}")
    print(f"Total Commission: ${metrics['total_commission']:,.2f}")
    print(f"Total Slippage  : ${metrics.get('total_slippage', 0):,.2f}")
    print(f"Net Cost        : ${net_cost:,.2f}")
    print("=" * 55)


def plot_results(portfolio_df, trades_df, spy_raw, output_path, args, backtest_end, folds):
    """4-subplot 그래프 생성 및 저장 (폴드 경계선 포함)"""
    fig, axes = plt.subplots(4, 1, figsize=(14, 20))
    title = (
        f"Hybrid_New Strategy Backtest (Walk-Forward)\n"
        f"Train Init: {args.train_start}~{args.train_end}  "
        f"| Backtest: {args.train_end}~{backtest_end}  "
        f"| WF Step: {args.wf_step_months}M  "
        f"| Momentum {args.momentum_weight*100:.0f}% / AI {args.ai_weight*100:.0f}%\n"
        f"XGB: depth=3, sample=0.5  |  Label: 5D-{TARGET_RETURN_OPTIMIZED*100:.0f}%"
    )
    fig.suptitle(title, fontsize=12, fontweight='bold')

    portfolio_df = portfolio_df.copy()
    portfolio_df['date'] = pd.to_datetime(portfolio_df['date'])
    portfolio_df['normalized'] = portfolio_df['value'] / portfolio_df['value'].iloc[0] * 100

    # 폴드 경계 날짜 (두 번째 폴드부터 test_start)
    fold_boundaries = [pd.Timestamp(f['test_start']) for f in folds[1:]]

    # ----- Subplot 1: 누적 수익률 + SPY + 폴드 경계 -----
    ax1 = axes[0]
    ax1.plot(portfolio_df['date'], portfolio_df['normalized'],
             label='Hybrid_New Portfolio', linewidth=2, color='steelblue')

    if 'SPY' in spy_raw['symbol'].unique():
        spy = spy_raw[spy_raw['symbol'] == 'SPY'].sort_values('date').copy()
        spy['date'] = pd.to_datetime(spy['date'])
        spy = spy[spy['date'] >= portfolio_df['date'].iloc[0]]
        if not spy.empty:
            spy['normalized'] = spy['close'] / spy['close'].iloc[0] * 100
            ax1.plot(spy['date'], spy['normalized'],
                     label='SPY', linewidth=2, linestyle='--', color='orange', alpha=0.8)

    for boundary in fold_boundaries:
        ax1.axvline(x=boundary, color='gray', linewidth=0.8, linestyle=':', alpha=0.7)

    if not trades_df.empty:
        trades_plot = trades_df.copy()
        trades_plot['date'] = pd.to_datetime(trades_plot['date'])
        port_idx = portfolio_df.set_index('date')['normalized']

        stop_trades = trades_plot[trades_plot['action'] == 'STOP_LOSS']
        if not stop_trades.empty:
            stop_values = port_idx.reindex(stop_trades['date'].values).values
            ax1.scatter(stop_trades['date'].values, stop_values,
                        color='crimson', marker='o', s=50, alpha=0.9,
                        label='Stop Loss', zorder=6)

    ax1.set_title(f'Cumulative Return vs SPY | WF 폴드 경계 ({len(folds)}개)', fontsize=12)
    ax1.set_ylabel('Value (base=100)')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))

    # ----- Subplot 2: 월별 수익률 바 차트 -----
    ax2 = axes[1]
    portfolio_df['month'] = portfolio_df['date'].dt.to_period('M')
    monthly = portfolio_df.groupby('month')['value'].last()
    monthly_returns = monthly.pct_change().dropna()

    colors = ['steelblue' if r >= 0 else 'crimson' for r in monthly_returns]
    x = range(len(monthly_returns))
    ax2.bar(x, monthly_returns * 100, color=colors, alpha=0.8)
    ax2.axhline(y=0, color='black', linewidth=0.8)
    ax2.set_title('Monthly Returns (%)', fontsize=12)
    ax2.set_ylabel('Return (%)')
    ax2.grid(True, alpha=0.3, axis='y')

    step = max(1, len(monthly_returns) // 12)
    ax2.set_xticks(list(x)[::step])
    ax2.set_xticklabels([str(m) for m in monthly_returns.index[::step]],
                        rotation=45, ha='right', fontsize=8)

    # ----- Subplot 3: Drawdown -----
    ax3 = axes[2]
    peak = portfolio_df['value'].cummax()
    drawdown = (portfolio_df['value'] - peak) / peak * 100
    ax3.fill_between(portfolio_df['date'], drawdown, 0, color='crimson', alpha=0.4)
    ax3.plot(portfolio_df['date'], drawdown, color='crimson', linewidth=1)
    ax3.set_title('Drawdown (%)', fontsize=12)
    ax3.set_ylabel('Drawdown (%)')
    ax3.grid(True, alpha=0.3)
    ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))

    # ----- Subplot 4: 폴드별 수익률 -----
    ax4 = axes[3]
    fold_returns = []
    fold_labels = []
    for i, f in enumerate(folds):
        fold_labels.append(f"F{i+1}\n{f['test_start'][:7]}")
    for i, f in enumerate(folds):
        ts = pd.Timestamp(f['test_start'])
        te = pd.Timestamp(f['test_end'])
        fold_port = portfolio_df[
            (portfolio_df['date'] >= ts) & (portfolio_df['date'] <= te)
        ]
        if len(fold_port) >= 2:
            ret = (fold_port['value'].iloc[-1] / fold_port['value'].iloc[0] - 1) * 100
            fold_returns.append(ret)
        else:
            fold_returns.append(0)

    bar_colors = ['steelblue' if r >= 0 else 'crimson' for r in fold_returns]
    bars = ax4.bar(fold_labels, fold_returns, color=bar_colors, alpha=0.8)
    ax4.axhline(y=0, color='black', linewidth=0.8)
    for bar, v in zip(bars, fold_returns):
        ax4.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                 f'{v:+.1f}%', ha='center',
                 va='bottom' if v >= 0 else 'top', fontsize=8)
    ax4.set_title('Return by Walk-Forward Fold (%)', fontsize=12)
    ax4.set_ylabel('Return (%)')
    ax4.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nGraph saved: {output_path}")


def main():
    args = parse_args()

    backtest_end = args.backtest_end if args.backtest_end else datetime.now().strftime('%Y-%m-%d')

    # 가중치 합 검증
    total_weight = args.momentum_weight + args.ai_weight
    if abs(total_weight - 1.0) > 0.01:
        print(f"경고: momentum_weight({args.momentum_weight}) + "
              f"ai_weight({args.ai_weight}) = {total_weight:.2f}")

    # ----- Walk-Forward 폴드 생성 -----
    folds = generate_walk_forward_folds(
        args.train_start, args.train_end, backtest_end, args.wf_step_months
    )

    print("=" * 60)
    print("[하이브리드_New 백테스트 설정 (Walk-Forward)]")
    print(f"학습 초기   : {args.train_start} ~ {args.train_end}")
    print(f"백테스트    : {args.train_end} ~ {backtest_end}")
    print(f"WF 스텝     : {args.wf_step_months}개월  /  폴드 수: {len(folds)}개")
    print(f"가중치      : Momentum {args.momentum_weight*100:.0f}% / AI {args.ai_weight*100:.0f}%")
    print(f"XGBoost     : max_depth=3, subsample=0.5, colsample=0.5")
    print(f"라벨        : 5D-{TARGET_RETURN_OPTIMIZED*100:.0f}%")
    print(f"수수료/슬립 : {args.commission*100:.2f}% / {args.slippage*100:.2f}%")
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

    # ----- [3] 전체 피처 한 번에 생성 (3% 라벨) -----
    print("\n[3] 피처 생성 중 (5D-3% 라벨)...")
    all_features = create_features(all_raw, target_return=TARGET_RETURN_OPTIMIZED)
    all_features['date'] = pd.to_datetime(all_features['date'])
    feature_cols = get_feature_columns(all_features)
    print(f"피처 수: {len(feature_cols)}  /  전체 샘플: {len(all_features):,}")

    # ----- [4] Walk-Forward 폴드별 실행 -----
    fold_portfolios = []
    fold_trades = []

    for fold_idx, fold in enumerate(folds):
        fold_train_start = pd.Timestamp(fold['train_start'])
        fold_train_end = pd.Timestamp(fold['train_end'])
        fold_test_end = pd.Timestamp(fold['test_end'])

        print(f"\n{'='*60}")
        print(f"[Fold {fold_idx+1}/{len(folds)}]  "
              f"Train: {fold['train_start']}~{fold['train_end']}  "
              f"| Test: {fold['test_start']}~{fold['test_end']}")
        print(f"{'='*60}")

        # 폴드별 데이터 슬라이싱
        train_features_fold = all_features[
            (all_features['date'] >= fold_train_start) &
            (all_features['date'] <= fold_train_end)
        ]
        test_features_fold = all_features[
            (all_features['date'] > fold_train_end) &
            (all_features['date'] <= fold_test_end)
        ]

        if len(test_features_fold) == 0:
            print(f"  경고: 테스트 데이터 없음, Fold {fold_idx+1} 스킵")
            continue

        print(f"학습 샘플: {len(train_features_fold):,}  /  테스트 샘플: {len(test_features_fold):,}")

        # 모멘텀 점수 계산에 과거 7개월 데이터 필요
        momentum_lookback = fold_train_end - pd.DateOffset(months=7)
        price_raw_fold = all_raw[
            (all_raw['date'] >= momentum_lookback) &
            (all_raw['date'] <= fold_test_end)
        ]
        price_df_fold = prepare_price_data(price_raw_fold)

        # HybridStrategy 학습 및 백테스트 (XGB_PARAMS_OPTIMIZED 사용)
        try:
            # AIStrategy를 최적 파라미터로 직접 생성 및 학습
            ai = AIStrategy(params=XGB_PARAMS_OPTIMIZED)
            ai.train(train_features_fold, feature_cols)

            # HybridStrategy 초기화 및 AI 주입
            strategy = HybridStrategy(
                weight_momentum=args.momentum_weight,
                weight_ai=args.ai_weight,
            )
            strategy.feature_cols = feature_cols
            strategy.ai_strategy = ai

            # 모멘텀 전략 준비
            strategy.momentum_strategy = CustomStrategy()
            tuesday_df = filter_tuesday(price_df_fold)
            strategy.score_df, strategy.correlation_df, strategy.ret_1m = \
                strategy.momentum_strategy.prepare(price_df_fold, tuesday_df)
            strategy.is_prepared = True

            result = run_ai_backtest(
                strategy=strategy,
                test_df=test_features_fold,
                feature_cols=feature_cols,
                initial_capital=args.initial_capital,
                commission=args.commission,
                slippage=args.slippage,
            )

            fold_portfolios.append(result['portfolio'])
            fold_trades.append(result['trades'])

            m = result['metrics']
            print(f"  → 수익률: {m['total_return']*100:+.1f}%  "
                  f"CAGR: {m['cagr']*100:.1f}%  "
                  f"Sharpe: {m['sharpe_ratio']:.2f}  "
                  f"MDD: {m['mdd']*100:.1f}%")
        except Exception as e:
            print(f"  ✗ Fold {fold_idx+1} 실패: {e}")

    if not fold_portfolios:
        print("유효한 폴드 결과가 없습니다. 종료합니다.")
        return

    # ----- [5] 폴드 결과 집계 -----
    print(f"\n\n[집계] Walk-Forward 결과 통합 중...")

    # 포트폴리오 체이닝
    stitched_portfolio = stitch_portfolios(fold_portfolios, args.initial_capital)

    # 거래 내역 합산
    trade_dfs = [t for t in fold_trades if not t.empty]
    all_trades_df = pd.concat(trade_dfs, ignore_index=True) if trade_dfs else pd.DataFrame()

    # SPY 데이터
    spy_raw = all_raw[
        (all_raw['symbol'] == 'SPY') &
        (all_raw['date'] > pd.Timestamp(args.train_end))
    ][['date', 'close']].rename(columns={'close': 'spy_close'})

    # 전체 기간 메트릭 계산
    metrics = calculate_ai_metrics(
        stitched_portfolio, all_trades_df, spy_raw,
        args.initial_capital, args.slippage
    )

    # ----- [6] 결과 출력 -----
    print_results(metrics, args, backtest_end, folds)

    # ----- [7] 그래프 저장 -----
    plot_results(stitched_portfolio, all_trades_df, all_raw, args.output, args, backtest_end, folds)


if __name__ == '__main__':
    main()
