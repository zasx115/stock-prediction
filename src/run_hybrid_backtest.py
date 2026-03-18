#!/usr/bin/env python3
# ============================================
# 파일명: src/run_hybrid_backtest.py
# 설명: 하이브리드 전략 백테스트 실행기 (CLI)
#
# 핵심: 학습기간(train_start~train_end)과
#       백테스트기간(backtest_start~backtest_end)을 독립적으로 설정 가능
#
# 사용법:
#   python run_hybrid_backtest.py \
#     --train_start 2015-01-01 \
#     --train_end 2020-01-01 \
#     --backtest_start 2020-01-01 \
#     --backtest_end 2024-12-31 \
#     --initial_capital 10000 \
#     --momentum_weight 0.35 \
#     --ai_weight 0.65 \
#     --commission 0.001 \
#     --slippage 0.001 \
#     --output backtest_result_hybrid.png
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
from ai_data import create_features, get_feature_columns
from hybrid_strategy import HybridStrategy
from ai_backtest import run_ai_backtest
from strategy import prepare_price_data


def parse_args():
    parser = argparse.ArgumentParser(description='하이브리드 전략 백테스트')
    parser.add_argument('--train_start', type=str, default='2015-01-01',
                        help='AI 학습 시작일 (YYYY-MM-DD)')
    parser.add_argument('--train_end', type=str, default='2020-01-01',
                        help='AI 학습 종료일 (YYYY-MM-DD)')
    parser.add_argument('--backtest_start', type=str, default='2020-01-01',
                        help='백테스트(시뮬레이션) 시작일 (YYYY-MM-DD)')
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
    parser.add_argument('--output', type=str, default='backtest_result_hybrid.png',
                        help='그래프 저장 경로')
    return parser.parse_args()


def print_results(metrics, args, backtest_end):
    """텍스트 메트릭 출력"""
    calmar = abs(metrics['cagr'] / metrics['mdd']) if metrics['mdd'] != 0 else 0
    net_cost = metrics['total_commission'] + metrics.get('total_slippage', 0)

    print()
    print("=" * 55)
    print("   Hybrid Backtest Result")
    print("=" * 55)
    print(f"[기간 설정]")
    print(f"학습기간       : {args.train_start} ~ {args.train_end}")
    print(f"백테스트기간   : {args.backtest_start} ~ {backtest_end}")
    print(f"Initial Capital: ${args.initial_capital:,.2f}")
    print(f"Weights        : Momentum {args.momentum_weight*100:.0f}% / AI {args.ai_weight*100:.0f}%")
    print(f"Commission     : {args.commission*100:.2f}%")
    print(f"Slippage       : {args.slippage*100:.2f}%")
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


def plot_results(portfolio_df, trades_df, test_df, output_path, args, backtest_end):
    """3 subplot 그래프 생성 및 저장"""
    fig, axes = plt.subplots(3, 1, figsize=(14, 14))
    title = (f"Hybrid Strategy Backtest\n"
             f"Train: {args.train_start}~{args.train_end}  "
             f"| Backtest: {args.backtest_start}~{backtest_end}  "
             f"| Momentum {args.momentum_weight*100:.0f}% / AI {args.ai_weight*100:.0f}%")
    fig.suptitle(title, fontsize=13, fontweight='bold')

    portfolio_df = portfolio_df.copy()
    portfolio_df['date'] = pd.to_datetime(portfolio_df['date'])
    portfolio_df['normalized'] = portfolio_df['value'] / portfolio_df['value'].iloc[0] * 100

    # ----- Subplot 1: 누적 수익률 + SPY + 매매 시점 -----
    ax1 = axes[0]
    ax1.plot(portfolio_df['date'], portfolio_df['normalized'],
             label='Hybrid Portfolio', linewidth=2, color='steelblue')

    if 'SPY' in test_df['symbol'].unique():
        spy = test_df[test_df['symbol'] == 'SPY'].sort_values('date').copy()
        spy['date'] = pd.to_datetime(spy['date'])
        spy['normalized'] = spy['close'] / spy['close'].iloc[0] * 100
        ax1.plot(spy['date'], spy['normalized'],
                 label='SPY', linewidth=2, linestyle='--', color='orange', alpha=0.8)

    if not trades_df.empty:
        trades_df = trades_df.copy()
        trades_df['date'] = pd.to_datetime(trades_df['date'])

        # 매수 시점
        buy_trades = trades_df[trades_df['action'].isin(['BUY', 'ADD'])]
        if not buy_trades.empty:
            port_idx = portfolio_df.set_index('date')['normalized']
            buy_values = port_idx.reindex(buy_trades['date'].values).values
            ax1.scatter(buy_trades['date'].values, buy_values,
                        color='royalblue', marker='o', s=30, alpha=0.7,
                        label='Buy', zorder=5)

        # 손절 시점
        stop_trades = trades_df[trades_df['action'] == 'STOP_LOSS']
        if not stop_trades.empty:
            port_idx = portfolio_df.set_index('date')['normalized']
            stop_values = port_idx.reindex(stop_trades['date'].values).values
            ax1.scatter(stop_trades['date'].values, stop_values,
                        color='crimson', marker='o', s=50, alpha=0.9,
                        label='Stop Loss', zorder=6)

    ax1.set_title('Cumulative Return vs SPY', fontsize=12)
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

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nGraph saved: {output_path}")


def main():
    args = parse_args()

    backtest_end = args.backtest_end if args.backtest_end else datetime.now().strftime('%Y-%m-%d')

    # 가중치 합 검증
    total_weight = args.momentum_weight + args.ai_weight
    if abs(total_weight - 1.0) > 0.01:
        print(f"경고: momentum_weight({args.momentum_weight}) + ai_weight({args.ai_weight}) = {total_weight:.2f} (합계가 1.0이 아님)")

    print("=" * 55)
    print("[하이브리드 백테스트 설정]")
    print(f"학습기간   : {args.train_start} ~ {args.train_end}")
    print(f"백테스트   : {args.backtest_start} ~ {backtest_end}")
    print(f"가중치     : Momentum {args.momentum_weight*100:.0f}% / AI {args.ai_weight*100:.0f}%")
    print("=" * 55)

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

    # ----- [5] 가격 데이터 (백테스트 모멘텀용) -----
    price_df = prepare_price_data(test_raw)

    # ----- [6] 하이브리드 전략 학습 -----
    print(f"\n[5] 하이브리드 전략 준비 (AI 학습)...")
    strategy = HybridStrategy(
        weight_momentum=args.momentum_weight,
        weight_ai=args.ai_weight
    )
    strategy.prepare(train_features, price_df, feature_cols)

    # ----- [7] 백테스트 실행 -----
    print(f"\n[6] 백테스트 시뮬레이션 실행...")
    result = run_ai_backtest(
        strategy=strategy,
        test_df=test_features,
        feature_cols=feature_cols,
        initial_capital=args.initial_capital,
        commission=args.commission,
        slippage=args.slippage
    )

    portfolio_df = result['portfolio']
    trades_df = result['trades']
    metrics = result['metrics']

    # SPY 수익률 보정: create_features()는 SPY를 제외하므로 test_raw에서 직접 계산
    if 'SPY' in test_raw['symbol'].unique():
        spy = test_raw[test_raw['symbol'] == 'SPY'].sort_values('date')
        if len(spy) >= 2:
            spy_return = (spy.iloc[-1]['close'] - spy.iloc[0]['close']) / spy.iloc[0]['close']
            metrics['spy_return'] = spy_return
            metrics['alpha'] = metrics['total_return'] - spy_return

    print_results(metrics, args, backtest_end)
    plot_results(portfolio_df, trades_df, test_raw, args.output, args, backtest_end)


if __name__ == '__main__':
    main()
