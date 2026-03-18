#!/usr/bin/env python3
# ============================================
# 파일명: src/run_momentum_backtest.py
# 설명: 모멘텀 전략 백테스트 실행기 (CLI)
#
# 사용법:
#   python run_momentum_backtest.py \
#     --start_date 2020-01-01 \
#     --end_date 2024-12-31 \
#     --initial_capital 10000 \
#     --commission 0.001 \
#     --slippage 0.001 \
#     --output backtest_result_momentum.png
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

from data import get_backtest_data
from backtest import run_backtest


def parse_args():
    parser = argparse.ArgumentParser(description='모멘텀 전략 백테스트')
    parser.add_argument('--start_date', type=str, default='2020-01-01',
                        help='백테스트 시작일 (YYYY-MM-DD)')
    parser.add_argument('--end_date', type=str, default='',
                        help='백테스트 종료일 (YYYY-MM-DD, 빈값=오늘)')
    parser.add_argument('--initial_capital', type=float, default=10000,
                        help='초기 자본금 (USD)')
    parser.add_argument('--commission', type=float, default=0.001,
                        help='수수료율 (기본 0.001 = 0.1%%)')
    parser.add_argument('--slippage', type=float, default=0.001,
                        help='슬리피지율 (기본 0.001 = 0.1%%)')
    parser.add_argument('--output', type=str, default='backtest_result_momentum.png',
                        help='그래프 저장 경로')
    return parser.parse_args()


def print_results(metrics, start_date, end_date, args):
    """텍스트 메트릭 출력"""
    calmar = abs(metrics['cagr'] / metrics['mdd']) if metrics['mdd'] != 0 else 0
    net_cost = metrics['total_commission'] + metrics.get('total_slippage', 0)

    print()
    print("=" * 50)
    print("   Momentum Backtest Result")
    print("=" * 50)
    print(f"Period         : {start_date} ~ {end_date}")
    print(f"Initial Capital: ${args.initial_capital:,.2f}")
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
    print("=" * 50)


def plot_results(portfolio_df, trades_df, df, output_path):
    """3 subplot 그래프 생성 및 저장"""
    fig, axes = plt.subplots(3, 1, figsize=(14, 14))
    fig.suptitle('Momentum Strategy Backtest', fontsize=16, fontweight='bold')

    portfolio_df = portfolio_df.copy()
    portfolio_df['date'] = pd.to_datetime(portfolio_df['date'])
    portfolio_df['normalized'] = portfolio_df['value'] / portfolio_df['value'].iloc[0] * 100

    # ----- Subplot 1: 누적 수익률 + SPY + 매매 시점 -----
    ax1 = axes[0]
    ax1.plot(portfolio_df['date'], portfolio_df['normalized'],
             label='Momentum Portfolio', linewidth=2, color='steelblue')

    if 'SPY' in df['symbol'].unique():
        spy = df[df['symbol'] == 'SPY'].sort_values('date').copy()
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
            buy_values = portfolio_df.set_index('date').reindex(
                buy_trades['date'].values)['normalized'].values
            ax1.scatter(buy_trades['date'].values, buy_values,
                        color='royalblue', marker='o', s=30, alpha=0.7,
                        label='Buy', zorder=5)

        # 손절 시점
        stop_trades = trades_df[trades_df['action'] == 'STOP_LOSS']
        if not stop_trades.empty:
            stop_values = portfolio_df.set_index('date').reindex(
                stop_trades['date'].values)['normalized'].values
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

    end_date = args.end_date if args.end_date else datetime.now().strftime('%Y-%m-%d')

    print(f"데이터 다운로드 중: {args.start_date} ~ {end_date}")
    df = get_backtest_data(start_date=args.start_date, end_date=end_date)

    result = run_backtest(
        df=df,
        initial_capital=args.initial_capital,
        commission=args.commission,
        slippage=args.slippage
    )

    portfolio_df = result['portfolio']
    trades_df = result['trades']
    metrics = result['metrics']

    print_results(metrics, args.start_date, end_date, args)
    plot_results(portfolio_df, trades_df, df, args.output)


if __name__ == '__main__':
    main()
