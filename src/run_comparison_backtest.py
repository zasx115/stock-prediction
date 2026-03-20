#!/usr/bin/env python3
# ============================================
# 파일명: src/run_comparison_backtest.py
# 설명: 5가지 전략 비교 백테스트
#
# 전략 목록:
#   1. 모멘텀 전략          (2025-01-01 ~ 현재)
#   2. 하이브리드 전략       (학습: 2020-01-01~2024-12-31, 백테스트: 2025-01-01~현재)
#   3. 하이브리드 + Kelly    (동적 포지션 사이징)
#   4. 하이브리드 + Trailing (트레일링 스탑 -7%)
#   5. 하이브리드 + Kelly + Trailing
#   + SPY 벤치마크
#
# 실행:
#   python run_comparison_backtest.py
#   python run_comparison_backtest.py --output result.png
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
from strategy import CustomStrategy, prepare_price_data, filter_tuesday
from config import (
    INITIAL_CAPITAL, BUY_COMMISSION, SELL_COMMISSION, SLIPPAGE, SP500_BACKUP
)


# ============================================
# [설정]
# ============================================

MOMENTUM_START      = '2025-01-01'   # 모멘텀 백테스트 시작
MOMENTUM_WARMUP     = '2024-07-01'   # 모멘텀 워밍업 데이터 시작 (6개월 lookback)
HYBRID_TRAIN_START  = '2020-01-01'   # 하이브리드 AI 학습 시작
HYBRID_TRAIN_END    = '2024-12-31'   # 하이브리드 AI 학습 종료
HYBRID_TEST_START   = '2025-01-01'   # 하이브리드 백테스트 시작

STOP_LOSS_FIXED     = -0.07          # 기존 고정 손절 (-7%)
TRAILING_STOP_PCT   =  0.07          # 트레일링 스탑 (고점 대비 -7%)
KELLY_MIN_TRADES    = 20             # Kelly 계산에 필요한 최소 거래 수
KELLY_HALF          = True           # Half-Kelly 사용 여부 (보수적)


# ============================================
# [1] 공통 유틸리티
# ============================================

def get_spy_series(raw_df, start_date):
    """SPY 정규화 시계열 반환 (start_date 기준 100으로 정규화)"""
    spy = raw_df[raw_df['symbol'] == 'SPY'].copy()
    spy['date'] = pd.to_datetime(spy['date'])
    spy = spy[spy['date'] >= pd.Timestamp(start_date)].sort_values('date')
    if spy.empty:
        return pd.Series(dtype=float)
    spy['normalized'] = spy['close'] / spy['close'].iloc[0] * 100
    return spy.set_index('date')['normalized']


def calc_metrics(portfolio_df, spy_series, initial_capital):
    """성과 지표 계산"""
    values = portfolio_df['value'].values
    dates = portfolio_df['date']

    total_return = (values[-1] - values[0]) / values[0]
    daily_returns = pd.Series(values).pct_change().dropna()
    days = (dates.iloc[-1] - dates.iloc[0]).days
    years = max(days / 365, 0.01)
    cagr = (values[-1] / values[0]) ** (1 / years) - 1

    volatility = daily_returns.std() * np.sqrt(252)
    sharpe = (cagr - 0.03) / volatility if volatility > 0 else 0

    peak = pd.Series(values).cummax()
    mdd = ((pd.Series(values) - peak) / peak).min()

    spy_return = 0.0
    if not spy_series.empty:
        spy_return = (spy_series.iloc[-1] - spy_series.iloc[0]) / spy_series.iloc[0]

    return {
        'total_return': total_return,
        'cagr': cagr,
        'volatility': volatility,
        'sharpe': sharpe,
        'mdd': mdd,
        'spy_return': spy_return,
        'alpha': total_return - spy_return,
    }


def calc_kelly_allocations(completed_trades, n_picks=3):
    """
    Kelly Criterion으로 포지션 비중 계산

    공식: f* = (W * R - (1-W)) / R
      W = 승률 (수익 거래 / 전체 매도 거래)
      R = 평균이익 / 평균손실 (절댓값)

    n_picks개 종목에 Kelly 비중을 동일하게 적용 후 정규화.
    Half-Kelly 사용 시 f* / 2.
    """
    if len(completed_trades) < KELLY_MIN_TRADES:
        # 데이터 부족 → 기본 비중
        if n_picks == 1:
            return [1.0]
        elif n_picks == 2:
            return [0.5, 0.5]
        else:
            return [0.4, 0.3, 0.3]

    returns = [t['return_pct'] for t in completed_trades]
    wins = [r for r in returns if r > 0]
    losses = [abs(r) for r in returns if r < 0]

    if not wins or not losses:
        return [1.0 / n_picks] * n_picks

    W = len(wins) / len(returns)
    R = (sum(wins) / len(wins)) / (sum(losses) / len(losses))

    kelly_f = (W * R - (1 - W)) / R
    kelly_f = max(kelly_f, 0.0)  # 음수 → 0 (매수 안 함)

    if KELLY_HALF:
        kelly_f = kelly_f / 2

    # n_picks개 모두 동일한 kelly_f 적용 후 합산이 1 넘으면 정규화
    total = kelly_f * n_picks
    if total > 1.0:
        per_stock = kelly_f / total  # 정규화
    else:
        per_stock = kelly_f

    # 첫 종목에 더 비중 (kelly_f 기준 내림차순 균등 분배)
    if n_picks == 1:
        return [min(per_stock * n_picks, 1.0)]
    elif n_picks == 2:
        alloc = [per_stock * 1.2, per_stock * 0.8]
    else:
        alloc = [per_stock * 1.3, per_stock * 1.0, per_stock * 0.7]

    # 합이 1 초과하면 재정규화
    total_alloc = sum(alloc[:n_picks])
    if total_alloc > 0.95:
        alloc = [a / total_alloc * 0.95 for a in alloc]

    return alloc[:n_picks]


# ============================================
# [2] 모멘텀 전략 백테스트
# ============================================

def run_momentum_backtest(df, start_date, initial_capital=INITIAL_CAPITAL,
                          use_trailing_stop=False, use_kelly=False):
    """
    모멘텀 전략 백테스트

    Args:
        df: 원시 데이터 (워밍업 기간 포함)
        start_date: 포트폴리오 추적 시작일
        use_trailing_stop: 트레일링 스탑 사용 여부
        use_kelly: Kelly Criterion 사용 여부
    """
    _buy_comm = BUY_COMMISSION
    _sell_comm = SELL_COMMISSION
    _slippage = SLIPPAGE

    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    start_ts = pd.Timestamp(start_date)

    # 전략 준비 (전체 데이터로 모멘텀 점수 캐싱)
    strategy = CustomStrategy()
    price_df = prepare_price_data(df)
    tuesday_df = filter_tuesday(price_df)
    score_df, correlation_df, ret_1m = strategy.prepare(price_df, tuesday_df)
    score_dates = score_df.dropna(how='all').index.tolist()

    # 화요일 → 수요일 매핑
    all_dates = sorted(df['date'].unique())
    date_weekday = {d: pd.Timestamp(d).day_name() for d in all_dates}
    trade_map = {}
    for i, d in enumerate(all_dates):
        if date_weekday[d] == 'Tuesday':
            for j in range(i + 1, len(all_dates)):
                if date_weekday[all_dates[j]] == 'Wednesday':
                    trade_map[d] = all_dates[j]
                    break

    # 시뮬레이션 상태
    portfolio_values = []
    completed_trades = []  # Kelly 계산용 완료 거래
    cash = initial_capital
    holdings = {}   # {symbol: {'shares', 'avg_price', 'peak_price'}}
    pending_orders = []

    for date in all_dates:
        date_df = df[df['date'] == date]
        price_dict = dict(zip(date_df['symbol'], date_df['close']))

        # ── Step 1: 손절 체크 ──
        symbols_to_sell = []
        for symbol, info in holdings.items():
            if symbol not in price_dict:
                continue
            current_price = price_dict[symbol]

            # 트레일링 스탑: 고점 갱신
            if use_trailing_stop:
                info['peak_price'] = max(info['peak_price'], current_price)
                stop_price = info['peak_price'] * (1 - TRAILING_STOP_PCT)
                triggered = current_price <= stop_price
            else:
                return_rate = (current_price - info['avg_price']) / info['avg_price']
                triggered = return_rate <= STOP_LOSS_FIXED

            if triggered:
                symbols_to_sell.append(symbol)
                sell_price = current_price * (1 - _slippage)
                sell_amount = sell_price * info['shares']
                commission = sell_amount * _sell_comm
                cash += sell_amount - commission
                ret_pct = (sell_price - info['avg_price']) / info['avg_price'] * 100
                completed_trades.append({'return_pct': ret_pct})

        for s in symbols_to_sell:
            del holdings[s]

        # ── Step 2: 대기 주문 실행 ──
        for order in pending_orders:
            if order['trade_date'] != date:
                continue
            symbol = order['symbol']
            target_alloc = order['allocation']
            if symbol not in price_dict:
                continue

            current_price = price_dict[symbol]
            buy_price = current_price * (1 + _slippage)
            total_value = cash + sum(
                price_dict.get(s, i['avg_price']) * i['shares']
                for s, i in holdings.items()
            )
            target_amount = total_value * target_alloc
            current_amount = holdings[symbol]['shares'] * current_price if symbol in holdings else 0
            diff = target_amount - current_amount

            if diff > 10:
                buy_amount = min(diff, cash * 0.95)
                commission = buy_amount * _buy_comm
                if cash >= buy_amount + commission:
                    shares = int(buy_amount / buy_price)
                    if shares > 0:
                        actual_amount = shares * buy_price
                        cash -= actual_amount + commission
                        if symbol in holdings:
                            old = holdings[symbol]
                            new_shares = old['shares'] + shares
                            new_avg = (old['avg_price'] * old['shares'] + buy_price * shares) / new_shares
                            holdings[symbol]['shares'] = new_shares
                            holdings[symbol]['avg_price'] = new_avg
                            holdings[symbol]['peak_price'] = max(old['peak_price'], buy_price)
                        else:
                            holdings[symbol] = {
                                'shares': shares,
                                'avg_price': buy_price,
                                'peak_price': buy_price
                            }

            elif diff < -50:
                if symbol in holdings:
                    sell_shares = min(int(abs(diff) / current_price), holdings[symbol]['shares'])
                    if sell_shares > 0:
                        sell_price = current_price * (1 - _slippage)
                        sell_amount = sell_shares * sell_price
                        commission = sell_amount * _sell_comm
                        cash += sell_amount - commission
                        ret_pct = (sell_price - holdings[symbol]['avg_price']) / holdings[symbol]['avg_price'] * 100
                        completed_trades.append({'return_pct': ret_pct})
                        holdings[symbol]['shares'] -= sell_shares
                        if holdings[symbol]['shares'] <= 0:
                            del holdings[symbol]

        pending_orders = [o for o in pending_orders if o['trade_date'] != date]

        # ── Step 3: 화요일 신호 생성 ──
        if date in score_dates and date in trade_map:
            result = strategy.select_stocks(score_df, correlation_df, date, ret_1m)
            if result is not None:
                trade_date = trade_map[date]
                pending_orders = []

                n_picks = len(result['picks'])
                if use_kelly:
                    allocs = calc_kelly_allocations(completed_trades, n_picks)
                else:
                    allocs = result['allocations'][:n_picks]

                for symbol, score, alloc in zip(result['picks'], result['scores'], allocs):
                    pending_orders.append({
                        'symbol': symbol, 'score': score,
                        'allocation': alloc, 'trade_date': trade_date
                    })
                new_symbols = set(result['picks'])
                for symbol in list(holdings.keys()):
                    if symbol not in new_symbols:
                        pending_orders.append({
                            'symbol': symbol, 'score': 0,
                            'allocation': 0, 'trade_date': trade_date
                        })

        # ── Step 4: 포트폴리오 가치 기록 (start_date 이후만) ──
        if date >= start_ts:
            stock_value = sum(
                price_dict.get(s, i['avg_price']) * i['shares']
                for s, i in holdings.items()
            )
            portfolio_values.append({
                'date': date,
                'value': cash + stock_value
            })

    return pd.DataFrame(portfolio_values)


# ============================================
# [3] 하이브리드 전략 백테스트
# ============================================

def run_hybrid_backtest(strategy, test_features, initial_capital=INITIAL_CAPITAL,
                        use_trailing_stop=False, use_kelly=False,
                        label='Hybrid'):
    """
    하이브리드 전략 백테스트 (Kelly / Trailing Stop 옵션 포함)

    Args:
        strategy: prepare() 완료된 HybridStrategy 인스턴스
        test_features: 테스트 피처 DataFrame
        use_trailing_stop: 트레일링 스탑 사용 여부
        use_kelly: Kelly Criterion 사용 여부
        label: 출력용 전략명
    """
    _buy_comm = BUY_COMMISSION
    _sell_comm = SELL_COMMISSION
    _slippage = SLIPPAGE

    test_df = test_features.copy()
    test_df['date'] = pd.to_datetime(test_df['date'])
    dates = sorted(test_df['date'].unique())

    print(f"\n[{label}] 백테스트: {dates[0].strftime('%Y-%m-%d')} ~ {dates[-1].strftime('%Y-%m-%d')}")

    tuesday_dates = [d for d in dates if d.day_name() == 'Tuesday']
    trade_map = {}
    for tue in tuesday_dates:
        for d in dates:
            if d > tue and d.day_name() == 'Wednesday':
                trade_map[tue] = d
                break

    feature_cols = strategy.feature_cols
    portfolio_values = []
    completed_trades = []
    cash = initial_capital
    holdings = {}
    pending_orders = []

    for date in dates:
        date_df = test_df[test_df['date'] == date]
        price_dict = dict(zip(date_df['symbol'], date_df['close']))

        # ── Step 1: 손절 체크 ──
        symbols_to_sell = []
        for symbol, info in holdings.items():
            if symbol not in price_dict:
                continue
            current_price = price_dict[symbol]

            if use_trailing_stop:
                info['peak_price'] = max(info['peak_price'], current_price)
                stop_price = info['peak_price'] * (1 - TRAILING_STOP_PCT)
                triggered = current_price <= stop_price
            else:
                return_rate = (current_price - info['avg_price']) / info['avg_price']
                triggered = return_rate <= STOP_LOSS_FIXED

            if triggered:
                symbols_to_sell.append(symbol)
                sell_price = current_price * (1 - _slippage)
                sell_amount = sell_price * info['shares']
                commission = sell_amount * _sell_comm
                cash += sell_amount - commission
                ret_pct = (sell_price - info['avg_price']) / info['avg_price'] * 100
                completed_trades.append({'return_pct': ret_pct})

        for s in symbols_to_sell:
            del holdings[s]

        # ── Step 2: 대기 주문 실행 ──
        for order in pending_orders:
            if order['trade_date'] != date:
                continue
            symbol = order['symbol']
            target_alloc = order['allocation']
            if symbol not in price_dict:
                continue

            current_price = price_dict[symbol]
            buy_price = current_price * (1 + _slippage)
            total_value = cash + sum(
                price_dict.get(s, i['avg_price']) * i['shares']
                for s, i in holdings.items()
            )
            target_amount = total_value * target_alloc
            current_amount = holdings[symbol]['shares'] * current_price if symbol in holdings else 0
            diff = target_amount - current_amount

            if diff > 50:
                buy_amount = min(diff, cash * 0.95)
                commission = buy_amount * _buy_comm
                if cash >= buy_amount + commission:
                    shares = int(buy_amount / buy_price)
                    if shares > 0:
                        actual_amount = shares * buy_price
                        cash -= actual_amount + commission
                        if symbol in holdings:
                            old = holdings[symbol]
                            new_shares = old['shares'] + shares
                            new_avg = (old['avg_price'] * old['shares'] + buy_price * shares) / new_shares
                            holdings[symbol]['shares'] = new_shares
                            holdings[symbol]['avg_price'] = new_avg
                            holdings[symbol]['peak_price'] = max(old['peak_price'], buy_price)
                        else:
                            holdings[symbol] = {
                                'shares': shares,
                                'avg_price': buy_price,
                                'peak_price': buy_price
                            }

            elif diff < -50:
                if symbol in holdings:
                    sell_shares = min(int(abs(diff) / current_price), holdings[symbol]['shares'])
                    if sell_shares > 0:
                        sell_price = current_price * (1 - _slippage)
                        sell_amount = sell_shares * sell_price
                        commission = sell_amount * _sell_comm
                        cash += sell_amount - commission
                        ret_pct = (sell_price - info['avg_price']) / info['avg_price'] * 100
                        completed_trades.append({'return_pct': ret_pct})
                        holdings[symbol]['shares'] -= sell_shares
                        if holdings[symbol]['shares'] <= 0:
                            del holdings[symbol]

        pending_orders = [o for o in pending_orders if o['trade_date'] != date]

        # ── Step 3: 화요일 신호 생성 ──
        if date in trade_map:
            result = strategy.select_stocks(test_df, feature_cols, date)
            if result is not None:
                trade_date = trade_map[date]
                pending_orders = []

                n_picks = len(result['picks'])
                if use_kelly:
                    allocs = calc_kelly_allocations(completed_trades, n_picks)
                else:
                    allocs = result['allocations'][:n_picks]

                for i, symbol in enumerate(result['picks']):
                    pending_orders.append({
                        'symbol': symbol,
                        'score': result['scores'][i],
                        'allocation': allocs[i],
                        'trade_date': trade_date
                    })
                new_symbols = set(result['picks'])
                for symbol in list(holdings.keys()):
                    if symbol not in new_symbols:
                        pending_orders.append({
                            'symbol': symbol, 'score': 0,
                            'allocation': 0, 'trade_date': trade_date
                        })

        # ── Step 4: 포트폴리오 가치 기록 ──
        stock_value = sum(
            price_dict.get(s, i['avg_price']) * i['shares']
            for s, i in holdings.items()
        )
        portfolio_values.append({
            'date': date,
            'value': cash + stock_value
        })

    return pd.DataFrame(portfolio_values)


# ============================================
# [4] 결과 출력
# ============================================

def print_summary(results, spy_return):
    """성과 요약 테이블 출력"""
    print()
    print("=" * 80)
    print("  전략 비교 결과")
    print("=" * 80)
    header = f"{'전략':<30} {'총수익률':>9} {'CAGR':>8} {'MDD':>8} {'샤프':>6} {'Alpha':>8}"
    print(header)
    print("-" * 80)

    # SPY 먼저 출력
    print(f"{'SPY (벤치마크)':<30} {spy_return*100:>8.2f}%  {'':>7} {'':>7} {'':>5} {'':>7}")
    print("-" * 80)

    for name, m in results.items():
        print(
            f"{name:<30} "
            f"{m['total_return']*100:>8.2f}%  "
            f"{m['cagr']*100:>6.2f}%  "
            f"{m['mdd']*100:>6.2f}%  "
            f"{m['sharpe']:>5.2f}  "
            f"{m['alpha']*100:>+7.2f}%"
        )
    print("=" * 80)


# ============================================
# [5] 차트
# ============================================

def plot_comparison(portfolios, spy_series, results, output_path):
    """
    5개 전략 + SPY 수익률 비교 차트

    패널 1: 누적 수익률 곡선 (정규화 100 기준)
    패널 2: MDD 비교 바 차트
    패널 3: 샤프/Alpha 비교 바 차트
    """
    colors = {
        '1. 모멘텀':                     '#2196F3',
        '2. 하이브리드':                  '#4CAF50',
        '3. 하이브리드+Kelly':            '#FF9800',
        '4. 하이브리드+TrailingStop':     '#E91E63',
        '5. 하이브리드+Kelly+Trailing':   '#9C27B0',
    }

    fig = plt.figure(figsize=(16, 14))
    gs = fig.add_gridspec(3, 2, height_ratios=[3, 1, 1], hspace=0.45, wspace=0.35)

    ax_main = fig.add_subplot(gs[0, :])   # 수익률 곡선 (전체 너비)
    ax_mdd  = fig.add_subplot(gs[1, 0])
    ax_shp  = fig.add_subplot(gs[1, 1])
    ax_ret  = fig.add_subplot(gs[2, 0])
    ax_alp  = fig.add_subplot(gs[2, 1])

    today_str = datetime.now().strftime('%Y-%m-%d')
    fig.suptitle(
        f'전략 비교 백테스트  |  하이브리드 학습: {HYBRID_TRAIN_START}~{HYBRID_TRAIN_END}'
        f'  |  백테스트: {HYBRID_TEST_START}~{today_str}',
        fontsize=13, fontweight='bold', y=0.98
    )

    # ── 패널 1: 누적 수익률 ──
    if not spy_series.empty:
        ax_main.plot(spy_series.index, spy_series.values,
                     label='SPY', color='gray', linewidth=2,
                     linestyle='--', alpha=0.8)

    for name, port_df in portfolios.items():
        port = port_df.copy()
        port['date'] = pd.to_datetime(port['date'])
        port['norm'] = port['value'] / port['value'].iloc[0] * 100
        ax_main.plot(port['date'], port['norm'],
                     label=name, color=colors.get(name, 'black'),
                     linewidth=1.8, alpha=0.9)

    ax_main.axhline(100, color='black', linewidth=0.5, linestyle=':')
    ax_main.set_title('누적 수익률 (초기=100)', fontsize=12)
    ax_main.set_ylabel('Value (base=100)')
    ax_main.legend(loc='upper left', fontsize=9)
    ax_main.grid(True, alpha=0.3)
    ax_main.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))

    # ── 패널 2~5: 지표 바 차트 ──
    names = list(results.keys())
    short_names = ['모멘텀', '하이브리드', '+Kelly', '+Trailing', '+K+T']
    x = range(len(names))

    bar_colors = [colors.get(n, '#607D8B') for n in names]

    # MDD
    mdd_vals = [results[n]['mdd'] * 100 for n in names]
    bars = ax_mdd.bar(x, mdd_vals, color=bar_colors, alpha=0.8)
    ax_mdd.set_title('MDD (%)', fontsize=10)
    ax_mdd.set_xticks(list(x))
    ax_mdd.set_xticklabels(short_names, fontsize=8)
    ax_mdd.grid(True, alpha=0.3, axis='y')
    for bar, v in zip(bars, mdd_vals):
        ax_mdd.text(bar.get_x() + bar.get_width()/2, v - 0.3,
                    f'{v:.1f}%', ha='center', va='top', fontsize=7)

    # 샤프
    shp_vals = [results[n]['sharpe'] for n in names]
    bars = ax_shp.bar(x, shp_vals, color=bar_colors, alpha=0.8)
    ax_shp.set_title('Sharpe Ratio', fontsize=10)
    ax_shp.set_xticks(list(x))
    ax_shp.set_xticklabels(short_names, fontsize=8)
    ax_shp.grid(True, alpha=0.3, axis='y')
    for bar, v in zip(bars, shp_vals):
        ax_shp.text(bar.get_x() + bar.get_width()/2, v + 0.02,
                    f'{v:.2f}', ha='center', va='bottom', fontsize=7)

    # 총수익률
    ret_vals = [results[n]['total_return'] * 100 for n in names]
    bars = ax_ret.bar(x, ret_vals, color=bar_colors, alpha=0.8)
    ax_ret.set_title('총 수익률 (%)', fontsize=10)
    ax_ret.set_xticks(list(x))
    ax_ret.set_xticklabels(short_names, fontsize=8)
    ax_ret.axhline(0, color='black', linewidth=0.5)
    ax_ret.grid(True, alpha=0.3, axis='y')
    for bar, v in zip(bars, ret_vals):
        offset = 0.3 if v >= 0 else -0.3
        va = 'bottom' if v >= 0 else 'top'
        ax_ret.text(bar.get_x() + bar.get_width()/2, v + offset,
                    f'{v:.1f}%', ha='center', va=va, fontsize=7)

    # Alpha
    alp_vals = [results[n]['alpha'] * 100 for n in names]
    bar_c = ['#4CAF50' if v >= 0 else '#F44336' for v in alp_vals]
    bars = ax_alp.bar(x, alp_vals, color=bar_c, alpha=0.8)
    ax_alp.set_title('Alpha vs SPY (%)', fontsize=10)
    ax_alp.set_xticks(list(x))
    ax_alp.set_xticklabels(short_names, fontsize=8)
    ax_alp.axhline(0, color='black', linewidth=0.5)
    ax_alp.grid(True, alpha=0.3, axis='y')
    for bar, v in zip(bars, alp_vals):
        offset = 0.1 if v >= 0 else -0.1
        va = 'bottom' if v >= 0 else 'top'
        ax_alp.text(bar.get_x() + bar.get_width()/2, v + offset,
                    f'{v:+.1f}%', ha='center', va=va, fontsize=7)

    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n차트 저장: {output_path}")


# ============================================
# [6] 메인
# ============================================

def parse_args():
    parser = argparse.ArgumentParser(description='5가지 전략 비교 백테스트')
    parser.add_argument('--initial_capital', type=float, default=INITIAL_CAPITAL,
                        help=f'초기 자본금 (기본: {INITIAL_CAPITAL})')
    parser.add_argument('--output', type=str, default='comparison_backtest.png',
                        help='차트 저장 경로 (기본: comparison_backtest.png)')
    return parser.parse_args()


def main():
    args = parse_args()
    today_str = datetime.now().strftime('%Y-%m-%d')

    print("=" * 65)
    print("  5가지 전략 비교 백테스트")
    print("=" * 65)
    print(f"  모멘텀 기간     : {MOMENTUM_START} ~ {today_str}")
    print(f"  하이브리드 학습  : {HYBRID_TRAIN_START} ~ {HYBRID_TRAIN_END}")
    print(f"  하이브리드 테스트: {HYBRID_TEST_START} ~ {today_str}")
    print(f"  초기 자본금     : ${args.initial_capital:,.2f}")
    print(f"  고정 손절       : {STOP_LOSS_FIXED*100:.1f}%")
    print(f"  트레일링 스탑    : 고점 대비 -{TRAILING_STOP_PCT*100:.1f}%")
    print(f"  Kelly Half      : {KELLY_HALF}")
    print("=" * 65)

    # ── [1] 종목 목록 ──
    print("\n[1] S&P 500 종목 목록 로딩...")
    try:
        sp500 = get_sp500_list()
        symbols = sp500['symbol'].tolist() + ['SPY']
    except Exception:
        symbols = SP500_BACKUP + ['SPY']
    print(f"    종목 수: {len(symbols)}")

    # ── [2] 데이터 다운로드 ──
    print(f"\n[2] 모멘텀 데이터 다운로드 ({MOMENTUM_WARMUP} ~ {today_str})...")
    momentum_raw = get_backtest_data(symbols, start_date=MOMENTUM_WARMUP, end_date=today_str)
    print(f"    rows: {len(momentum_raw):,}")

    print(f"\n[3] 하이브리드 학습 데이터 다운로드 ({HYBRID_TRAIN_START} ~ {HYBRID_TRAIN_END})...")
    train_raw = get_backtest_data(symbols, start_date=HYBRID_TRAIN_START, end_date=HYBRID_TRAIN_END)
    print(f"    rows: {len(train_raw):,}")

    print(f"\n[4] 하이브리드 테스트 데이터 다운로드 ({HYBRID_TEST_START} ~ {today_str})...")
    test_raw = get_backtest_data(symbols, start_date=HYBRID_TEST_START, end_date=today_str)
    print(f"    rows: {len(test_raw):,}")

    # ── [3] 피처 생성 ──
    print("\n[5] 피처 생성 중...")
    train_features = create_features(train_raw)
    test_features  = create_features(test_raw)
    feature_cols   = get_feature_columns(train_features)
    print(f"    피처 수: {len(feature_cols)}, 학습: {len(train_features):,}, 테스트: {len(test_features):,}")

    # ── [4] 하이브리드 전략 학습 (1번만) ──
    print("\n[6] 하이브리드 전략 AI 학습 중...")
    price_df_test = prepare_price_data(test_raw)
    strategy = HybridStrategy()
    strategy.prepare(train_features, price_df_test, feature_cols)

    # ── [5] SPY 시계열 ──
    spy_series = get_spy_series(test_raw, HYBRID_TEST_START)

    # ── [6] 5가지 백테스트 실행 ──
    print("\n" + "=" * 65)
    print("  백테스트 실행")
    print("=" * 65)

    portfolios = {}
    results = {}

    # 1. 모멘텀
    print("\n[전략 1] 모멘텀")
    p1 = run_momentum_backtest(
        momentum_raw, MOMENTUM_START,
        initial_capital=args.initial_capital,
        use_trailing_stop=False, use_kelly=False
    )
    portfolios['1. 모멘텀'] = p1
    results['1. 모멘텀'] = calc_metrics(p1, spy_series, args.initial_capital)

    # 2. 하이브리드
    print("\n[전략 2] 하이브리드")
    p2 = run_hybrid_backtest(
        strategy, test_features,
        initial_capital=args.initial_capital,
        use_trailing_stop=False, use_kelly=False,
        label='하이브리드'
    )
    portfolios['2. 하이브리드'] = p2
    results['2. 하이브리드'] = calc_metrics(p2, spy_series, args.initial_capital)

    # 3. 하이브리드 + Kelly
    print("\n[전략 3] 하이브리드 + Kelly")
    p3 = run_hybrid_backtest(
        strategy, test_features,
        initial_capital=args.initial_capital,
        use_trailing_stop=False, use_kelly=True,
        label='하이브리드+Kelly'
    )
    portfolios['3. 하이브리드+Kelly'] = p3
    results['3. 하이브리드+Kelly'] = calc_metrics(p3, spy_series, args.initial_capital)

    # 4. 하이브리드 + Trailing Stop
    print("\n[전략 4] 하이브리드 + TrailingStop")
    p4 = run_hybrid_backtest(
        strategy, test_features,
        initial_capital=args.initial_capital,
        use_trailing_stop=True, use_kelly=False,
        label='하이브리드+TrailingStop'
    )
    portfolios['4. 하이브리드+TrailingStop'] = p4
    results['4. 하이브리드+TrailingStop'] = calc_metrics(p4, spy_series, args.initial_capital)

    # 5. 하이브리드 + Kelly + Trailing
    print("\n[전략 5] 하이브리드 + Kelly + Trailing")
    p5 = run_hybrid_backtest(
        strategy, test_features,
        initial_capital=args.initial_capital,
        use_trailing_stop=True, use_kelly=True,
        label='하이브리드+Kelly+Trailing'
    )
    portfolios['5. 하이브리드+Kelly+Trailing'] = p5
    results['5. 하이브리드+Kelly+Trailing'] = calc_metrics(p5, spy_series, args.initial_capital)

    # ── [7] 결과 출력 ──
    spy_return = (spy_series.iloc[-1] - spy_series.iloc[0]) / spy_series.iloc[0] if not spy_series.empty else 0
    print_summary(results, spy_return)

    # ── [8] 차트 저장 ──
    plot_comparison(portfolios, spy_series, results, args.output)

    print("\n완료!")


if __name__ == '__main__':
    main()
