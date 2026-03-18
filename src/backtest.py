# ============================================
# 파일명: src/backtest.py
# 설명: 모멘텀 전략 백테스트
# 
# 구조:
# - config.py에서 설정 가져옴
# - strategy.py의 CustomStrategy 사용
# - data.py의 함수 사용
# ============================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# config.py에서 설정 가져오기
from config import (
    INITIAL_CAPITAL,
    STOP_LOSS,
    TOP_N,
    BUY_COMMISSION,
    SELL_COMMISSION,
    SLIPPAGE,
    BACKTEST_START,
    BACKTEST_END
)

# strategy.py에서 전략 가져오기
from strategy import (
    CustomStrategy,
    prepare_price_data,
    filter_tuesday
)


# ============================================
# [1] 매수일 매핑 생성
# ============================================

def create_trade_mapping(df):
    """
    화요일 → 수요일 매수일 매핑을 생성합니다.
    """
    dates = sorted(df['date'].unique())
    date_weekday = {d: pd.Timestamp(d).day_name() for d in dates}
    
    trade_map = {}
    
    for i, date in enumerate(dates):
        if date_weekday[date] == 'Tuesday':
            for j in range(i+1, len(dates)):
                if date_weekday[dates[j]] == 'Wednesday':
                    trade_map[date] = dates[j]
                    break
    
    return trade_map


# ============================================
# [2] 백테스트 메인 함수
# ============================================

def run_backtest(df=None, initial_capital=None, commission=None, slippage=None):
    """
    백테스트를 실행합니다.

    Args:
        df: 원본 데이터프레임 (None이면 자동 로딩)
        initial_capital: 초기 자본금 (None이면 config 값 사용)
        commission: 수수료율 (None이면 config 값 사용)
        slippage: 슬리피지율 (None이면 config 값 사용)

    Returns:
        dict: {portfolio, trades, metrics, df}
    """
    # 파라미터 기본값 설정
    _capital = initial_capital if initial_capital is not None else INITIAL_CAPITAL
    _buy_comm = commission if commission is not None else BUY_COMMISSION
    _sell_comm = commission if commission is not None else SELL_COMMISSION
    _slippage = slippage if slippage is not None else SLIPPAGE

    # ===== 데이터 자동 로딩 =====
    if df is None:
        from data import get_backtest_data
        df = get_backtest_data()

    # ===== 전략 초기화 =====
    strategy = CustomStrategy()

    # ===== 설정 출력 =====
    print("=" * 60)
    print("[백테스트 실행]")
    print("=" * 60)
    print(f"전략: CustomStrategy (상관관계 필터 + 중장기 모멘텀)")
    print(f"상관관계 기준: {strategy.correlation_threshold}")
    print(f"종목 수: {strategy.top_n}개")
    print(f"초기 자본금: ${_capital:,}")
    print(f"수수료: 매수 {_buy_comm*100:.2f}% + 매도 {_sell_comm*100:.2f}%")
    print(f"슬리피지: {_slippage*100:.2f}%")
    print(f"손절: {STOP_LOSS*100:.1f}%")
    print("=" * 60)
    
    # ===== 데이터 준비 =====
    df_daily = df.copy().sort_values('date').reset_index(drop=True)
    daily_dates = sorted(df_daily['date'].unique())
    
    print(f"데이터 기간: {daily_dates[0].strftime('%Y-%m-%d')} ~ {daily_dates[-1].strftime('%Y-%m-%d')}")
    print(f"총 {len(daily_dates)}일")
    
    # 피벗 테이블 생성
    price_df = prepare_price_data(df)
    
    # 화요일만 필터링
    tuesday_df = filter_tuesday(price_df)
    if 'SPY' in tuesday_df.columns:
        tuesday_df = tuesday_df.dropna(subset=['SPY'])
    print(f"화요일 데이터: {len(tuesday_df)}개")
    
    # 전략 준비 (점수, 상관관계 계산)
    score_df, correlation_df, ret_1m = strategy.prepare(price_df, tuesday_df)
    
    # 화요일 → 수요일 매핑
    trade_map = create_trade_mapping(df)
    print(f"매핑된 거래일: {len(trade_map)}개")
    
    # 점수가 있는 날짜
    score_dates = score_df.dropna(how='all').index.tolist()
    
    # ===== 시뮬레이션 변수 =====
    portfolio_values = []
    trades = []

    cash = _capital
    holdings = {}  # {symbol: {'shares': int, 'avg_price': float, 'buy_date': date}}
    
    pending_orders = []
    
    # ===== 일별 시뮬레이션 =====
    for date in daily_dates:
        date_df = df_daily[df_daily['date'] == date]
        price_dict = dict(zip(date_df['symbol'], date_df['close']))
        
        # ----- 1. 손절 체크 -----
        symbols_to_sell = []
        for symbol, info in holdings.items():
            if symbol not in price_dict:
                continue
            
            current_price = price_dict[symbol]
            return_rate = (current_price - info['avg_price']) / info['avg_price']
            
            if return_rate <= STOP_LOSS:
                symbols_to_sell.append(symbol)
                
                sell_price = current_price * (1 - _slippage)
                sell_amount = sell_price * info['shares']
                commission = sell_amount * _sell_comm
                cash += sell_amount - commission

                trades.append({
                    'date': date,
                    'symbol': symbol,
                    'action': 'STOP_LOSS',
                    'shares': info['shares'],
                    'price': sell_price,
                    'amount': sell_amount,
                    'commission': commission,
                    'return_pct': return_rate * 100
                })
        
        for symbol in symbols_to_sell:
            del holdings[symbol]
        
        # ----- 2. 대기 주문 실행 (수요일) -----
        for order in pending_orders:
            if order['trade_date'] == date:
                symbol = order['symbol']
                target_allocation = order['allocation']
                
                if symbol not in price_dict:
                    continue
                
                current_price = price_dict[symbol]
                buy_price = current_price * (1 + _slippage)

                # 목표 금액
                total_value = cash + sum(
                    price_dict.get(s, info['avg_price']) * info['shares']
                    for s, info in holdings.items()
                )
                target_amount = total_value * target_allocation
                
                # 현재 보유 금액
                current_amount = 0
                if symbol in holdings:
                    current_amount = holdings[symbol]['shares'] * current_price
                
                # 차이 계산
                diff = target_amount - current_amount
                
                if diff > 0:  # 매수
                    buy_amount = min(diff, cash * 0.95)
                    commission = buy_amount * _buy_comm
                    
                    if cash >= buy_amount + commission and buy_amount > 10:
                        shares = int(buy_amount / buy_price)
                        if shares > 0:
                            actual_amount = shares * buy_price
                            cash -= actual_amount + commission
                            
                            if symbol in holdings:
                                # 추가 매수
                                old_shares = holdings[symbol]['shares']
                                old_avg = holdings[symbol]['avg_price']
                                new_shares = old_shares + shares
                                new_avg = (old_avg * old_shares + buy_price * shares) / new_shares
                                holdings[symbol]['shares'] = new_shares
                                holdings[symbol]['avg_price'] = new_avg
                                action = 'ADD'
                            else:
                                # 신규 매수
                                holdings[symbol] = {
                                    'shares': shares,
                                    'avg_price': buy_price,
                                    'buy_date': date
                                }
                                action = 'BUY'
                            
                            trades.append({
                                'date': date,
                                'symbol': symbol,
                                'action': action,
                                'shares': shares,
                                'price': buy_price,
                                'amount': actual_amount,
                                'commission': commission,
                                'score': order.get('score', 0),
                                'return_pct': 0
                            })
                
                elif diff < -50:  # 매도 (비중 축소)
                    if symbol in holdings:
                        sell_shares = int(abs(diff) / current_price)
                        if sell_shares > 0:
                            sell_shares = min(sell_shares, holdings[symbol]['shares'])
                            sell_price = current_price * (1 - _slippage)
                            sell_amount = sell_shares * sell_price
                            commission = sell_amount * _sell_comm
                            cash += sell_amount - commission
                            
                            return_pct = (sell_price - holdings[symbol]['avg_price']) / holdings[symbol]['avg_price'] * 100
                            
                            trades.append({
                                'date': date,
                                'symbol': symbol,
                                'action': 'REDUCE',
                                'shares': sell_shares,
                                'price': sell_price,
                                'amount': sell_amount,
                                'commission': commission,
                                'return_pct': return_pct
                            })
                            
                            holdings[symbol]['shares'] -= sell_shares
                            if holdings[symbol]['shares'] <= 0:
                                del holdings[symbol]
        
        # 실행된 주문 제거
        pending_orders = [o for o in pending_orders if o['trade_date'] != date]
        
        # ----- 3. 화요일: 종목 선정 -----
        if date in score_dates and date in trade_map:
            result = strategy.select_stocks(score_df, correlation_df, date, ret_1m)
            
            if result is not None:
                trade_date = trade_map[date]
                
                # 기존 주문 초기화
                pending_orders = []
                
                # 새 주문 추가
                for symbol, score, allocation in zip(result['picks'], result['scores'], result['allocations']):
                    pending_orders.append({
                        'symbol': symbol,
                        'score': score,
                        'allocation': allocation,
                        'trade_date': trade_date
                    })
                
                # 보유 중이지만 새 리스트에 없는 종목 매도 예약
                new_symbols = set(result['picks'])
                for symbol in list(holdings.keys()):
                    if symbol not in new_symbols:
                        pending_orders.append({
                            'symbol': symbol,
                            'score': 0,
                            'allocation': 0,
                            'trade_date': trade_date
                        })
        
        # ----- 4. 포트폴리오 가치 계산 -----
        stock_value = sum(
            price_dict.get(s, info['avg_price']) * info['shares']
            for s, info in holdings.items()
        )
        total_value = cash + stock_value
        
        portfolio_values.append({
            'date': date,
            'value': total_value,
            'cash': cash,
            'stock_value': stock_value,
            'holdings': len(holdings)
        })
    
    # ===== 결과 정리 =====
    portfolio_df = pd.DataFrame(portfolio_values)
    trades_df = pd.DataFrame(trades) if trades else pd.DataFrame()
    
    # 성과 지표 계산
    metrics = calculate_metrics(portfolio_df, trades_df, df, _slippage)
    
    print("\n" + "=" * 60)
    print("✅ 백테스트 완료!")
    print("=" * 60)
    
    return {
        'portfolio': portfolio_df,
        'trades': trades_df,
        'metrics': metrics,
        'df': df
    }


# ============================================
# [3] 성과 지표 계산
# ============================================

def calculate_metrics(portfolio_df, trades_df, df, slippage_rate=SLIPPAGE):
    """
    백테스트 성과 지표를 계산합니다.
    """
    values = portfolio_df['value'].values
    dates = portfolio_df['date']
    
    # 수익률
    initial = values[0]
    final = values[-1]
    total_return = (final - initial) / initial
    
    # 일별 수익률
    daily_returns = pd.Series(values).pct_change().dropna()
    
    # CAGR
    days = (dates.iloc[-1] - dates.iloc[0]).days
    years = days / 365
    cagr = (final / initial) ** (1 / years) - 1 if years > 0 else 0
    
    # 변동성 & 샤프
    volatility = daily_returns.std() * np.sqrt(252)
    risk_free_rate = 0.03
    sharpe = (cagr - risk_free_rate) / volatility if volatility > 0 else 0
    
    # MDD
    peak = pd.Series(values).cummax()
    drawdown = (pd.Series(values) - peak) / peak
    mdd = drawdown.min()
    
    # 승률
    win_rate = (daily_returns > 0).mean()
    
    # SPY 수익률
    spy_return = 0
    if 'SPY' in df['symbol'].unique():
        spy = df[df['symbol'] == 'SPY'].sort_values('date')
        if len(spy) >= 2:
            spy_initial = spy.iloc[0]['close']
            spy_final = spy.iloc[-1]['close']
            spy_return = (spy_final - spy_initial) / spy_initial
    
    # 거래 통계
    total_trades = len(trades_df) if not trades_df.empty else 0
    total_commission = trades_df['commission'].sum() if not trades_df.empty else 0
    total_slippage = trades_df['amount'].sum() * slippage_rate if not trades_df.empty else 0
    stop_loss_count = len(trades_df[trades_df['action'] == 'STOP_LOSS']) if not trades_df.empty else 0
    
    buy_count = len(trades_df[trades_df['action'] == 'BUY']) if not trades_df.empty else 0
    sell_count = len(trades_df[trades_df['action'] == 'SELL']) if not trades_df.empty else 0
    add_count = len(trades_df[trades_df['action'] == 'ADD']) if not trades_df.empty else 0
    reduce_count = len(trades_df[trades_df['action'] == 'REDUCE']) if not trades_df.empty else 0
    
    return {
        'initial_capital': initial,
        'final_capital': final,
        'total_return': total_return,
        'cagr': cagr,
        'volatility': volatility,
        'sharpe_ratio': sharpe,
        'mdd': mdd,
        'win_rate': win_rate,
        'spy_return': spy_return,
        'alpha': total_return - spy_return,
        'total_trades': total_trades,
        'buy_count': buy_count,
        'sell_count': sell_count,
        'add_count': add_count,
        'reduce_count': reduce_count,
        'total_commission': total_commission,
        'total_slippage': total_slippage,
        'stop_loss_count': stop_loss_count
    }


# ============================================
# [4] 결과 출력
# ============================================

def print_metrics(metrics, trades_df=None):
    """
    백테스트 성과를 출력합니다.
    """
    print("\n" + "=" * 60)
    print("📊 백테스트 성과")
    print("=" * 60)
    
    print(f"\n💰 수익")
    print(f"  초기 자본금: ${metrics['initial_capital']:,.2f}")
    print(f"  최종 자본금: ${metrics['final_capital']:,.2f}")
    print(f"  총 수익률: {metrics['total_return']*100:.2f}%")
    print(f"  연환산 수익률 (CAGR): {metrics['cagr']*100:.2f}%")
    
    print(f"\n📈 벤치마크 비교")
    print(f"  SPY 수익률: {metrics['spy_return']*100:.2f}%")
    print(f"  초과 수익 (Alpha): {metrics['alpha']*100:.2f}%")
    
    print(f"\n⚠️ 위험 지표")
    print(f"  변동성: {metrics['volatility']*100:.2f}%")
    print(f"  최대 낙폭 (MDD): {metrics['mdd']*100:.2f}%")
    print(f"  샤프 비율: {metrics['sharpe_ratio']:.2f}")
    
    print(f"\n🎯 거래 통계")
    print(f"  총 거래 횟수: {metrics['total_trades']}회")
    print(f"    - 신규 매수 (BUY): {metrics['buy_count']}회")
    print(f"    - 추가 매수 (ADD): {metrics['add_count']}회")
    print(f"    - 일부 매도 (REDUCE): {metrics['reduce_count']}회")
    print(f"    - 손절 (STOP_LOSS): {metrics['stop_loss_count']}회")
    print(f"  총 수수료: ${metrics['total_commission']:,.2f}")
    
    print(f"\n📅 기타")
    print(f"  승률 (일 기준): {metrics['win_rate']*100:.2f}%")
    
    print("\n" + "=" * 60)


# ============================================
# [5] 그래프 출력
# ============================================

def plot_results(portfolio_df, trades_df, df, figsize=(14, 12)):
    """
    백테스트 결과를 그래프로 시각화합니다.
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    # ----- 1. 포트폴리오 vs SPY -----
    ax1 = axes[0, 0]
    
    portfolio_df = portfolio_df.copy()
    portfolio_df['normalized'] = portfolio_df['value'] / portfolio_df['value'].iloc[0] * 100
    ax1.plot(portfolio_df['date'], portfolio_df['normalized'], 
             label='Portfolio', linewidth=2, color='blue')
    
    if 'SPY' in df['symbol'].unique():
        spy = df[df['symbol'] == 'SPY'].sort_values('date').copy()
        spy['normalized'] = spy['close'] / spy['close'].iloc[0] * 100
        ax1.plot(spy['date'], spy['normalized'], 
                 label='SPY', linewidth=2, linestyle='--', color='orange')
    
    ax1.set_title('Portfolio vs SPY', fontsize=12)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # ----- 2. 일별 수익률 -----
    ax2 = axes[0, 1]
    daily_returns = portfolio_df['value'].pct_change().dropna()
    colors = ['green' if r > 0 else 'red' for r in daily_returns]
    ax2.bar(range(len(daily_returns)), daily_returns, color=colors, alpha=0.7)
    ax2.axhline(y=0, color='black', linewidth=0.5)
    ax2.set_title('Daily Returns', fontsize=12)
    ax2.grid(True, alpha=0.3)
    
    # ----- 3. 누적 수익률 -----
    ax3 = axes[1, 0]
    cumulative = (1 + daily_returns).cumprod() - 1
    ax3.fill_between(range(len(cumulative)), cumulative, alpha=0.3, color='blue')
    ax3.plot(range(len(cumulative)), cumulative, linewidth=2, color='blue')
    ax3.axhline(y=0, color='black', linewidth=0.5)
    ax3.set_title('Cumulative Returns', fontsize=12)
    ax3.grid(True, alpha=0.3)
    
    # ----- 4. Drawdown -----
    ax4 = axes[1, 1]
    peak = portfolio_df['value'].cummax()
    drawdown = (portfolio_df['value'] - peak) / peak
    ax4.fill_between(portfolio_df['date'], drawdown, 0, color='red', alpha=0.3)
    ax4.plot(portfolio_df['date'], drawdown, color='red', linewidth=1)
    ax4.set_title('Drawdown', fontsize=12)
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


# ============================================
# [6] 테스트 실행
# ============================================

if __name__ == "__main__":
    print("백테스트 모듈")
    print("Colab에서 실행:")
    print()
    print("from backtest import run_backtest, print_metrics, plot_results")
    print()
    print("results = run_backtest()")
    print("print_metrics(results['metrics'], results['trades'])")
    print("plot_results(results['portfolio'], results['trades'], results['df'])")