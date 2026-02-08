# ============================================
# 파일명: src/backtest.py
# 설명: 백테스트 시뮬레이션
# 
# strategy.py의 CustomStrategy를 사용하여
# 과거 데이터로 매매 시뮬레이션 수행
# ============================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from src.strategy import (
    CustomStrategy, 
    prepare_price_data, 
    filter_tuesday, 
    create_trade_mapping
)


# ============================================
# [설정] 백테스트 파라미터
# ============================================

INITIAL_CAPITAL = 2000       # 초기 자본금
BUY_COMMISSION = 0.0025      # 매수 수수료 (0.25%)
SELL_COMMISSION = 0.0025     # 매도 수수료 (0.25%)
SLIPPAGE = 0.001             # 슬리피지 (0.1%)
STOP_LOSS = -0.07            # 손절 기준 (-7%)


# ============================================
# [1] 백테스트 메인 함수
# ============================================

def run_backtest(df, strategy=None):
    """
    백테스트 실행
    
    Args:
        df: 원본 데이터프레임 (date, symbol, close)
        strategy: CustomStrategy 인스턴스 (없으면 기본값 사용)
    
    Returns:
        dict: {
            'portfolio': 일별 포트폴리오 가치,
            'trades': 거래 내역,
            'metrics': 성과 지표
        }
    """
    
    # 전략 인스턴스
    if strategy is None:
        strategy = CustomStrategy()
    
    # ===== 초기 설정 출력 =====
    print("=" * 60)
    print("[백테스트 실행]")
    print("=" * 60)
    print(f"전략: CustomStrategy (상관관계 + 중장기 모멘텀)")
    print(f"초기 자본금: ${INITIAL_CAPITAL:,}")
    print(f"수수료: {BUY_COMMISSION*100:.2f}% + {SELL_COMMISSION*100:.2f}%")
    print(f"슬리피지: {SLIPPAGE*100:.2f}%")
    print(f"손절: {STOP_LOSS*100:.1f}%")
    print("=" * 60)
    
    # ===== 데이터 준비 =====
    df_daily = df.copy().sort_values('date').reset_index(drop=True)
    daily_dates = sorted(df_daily['date'].unique())
    
    print(f"데이터 기간: {daily_dates[0].strftime('%Y-%m-%d')} ~ {daily_dates[-1].strftime('%Y-%m-%d')}")
    print(f"총 {len(daily_dates)}일")
    
    # 피벗 테이블
    price_df = prepare_price_data(df)
    tuesday_df = filter_tuesday(price_df)
    
    if 'SPY' in tuesday_df.columns:
        tuesday_df = tuesday_df.dropna(subset=['SPY'])
    
    print(f"화요일 데이터: {len(tuesday_df)}개")
    
    # 전략 데이터 준비
    score_df, correlation_df, ret_1m = strategy.prepare(price_df, tuesday_df)
    
    # 매수일 매핑
    trade_map = create_trade_mapping(df)
    print(f"매핑된 거래일: {len(trade_map)}개")
    
    score_dates = score_df.dropna(how='all').index.tolist()
    
    # ===== 시뮬레이션 변수 =====
    portfolio_values = []
    trades = []
    
    cash = INITIAL_CAPITAL
    holdings = {}
    pending_order = None
    
    print(f"\n{len(daily_dates)}일 시뮬레이션 시작...")
    
    # ===== 매일 시뮬레이션 =====
    for i, date in enumerate(daily_dates):
        
        if (i + 1) % 100 == 0:
            print(f"  진행중... {i+1}/{len(daily_dates)} ({(i+1)/len(daily_dates)*100:.1f}%)")
        
        today_data = df_daily[df_daily['date'] == date]
        date_ts = pd.Timestamp(date)
        
        # ----- 포트폴리오 가치 계산 -----
        portfolio_value = cash
        for symbol, info in holdings.items():
            stock = today_data[today_data['symbol'] == symbol]
            if not stock.empty:
                portfolio_value += info['shares'] * stock.iloc[0]['close']
        
        portfolio_values.append({
            'date': date,
            'value': portfolio_value,
            'cash': cash
        })
        
        # ----- 손절 체크 -----
        for symbol, info in list(holdings.items()):
            stock = today_data[today_data['symbol'] == symbol]
            if stock.empty:
                continue
            
            current_price = stock.iloc[0]['close']
            return_rate = (current_price - info['avg_price']) / info['avg_price']
            
            if return_rate <= STOP_LOSS:
                sell_price = current_price * (1 - SLIPPAGE)
                sell_amount = info['shares'] * sell_price
                commission = sell_amount * SELL_COMMISSION
                cash += sell_amount - commission
                
                trades.append({
                    'date': date,
                    'symbol': symbol,
                    'action': 'STOP_LOSS',
                    'shares': info['shares'],
                    'price': sell_price,
                    'amount': sell_amount,
                    'commission': commission,
                    'slippage': current_price * SLIPPAGE * info['shares'],
                    'return_rate': return_rate
                })
                
                del holdings[symbol]
        
        # ----- 매수 주문 실행 (수요일) -----
        if pending_order is not None and pending_order['trade_date'] == date:
            order = pending_order
            pending_order = None
            
            new_picks = order['picks']
            new_scores = order['scores']
            new_allocations = order['allocations']
            
            current_holdings = set(holdings.keys())
            new_holdings_set = set(new_picks)
            
            to_sell = current_holdings - new_holdings_set
            to_buy = new_holdings_set - current_holdings
            to_keep = current_holdings & new_holdings_set
            
            # 매도
            for symbol in to_sell:
                if symbol not in holdings:
                    continue
                
                info = holdings[symbol]
                stock = today_data[today_data['symbol'] == symbol]
                
                if not stock.empty:
                    base_price = stock.iloc[0]['close']
                    sell_price = base_price * (1 - SLIPPAGE)
                    sell_amount = info['shares'] * sell_price
                    commission = sell_amount * SELL_COMMISSION
                    cash += sell_amount - commission
                    
                    return_rate = (sell_price - info['avg_price']) / info['avg_price']
                    
                    trades.append({
                        'date': date,
                        'symbol': symbol,
                        'action': 'SELL',
                        'shares': info['shares'],
                        'price': sell_price,
                        'amount': sell_amount,
                        'commission': commission,
                        'slippage': base_price * SLIPPAGE * info['shares'],
                        'return_rate': return_rate
                    })
                    
                    del holdings[symbol]
            
            # 목표 비중
            target_allocations = {}
            for j, symbol in enumerate(new_picks):
                if j < len(new_allocations):
                    target_allocations[symbol] = new_allocations[j]
            
            # 유지 종목 비중 조절
            for symbol in to_keep:
                if symbol not in holdings or symbol not in target_allocations:
                    continue
                
                stock = today_data[today_data['symbol'] == symbol]
                if stock.empty:
                    continue
                
                current_price = stock.iloc[0]['close']
                current_value = holdings[symbol]['shares'] * current_price
                target_value = portfolio_value * target_allocations[symbol]
                
                diff_value = target_value - current_value
                diff_shares = int(abs(diff_value) / current_price)
                
                score_idx = new_picks.index(symbol) if symbol in new_picks else -1
                score = new_scores[score_idx] if 0 <= score_idx < len(new_scores) else 0
                
                if abs(diff_value) / portfolio_value > 0.05 and diff_shares > 0:
                    if diff_value > 0:
                        buy_price = current_price * (1 + SLIPPAGE)
                        buy_amount = diff_shares * buy_price
                        commission = buy_amount * BUY_COMMISSION
                        
                        if cash >= buy_amount + commission:
                            cash -= (buy_amount + commission)
                            
                            old_shares = holdings[symbol]['shares']
                            old_avg = holdings[symbol]['avg_price']
                            new_shares = old_shares + diff_shares
                            new_avg = (old_avg * old_shares + buy_amount) / new_shares
                            
                            holdings[symbol]['shares'] = new_shares
                            holdings[symbol]['avg_price'] = new_avg
                            
                            trades.append({
                                'date': date,
                                'symbol': symbol,
                                'action': 'ADD',
                                'shares': diff_shares,
                                'price': buy_price,
                                'amount': buy_amount,
                                'commission': commission,
                                'slippage': current_price * SLIPPAGE * diff_shares,
                                'return_rate': 0,
                                'score': score
                            })
                    else:
                        sell_price = current_price * (1 - SLIPPAGE)
                        sell_amount = diff_shares * sell_price
                        commission = sell_amount * SELL_COMMISSION
                        cash += sell_amount - commission
                        holdings[symbol]['shares'] -= diff_shares
                        
                        trades.append({
                            'date': date,
                            'symbol': symbol,
                            'action': 'REDUCE',
                            'shares': diff_shares,
                            'price': sell_price,
                            'amount': sell_amount,
                            'commission': commission,
                            'slippage': current_price * SLIPPAGE * diff_shares,
                            'return_rate': 0,
                            'score': score
                        })
            
            # 신규 매수
            for symbol in to_buy:
                if symbol not in target_allocations:
                    continue
                
                stock = today_data[today_data['symbol'] == symbol]
                if stock.empty:
                    continue
                
                base_price = stock.iloc[0]['close']
                buy_price = base_price * (1 + SLIPPAGE)
                
                if pd.isna(buy_price):
                    continue
                
                allocation = target_allocations[symbol]
                invest_amount = portfolio_value * allocation
                shares = int(invest_amount / buy_price)
                
                if shares <= 0:
                    continue
                
                buy_amount = shares * buy_price
                commission = buy_amount * BUY_COMMISSION
                
                if cash >= buy_amount + commission:
                    cash -= (buy_amount + commission)
                    
                    holdings[symbol] = {
                        'shares': shares,
                        'avg_price': buy_price
                    }
                    
                    score_idx = new_picks.index(symbol) if symbol in new_picks else -1
                    score = new_scores[score_idx] if 0 <= score_idx < len(new_scores) else 0
                    
                    trades.append({
                        'date': date,
                        'symbol': symbol,
                        'action': 'BUY',
                        'shares': shares,
                        'price': buy_price,
                        'amount': buy_amount,
                        'commission': commission,
                        'slippage': base_price * SLIPPAGE * shares,
                        'return_rate': 0,
                        'score': score
                    })
        
        # ----- 화요일: 종목 선정 -----
        if date_ts not in score_dates:
            continue
        
        if date not in trade_map:
            continue
        
        trade_date = trade_map[date]
        
        # 전략으로 종목 선정
        result = strategy.select_stocks(score_df, correlation_df, date_ts, ret_1m)
        
        if result is not None:
            pending_order = {
                'score_date': date,
                'trade_date': trade_date,
                'picks': result['picks'],
                'scores': result['scores'],
                'allocations': result['allocations']
            }
    
    # ===== 결과 정리 =====
    portfolio_df = pd.DataFrame(portfolio_values)
    trades_df = pd.DataFrame(trades) if trades else pd.DataFrame()
    metrics = calculate_metrics(portfolio_df, trades_df, df)
    
    print("\n" + "=" * 60)
    print("✅ 백테스트 완료!")
    print("=" * 60)
    
    return {
        'portfolio': portfolio_df,
        'trades': trades_df,
        'metrics': metrics
    }


# ============================================
# [2] 성과 지표 계산
# ============================================

def calculate_metrics(portfolio_df, trades_df, df):
    """성과 지표 계산"""
    values = portfolio_df['value'].values
    dates = portfolio_df['date']
    
    initial = values[0]
    final = values[-1]
    total_return = (final - initial) / initial
    
    daily_returns = pd.Series(values).pct_change().dropna()
    
    days = (dates.iloc[-1] - dates.iloc[0]).days
    years = days / 365
    cagr = (final / initial) ** (1 / years) - 1 if years > 0 else 0
    
    volatility = daily_returns.std() * np.sqrt(252)
    sharpe = (cagr - 0.03) / volatility if volatility > 0 else 0
    
    peak = pd.Series(values).cummax()
    drawdown = (pd.Series(values) - peak) / peak
    mdd = drawdown.min()
    
    win_rate = (daily_returns > 0).mean()
    
    spy_return = 0
    if 'SPY' in df['symbol'].unique():
        spy = df[df['symbol'] == 'SPY'].sort_values('date')
        if len(spy) >= 2:
            spy_return = (spy.iloc[-1]['close'] - spy.iloc[0]['close']) / spy.iloc[0]['close']
    
    total_trades = len(trades_df) if not trades_df.empty else 0
    total_commission = trades_df['commission'].sum() if not trades_df.empty else 0
    total_slippage = trades_df['slippage'].sum() if not trades_df.empty and 'slippage' in trades_df.columns else 0
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
# [3] 결과 출력
# ============================================

def print_metrics(metrics, trades_df=None):
    """성과 지표 출력"""
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
    print(f"    - 전량 매도 (SELL): {metrics['sell_count']}회")
    print(f"    - 추가 매수 (ADD): {metrics['add_count']}회")
    print(f"    - 일부 매도 (REDUCE): {metrics['reduce_count']}회")
    print(f"    - 손절 (STOP_LOSS): {metrics['stop_loss_count']}회")
    print(f"  총 수수료: ${metrics['total_commission']:,.2f}")
    print(f"  총 슬리피지: ${metrics['total_slippage']:,.2f}")
    
    print(f"\n📅 기타")
    print(f"  승률 (일 기준): {metrics['win_rate']*100:.2f}%")
    
    if trades_df is not None and not trades_df.empty:
        buy_trades = trades_df[trades_df['action'].isin(['BUY', 'ADD'])].copy()
        if not buy_trades.empty:
            recent_dates = buy_trades['date'].drop_duplicates().sort_values(ascending=False).head(5)
            print(f"\n🛒 최근 매수 내역")
            print("-" * 60)
            for buy_date in recent_dates:
                date_buys = buy_trades[buy_trades['date'] == buy_date]
                if 'score' in date_buys.columns:
                    date_buys = date_buys.sort_values('score', ascending=False)
                print(f"\n📅 {buy_date.strftime('%Y-%m-%d')}")
                for _, row in date_buys.iterrows():
                    score = row.get('score', 0)
                    print(f"  {row['action']:5} {row['symbol']:5} | 점수: {score:.4f} | ${row['amount']:,.2f}")
    
    print("\n" + "=" * 60)


# ============================================
# [4] 그래프 출력
# ============================================

def plot_results(portfolio_df, trades_df, df, figsize=(14, 10)):
    """결과 그래프"""
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    # 포트폴리오 vs SPY
    ax1 = axes[0, 0]
    portfolio_df = portfolio_df.copy()
    portfolio_df['normalized'] = portfolio_df['value'] / portfolio_df['value'].iloc[0] * 100
    ax1.plot(portfolio_df['date'], portfolio_df['normalized'], label='Portfolio', linewidth=2, color='blue')
    
    if 'SPY' in df['symbol'].unique():
        spy = df[df['symbol'] == 'SPY'].sort_values('date').copy()
        spy['normalized'] = spy['close'] / spy['close'].iloc[0] * 100
        ax1.plot(spy['date'], spy['normalized'], label='SPY', linewidth=2, linestyle='--', color='orange')
    
    ax1.set_title('Portfolio vs SPY', fontsize=12)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 일별 수익률
    ax2 = axes[0, 1]
    daily_returns = portfolio_df['value'].pct_change().dropna()
    colors = ['green' if r > 0 else 'red' for r in daily_returns]
    ax2.bar(range(len(daily_returns)), daily_returns, color=colors, alpha=0.7)
    ax2.set_title('Daily Returns', fontsize=12)
    ax2.grid(True, alpha=0.3)
    
    # 누적 수익률
    ax3 = axes[1, 0]
    cumulative = (1 + daily_returns).cumprod() - 1
    ax3.fill_between(range(len(cumulative)), cumulative, alpha=0.3, color='blue')
    ax3.plot(range(len(cumulative)), cumulative, linewidth=2, color='blue')
    ax3.set_title('Cumulative Returns', fontsize=12)
    ax3.grid(True, alpha=0.3)
    
    # Drawdown
    ax4 = axes[1, 1]
    peak = portfolio_df['value'].cummax()
    drawdown = (portfolio_df['value'] - peak) / peak
    ax4.fill_between(portfolio_df['date'], drawdown, 0, color='red', alpha=0.3)
    ax4.set_title('Drawdown', fontsize=12)
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
