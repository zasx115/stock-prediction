# ============================================
# 파일명: src/ai_backtest.py
# 설명: AI 전략 백테스트
# 
# 기능:
# - AI 모델 기반 매매 시뮬레이션
# - 주간 단위 리밸런싱
# - 성과 분석 및 시각화
# ============================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# config.py에서 설정 가져오기
from config import (
    INITIAL_CAPITAL,
    STOP_LOSS,
    TOP_N,
    BUY_COMMISSION,
    SELL_COMMISSION,
    SLIPPAGE
)


# ============================================
# [1] 설정
# ============================================

# 매매 설정
REBALANCE_DAY = 'Tuesday'  # 리밸런싱 요일
TRADE_DAY = 'Wednesday'    # 실제 매매일
MIN_TRADE_AMOUNT = 50      # 최소 거래 금액


# ============================================
# [2] AI 백테스트 실행
# ============================================

def run_ai_backtest(strategy, test_df, feature_cols, initial_capital=INITIAL_CAPITAL):
    """
    AI 전략 백테스트 실행
    
    Args:
        strategy: AIStrategy 인스턴스 (학습 완료된)
        test_df: 테스트 데이터
        feature_cols: 피처 컬럼 리스트
        initial_capital: 초기 자본금
    
    Returns:
        dict: {portfolio, trades, metrics}
    """
    print("=" * 60)
    print("AI 백테스트 실행")
    print("=" * 60)
    print(f"초기 자본금: ${initial_capital:,}")
    print(f"리밸런싱: 매주 {REBALANCE_DAY}")
    print(f"수수료: {BUY_COMMISSION*100:.2f}% / {SELL_COMMISSION*100:.2f}%")
    print(f"슬리피지: {SLIPPAGE*100:.2f}%")
    print(f"손절: {STOP_LOSS*100:.1f}%")
    print("=" * 60)
    
    # 날짜 정렬
    test_df = test_df.copy()
    test_df['date'] = pd.to_datetime(test_df['date'])
    dates = sorted(test_df['date'].unique())
    
    print(f"테스트 기간: {dates[0].strftime('%Y-%m-%d')} ~ {dates[-1].strftime('%Y-%m-%d')}")
    print(f"총 {len(dates)}일")
    
    # 화요일 찾기
    tuesday_dates = [d for d in dates if d.day_name() == REBALANCE_DAY]
    print(f"리밸런싱 횟수: {len(tuesday_dates)}회")
    
    # 화요일 → 수요일 매핑
    trade_map = {}
    for tue in tuesday_dates:
        # 다음 수요일 찾기
        for d in dates:
            if d > tue and d.day_name() == TRADE_DAY:
                trade_map[tue] = d
                break
    
    # ----- 시뮬레이션 변수 -----
    portfolio_values = []
    trades = []
    
    cash = initial_capital
    holdings = {}  # {symbol: {'shares': int, 'avg_price': float}}
    pending_orders = []  # 대기 주문
    
    # ----- 일별 시뮬레이션 -----
    for date in dates:
        date_df = test_df[test_df['date'] == date]
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
                
                sell_price = current_price * (1 - SLIPPAGE)
                sell_amount = sell_price * info['shares']
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
                    'return_pct': return_rate * 100
                })
        
        for symbol in symbols_to_sell:
            del holdings[symbol]
        
        # ----- 2. 대기 주문 실행 (수요일) -----
        for order in pending_orders:
            if order['trade_date'] == date:
                symbol = order['symbol']
                target_alloc = order['allocation']
                
                if symbol not in price_dict:
                    continue
                
                current_price = price_dict[symbol]
                buy_price = current_price * (1 + SLIPPAGE)
                
                # 총 자산 계산
                total_value = cash + sum(
                    price_dict.get(s, info['avg_price']) * info['shares']
                    for s, info in holdings.items()
                )
                
                # 목표 금액
                target_amount = total_value * target_alloc
                
                # 현재 보유 금액
                current_amount = 0
                if symbol in holdings:
                    current_amount = holdings[symbol]['shares'] * current_price
                
                # 차이 계산
                diff = target_amount - current_amount
                
                if diff > MIN_TRADE_AMOUNT:  # 매수
                    buy_amount = min(diff, cash * 0.95)
                    commission = buy_amount * BUY_COMMISSION
                    
                    if cash >= buy_amount + commission:
                        shares = int(buy_amount / buy_price)
                        if shares > 0:
                            actual_amount = shares * buy_price
                            cash -= actual_amount + commission
                            
                            if symbol in holdings:
                                # 추가 매수
                                old = holdings[symbol]
                                new_shares = old['shares'] + shares
                                new_avg = (old['avg_price'] * old['shares'] + buy_price * shares) / new_shares
                                holdings[symbol] = {'shares': new_shares, 'avg_price': new_avg}
                                action = 'ADD'
                            else:
                                # 신규 매수
                                holdings[symbol] = {'shares': shares, 'avg_price': buy_price}
                                action = 'BUY'
                            
                            trades.append({
                                'date': date,
                                'symbol': symbol,
                                'action': action,
                                'shares': shares,
                                'price': buy_price,
                                'amount': actual_amount,
                                'commission': commission,
                                'probability': order.get('score', 0),
                                'return_pct': 0
                            })
                
                elif diff < -MIN_TRADE_AMOUNT:  # 비중 축소
                    if symbol in holdings:
                        sell_shares = int(abs(diff) / current_price)
                        sell_shares = min(sell_shares, holdings[symbol]['shares'])
                        
                        if sell_shares > 0:
                            sell_price = current_price * (1 - SLIPPAGE)
                            sell_amount = sell_shares * sell_price
                            commission = sell_amount * SELL_COMMISSION
                            cash += sell_amount - commission
                            
                            avg_price = holdings[symbol]['avg_price']
                            ret_pct = (sell_price - avg_price) / avg_price * 100
                            
                            holdings[symbol]['shares'] -= sell_shares
                            if holdings[symbol]['shares'] <= 0:
                                del holdings[symbol]
                            
                            trades.append({
                                'date': date,
                                'symbol': symbol,
                                'action': 'REDUCE',
                                'shares': sell_shares,
                                'price': sell_price,
                                'amount': sell_amount,
                                'commission': commission,
                                'return_pct': ret_pct
                            })
        
        # 실행된 주문 제거
        pending_orders = [o for o in pending_orders if o['trade_date'] != date]
        
        # ----- 3. 화요일: 신호 생성 및 주문 예약 -----
        if date in trade_map:
            trade_date = trade_map[date]
            
            # AI 모델로 종목 선정
            result = strategy.select_stocks(test_df, feature_cols, date)
            
            if result is not None:
                # 매수 주문 예약
                for i, symbol in enumerate(result['picks']):
                    pending_orders.append({
                        'symbol': symbol,
                        'score': result['scores'][i],
                        'allocation': result['allocations'][i],
                        'trade_date': trade_date
                    })
                
                # 보유 중이지만 새 리스트에 없는 종목 → 매도 예약
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
    
    # ----- 결과 정리 -----
    portfolio_df = pd.DataFrame(portfolio_values)
    trades_df = pd.DataFrame(trades) if trades else pd.DataFrame()
    
    # SPY 데이터 추출
    spy_df = test_df[test_df['symbol'] == 'SPY'][['date', 'close']].copy()
    spy_df = spy_df.rename(columns={'close': 'spy_close'})
    
    # 성과 지표 계산
    metrics = calculate_ai_metrics(portfolio_df, trades_df, spy_df, initial_capital)
    
    print("\n" + "=" * 60)
    print("✅ AI 백테스트 완료!")
    print("=" * 60)
    
    return {
        'portfolio': portfolio_df,
        'trades': trades_df,
        'metrics': metrics
    }


# ============================================
# [3] 성과 지표 계산
# ============================================

def calculate_ai_metrics(portfolio_df, trades_df, spy_df, initial_capital):
    """
    백테스트 성과 지표 계산
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
    years = days / 365 if days > 0 else 1
    cagr = (final / initial) ** (1 / years) - 1 if years > 0 else 0
    
    # 변동성 & 샤프
    volatility = daily_returns.std() * np.sqrt(252)
    risk_free_rate = 0.03
    sharpe = (cagr - risk_free_rate) / volatility if volatility > 0 else 0
    
    # MDD
    peak = pd.Series(values).cummax()
    drawdown = (pd.Series(values) - peak) / peak
    mdd = drawdown.min()
    
    # SPY 수익률
    spy_return = 0
    if len(spy_df) >= 2:
        spy_df = spy_df.sort_values('date')
        spy_initial = spy_df.iloc[0]['spy_close']
        spy_final = spy_df.iloc[-1]['spy_close']
        spy_return = (spy_final - spy_initial) / spy_initial
    
    # 거래 통계
    total_trades = len(trades_df) if not trades_df.empty else 0
    total_commission = trades_df['commission'].sum() if not trades_df.empty else 0
    
    buy_count = len(trades_df[trades_df['action'] == 'BUY']) if not trades_df.empty else 0
    add_count = len(trades_df[trades_df['action'] == 'ADD']) if not trades_df.empty else 0
    sell_count = len(trades_df[trades_df['action'].isin(['SELL', 'REDUCE'])])if not trades_df.empty else 0
    stop_loss_count = len(trades_df[trades_df['action'] == 'STOP_LOSS']) if not trades_df.empty else 0
    
    # 승률 (매도 거래 기준)
    win_rate = 0
    if not trades_df.empty:
        sell_trades = trades_df[trades_df['action'].isin(['SELL', 'REDUCE', 'STOP_LOSS'])]
        if len(sell_trades) > 0:
            wins = (sell_trades['return_pct'] > 0).sum()
            win_rate = wins / len(sell_trades)
    
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
        'add_count': add_count,
        'sell_count': sell_count,
        'stop_loss_count': stop_loss_count,
        'total_commission': total_commission
    }


# ============================================
# [4] 결과 출력
# ============================================

def print_ai_metrics(metrics):
    """
    AI 백테스트 성과 출력
    """
    print("\n" + "=" * 60)
    print("📊 AI 백테스트 성과")
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
    print(f"    - 매도 (SELL/REDUCE): {metrics['sell_count']}회")
    print(f"    - 손절 (STOP_LOSS): {metrics['stop_loss_count']}회")
    print(f"  승률: {metrics['win_rate']*100:.1f}%")
    print(f"  총 수수료: ${metrics['total_commission']:,.2f}")
    
    print("\n" + "=" * 60)


# ============================================
# [5] 시각화
# ============================================

def plot_ai_results(portfolio_df, spy_df=None, figsize=(14, 10)):
    """
    AI 백테스트 결과 시각화
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    # ----- 1. 포트폴리오 vs SPY -----
    ax1 = axes[0, 0]
    
    portfolio_df = portfolio_df.copy()
    portfolio_df['normalized'] = portfolio_df['value'] / portfolio_df['value'].iloc[0] * 100
    ax1.plot(portfolio_df['date'], portfolio_df['normalized'], 
             label='AI Portfolio', linewidth=2, color='blue')
    
    if spy_df is not None and len(spy_df) > 0:
        spy_df = spy_df.sort_values('date').copy()
        spy_df['normalized'] = spy_df['spy_close'] / spy_df['spy_close'].iloc[0] * 100
        ax1.plot(spy_df['date'], spy_df['normalized'], 
                 label='SPY', linewidth=2, linestyle='--', color='orange')
    
    ax1.set_title('AI Portfolio vs SPY', fontsize=12)
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
# [6] 모멘텀 vs AI 비교
# ============================================

def compare_strategies(momentum_result, ai_result):
    """
    모멘텀 전략과 AI 전략 비교
    
    Args:
        momentum_result: 모멘텀 백테스트 결과
        ai_result: AI 백테스트 결과
    """
    print("\n" + "=" * 60)
    print("📊 전략 비교: 모멘텀 vs AI")
    print("=" * 60)
    
    m = momentum_result['metrics']
    a = ai_result['metrics']
    
    print(f"\n{'지표':20} | {'모멘텀':>12} | {'AI':>12} | {'차이':>12}")
    print("-" * 60)
    print(f"{'총 수익률':20} | {m['total_return']*100:>11.2f}% | {a['total_return']*100:>11.2f}% | {(a['total_return']-m['total_return'])*100:>+11.2f}%")
    print(f"{'CAGR':20} | {m['cagr']*100:>11.2f}% | {a['cagr']*100:>11.2f}% | {(a['cagr']-m['cagr'])*100:>+11.2f}%")
    print(f"{'Alpha':20} | {m['alpha']*100:>11.2f}% | {a['alpha']*100:>11.2f}% | {(a['alpha']-m['alpha'])*100:>+11.2f}%")
    print(f"{'MDD':20} | {m['mdd']*100:>11.2f}% | {a['mdd']*100:>11.2f}% | {(a['mdd']-m['mdd'])*100:>+11.2f}%")
    print(f"{'샤프 비율':20} | {m['sharpe_ratio']:>12.2f} | {a['sharpe_ratio']:>12.2f} | {a['sharpe_ratio']-m['sharpe_ratio']:>+12.2f}")
    print(f"{'승률':20} | {m['win_rate']*100:>11.1f}% | {a['win_rate']*100:>11.1f}% | {(a['win_rate']-m['win_rate'])*100:>+11.1f}%")
    print(f"{'총 거래':20} | {m['total_trades']:>12} | {a['total_trades']:>12} | {a['total_trades']-m['total_trades']:>+12}")
    print("-" * 60)
    
    # 승자 판정
    ai_wins = 0
    if a['total_return'] > m['total_return']: ai_wins += 1
    if a['alpha'] > m['alpha']: ai_wins += 1
    if a['mdd'] > m['mdd']: ai_wins += 1  # MDD는 클수록 좋음 (덜 빠짐)
    if a['sharpe_ratio'] > m['sharpe_ratio']: ai_wins += 1
    
    if ai_wins >= 3:
        print("\n🏆 AI 전략 승리!")
    elif ai_wins <= 1:
        print("\n🏆 모멘텀 전략 승리!")
    else:
        print("\n🤝 비슷한 성과")


# ============================================
# [7] 테스트
# ============================================

if __name__ == "__main__":
    print("AI Backtest 모듈")
    print("=" * 60)
    print("\nColab에서 실행:")
    print()
    print("# 1. 데이터 및 모델 준비")
    print("from ai_data import prepare_ai_data")
    print("from ai_strategy import AIStrategy")
    print("from ai_backtest import run_ai_backtest, print_ai_metrics, plot_ai_results")
    print()
    print("train_df, test_df, features = prepare_ai_data()")
    print("strategy = AIStrategy()")
    print("strategy.train(train_df, features)")
    print()
    print("# 2. 백테스트 실행")
    print("result = run_ai_backtest(strategy, test_df, features)")
    print("print_ai_metrics(result['metrics'])")
    print("plot_ai_results(result['portfolio'])")