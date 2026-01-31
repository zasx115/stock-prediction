# ============================================
# 파일명: src/backtest.py
# 설명: 백테스트 (과거 데이터로 전략 검증)
# ============================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# 전략 불러오기
from src.strategy import calculate_momentum_score, MIN_SCORE, TOP_N, ALLOCATIONS, MARKET_FILTER


# ============================================
# 설정 (수정하기 쉽게 변수로 분리)
# ============================================

INITIAL_CAPITAL = 10000000   # 초기 자본금 (1000만원)

# ----- 수수료 -----
BUY_COMMISSION = 0.0025      # 매수 수수료 (0.25%)
SELL_COMMISSION = 0.0025     # 매도 수수료 (0.25%)

# ----- 손절 -----
STOP_LOSS = -0.05            # 손절 기준 (-5%)


# ============================================
# 1. 단일 날짜 전략 실행
# ============================================

def run_strategy_for_date(df, target_date):
    """
    특정 날짜 기준으로 전략을 실행합니다.
    
    Args:
        df: 전체 주가 데이터
        target_date: 기준 날짜
    
    Returns:
        dict: 전략 결과 (signal, picks, allocations, scores)
    """
    # 해당 날짜까지의 데이터만 사용
    df_until = df[df['date'] <= target_date].copy()
    
    if df_until.empty:
        return {'signal': 'HOLD', 'picks': [], 'allocations': [], 'pick_scores': []}
    
    # 모멘텀 점수 계산
    scores_df = calculate_momentum_score(df_until)
    
    if scores_df.empty:
        return {'signal': 'HOLD', 'picks': [], 'allocations': [], 'pick_scores': []}
    
    # 시장 필터링 (평균 수익률 체크)
    if MARKET_FILTER:
        returns = []
        for symbol in df_until['symbol'].unique():
            stock = df_until[df_until['symbol'] == symbol]
            if len(stock) >= 2:
                today_close = stock.iloc[-1]['close']
                yesterday_close = stock.iloc[-2]['close']
                daily_return = (today_close - yesterday_close) / yesterday_close
                returns.append(daily_return)
        
        avg_return = np.mean(returns) if returns else 0
        
        if avg_return <= 0:
            return {'signal': 'HOLD', 'picks': [], 'allocations': [], 'pick_scores': [], 'market_return': avg_return}
    
    # 상위 종목 선정
    top_stocks = scores_df.head(TOP_N)
    qualified = top_stocks[top_stocks['score'] >= MIN_SCORE]
    
    if len(qualified) == 0:
        return {'signal': 'HOLD', 'picks': [], 'allocations': [], 'pick_scores': []}
    
    # 매수 신호
    picks = qualified['symbol'].tolist()
    pick_scores = qualified['score'].tolist()
    
    n_picks = len(picks)
    if n_picks >= 3:
        allocations = ALLOCATIONS[:3]
    elif n_picks == 2:
        allocations = [0.5, 0.5]
    else:
        allocations = [1.0]
    
    return {
        'signal': 'BUY',
        'picks': picks,
        'allocations': allocations,
        'pick_scores': pick_scores
    }


# ============================================
# 2. 손절 체크
# ============================================

def check_stop_loss(holdings, today_data):
    """
    손절 대상 종목을 확인합니다.
    
    Args:
        holdings: 보유 종목 딕셔너리
        today_data: 오늘 주가 데이터
    
    Returns:
        list: 손절할 종목 리스트
    """
    stop_loss_list = []
    
    for symbol, info in holdings.items():
        stock_today = today_data[today_data['symbol'] == symbol]
        
        if stock_today.empty:
            continue
        
        current_price = stock_today.iloc[0]['close']
        avg_price = info['avg_price']
        
        # 수익률 계산
        return_rate = (current_price - avg_price) / avg_price
        
        # 손절 기준 이하면 리스트에 추가
        if return_rate <= STOP_LOSS:
            stop_loss_list.append({
                'symbol': symbol,
                'return_rate': return_rate,
                'current_price': current_price
            })
    
    return stop_loss_list


# ============================================
# 3. 백테스트 실행 (메인)
# ============================================

def run_backtest(df, rebalance_days=5):
    """
    백테스트를 실행합니다.
    
    Args:
        df: 주가 데이터 (get_backtest_data 결과)
        rebalance_days: 리밸런싱 주기 (일)
    
    Returns:
        dict: {
            'portfolio': 일별 포트폴리오 가치,
            'trades': 거래 내역,
            'metrics': 성과 지표
        }
    """
    print("=" * 50)
    print("[백테스트 실행]")
    print(f"초기 자본금: {INITIAL_CAPITAL:,}원")
    print(f"리밸런싱 주기: {rebalance_days}일")
    print(f"매수 수수료: {BUY_COMMISSION*100:.2f}%")
    print(f"매도 수수료: {SELL_COMMISSION*100:.2f}%")
    print(f"손절 기준: {STOP_LOSS*100:.1f}%")
    print("=" * 50)
    
    df = df.copy()
    df = df.sort_values('date').reset_index(drop=True)
    
    # 날짜 리스트
    dates = sorted(df['date'].unique())
    
    # 결과 저장
    portfolio_values = []    # 일별 포트폴리오 가치
    trades = []              # 거래 내역
    
    # 현재 상태
    cash = INITIAL_CAPITAL   # 현금
    holdings = {}            # 보유 종목 {symbol: {'shares': 주수, 'avg_price': 평균단가}}
    last_rebalance = None    # 마지막 리밸런싱 날짜
    
    print(f"\n{len(dates)}일간 시뮬레이션 시작...")
    
    for i, date in enumerate(dates):
        # 진행 상황 출력
        if (i + 1) % 50 == 0:
            print(f"  진행중... {i + 1}/{len(dates)} ({(i+1)/len(dates)*100:.1f}%)")
        
        # 오늘 데이터
        today_data = df[df['date'] == date]
        
        # ----- 포트폴리오 가치 계산 -----
        portfolio_value = cash
        for symbol, info in holdings.items():
            stock_today = today_data[today_data['symbol'] == symbol]
            if not stock_today.empty:
                current_price = stock_today.iloc[0]['close']
                portfolio_value += info['shares'] * current_price
        
        portfolio_values.append({
            'date': date,
            'value': portfolio_value,
            'cash': cash
        })
        
        # ----- 손절 체크 (매일) -----
        stop_loss_list = check_stop_loss(holdings, today_data)
        
        for sl in stop_loss_list:
            symbol = sl['symbol']
            info = holdings[symbol]
            sell_price = sl['current_price']
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
                'return_rate': sl['return_rate']
            })
            
            del holdings[symbol]
        
        # ----- 리밸런싱 체크 -----
        do_rebalance = False
        
        if last_rebalance is None:
            do_rebalance = True
        else:
            days_since = (date - last_rebalance).days
            if days_since >= rebalance_days:
                do_rebalance = True
        
        if not do_rebalance:
            continue
        
        # ----- 전략 실행 -----
        result = run_strategy_for_date(df, date)
        
        if result['signal'] == 'HOLD':
            continue
        
        # ----- 기존 보유 종목 매도 (손절 제외) -----
        for symbol, info in list(holdings.items()):
            stock_today = today_data[today_data['symbol'] == symbol]
            if not stock_today.empty:
                sell_price = stock_today.iloc[0]['close']
                sell_amount = info['shares'] * sell_price
                commission = sell_amount * SELL_COMMISSION
                cash += sell_amount - commission
                
                # 수익률 계산
                return_rate = (sell_price - info['avg_price']) / info['avg_price']
                
                trades.append({
                    'date': date,
                    'symbol': symbol,
                    'action': 'SELL',
                    'shares': info['shares'],
                    'price': sell_price,
                    'amount': sell_amount,
                    'commission': commission,
                    'return_rate': return_rate
                })
        
        holdings = {}
        
        # ----- 새 종목 매수 -----
        for symbol, allocation in zip(result['picks'], result['allocations']):
            stock_today = today_data[today_data['symbol'] == symbol]
            if stock_today.empty:
                continue
            
            buy_price = stock_today.iloc[0]['close']
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
                
                trades.append({
                    'date': date,
                    'symbol': symbol,
                    'action': 'BUY',
                    'shares': shares,
                    'price': buy_price,
                    'amount': buy_amount,
                    'commission': commission,
                    'return_rate': 0
                })
        
        last_rebalance = date
    
    # ----- 결과 정리 -----
    portfolio_df = pd.DataFrame(portfolio_values)
    trades_df = pd.DataFrame(trades) if trades else pd.DataFrame()
    
    # 성과 지표 계산
    metrics = calculate_metrics(portfolio_df, trades_df, df)
    
    print("\n" + "=" * 50)
    print("백테스트 완료!")
    print("=" * 50)
    
    return {
        'portfolio': portfolio_df,
        'trades': trades_df,
        'metrics': metrics
    }


# ============================================
# 4. 성과 지표 계산
# ============================================

def calculate_metrics(portfolio_df, trades_df, df):
    """
    백테스트 성과 지표를 계산합니다.
    
    Returns:
        dict: 각종 성과 지표
    """
    values = portfolio_df['value'].values
    dates = portfolio_df['date'].values
    
    # 기본 수익률
    initial = values[0]
    final = values[-1]
    total_return = (final - initial) / initial
    
    # 일별 수익률
    daily_returns = pd.Series(values).pct_change().dropna()
    
    # 연환산 수익률 (CAGR)
    days = (dates[-1] - dates[0]).astype('timedelta64[D]').astype(int)
    years = days / 365
    cagr = (final / initial) ** (1 / years) - 1 if years > 0 else 0
    
    # 변동성 (연환산)
    volatility = daily_returns.std() * np.sqrt(252)
    
    # 샤프 비율 (무위험 수익률 3% 가정)
    risk_free = 0.03
    sharpe = (cagr - risk_free) / volatility if volatility > 0 else 0
    
    # 최대 낙폭 (MDD)
    peak = pd.Series(values).cummax()
    drawdown = (pd.Series(values) - peak) / peak
    mdd = drawdown.min()
    
    # 승률 (일 기준)
    win_rate = (daily_returns > 0).sum() / len(daily_returns) if len(daily_returns) > 0 else 0
    
    # SPY 수익률 (벤치마크)
    spy_return = 0
    if 'SPY' in df['symbol'].unique():
        spy = df[df['symbol'] == 'SPY'].sort_values('date')
        if len(spy) >= 2:
            spy_initial = spy.iloc[0]['close']
            spy_final = spy.iloc[-1]['close']
            spy_return = (spy_final - spy_initial) / spy_initial
    
    # 거래 통계
    total_trades = 0
    total_commission = 0
    stop_loss_count = 0
    
    if not trades_df.empty:
        total_trades = len(trades_df)
        total_commission = trades_df['commission'].sum()
        stop_loss_count = len(trades_df[trades_df['action'] == 'STOP_LOSS'])
    
    metrics = {
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
        'total_commission': total_commission,
        'stop_loss_count': stop_loss_count
    }
    
    return metrics


# ============================================
# 5. 결과 출력
# ============================================

def print_metrics(metrics):
    """
    성과 지표를 보기 좋게 출력합니다.
    """
    print("\n" + "=" * 50)
    print("📊 백테스트 성과")
    print("=" * 50)
    
    print(f"\n💰 수익")
    print(f"  초기 자본금: {metrics['initial_capital']:,.0f}원")
    print(f"  최종 자본금: {metrics['final_capital']:,.0f}원")
    print(f"  총 수익률: {metrics['total_return']*100:.2f}%")
    print(f"  연환산 수익률 (CAGR): {metrics['cagr']*100:.2f}%")
    
    print(f"\n📈 벤치마크 비교")
    print(f"  SPY 수익률: {metrics['spy_return']*100:.2f}%")
    print(f"  초과 수익 (Alpha): {metrics['alpha']*100:.2f}%")
    
    print(f"\n⚠️ 위험")
    print(f"  변동성: {metrics['volatility']*100:.2f}%")
    print(f"  최대 낙폭 (MDD): {metrics['mdd']*100:.2f}%")
    print(f"  샤프 비율: {metrics['sharpe_ratio']:.2f}")
    
    print(f"\n🎯 거래")
    print(f"  총 거래 횟수: {metrics['total_trades']}회")
    print(f"  총 수수료: {metrics['total_commission']:,.0f}원")
    print(f"  손절 횟수: {metrics['stop_loss_count']}회")
    
    print(f"\n📅 기타")
    print(f"  승률 (일 기준): {metrics['win_rate']*100:.2f}%")
    
    print("=" * 50)


# ============================================
# 6. 그래프 출력 (Colab용)
# ============================================

def plot_results(portfolio_df, df, figsize=(14, 10)):
    """
    백테스트 결과를 그래프로 출력합니다.
    Colab에서 사용하세요.
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    # ----- 1. 포트폴리오 가치 vs SPY -----
    ax1 = axes[0, 0]
    
    portfolio_df['normalized'] = portfolio_df['value'] / portfolio_df['value'].iloc[0] * 100
    ax1.plot(portfolio_df['date'], portfolio_df['normalized'], label='Portfolio', linewidth=2)
    
    if 'SPY' in df['symbol'].unique():
        spy = df[df['symbol'] == 'SPY'].sort_values('date')
        spy['normalized'] = spy['close'] / spy['close'].iloc[0] * 100
        ax1.plot(spy['date'], spy['normalized'], label='SPY', linewidth=2, alpha=0.7)
    
    ax1.set_title('Portfolio vs SPY (Normalized to 100)', fontsize=12)
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Value')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # ----- 2. 일별 수익률 -----
    ax2 = axes[0, 1]
    
    daily_returns = portfolio_df['value'].pct_change().dropna()
    ax2.bar(range(len(daily_returns)), daily_returns, color=['green' if r > 0 else 'red' for r in daily_returns], alpha=0.7)
    ax2.axhline(y=0, color='black', linewidth=0.5)
    ax2.set_title('Daily Returns', fontsize=12)
    ax2.set_xlabel('Days')
    ax2.set_ylabel('Return')
    ax2.grid(True, alpha=0.3)
    
    # ----- 3. 누적 수익률 -----
    ax3 = axes[1, 0]
    
    cumulative = (1 + daily_returns).cumprod() - 1
    ax3.fill_between(range(len(cumulative)), cumulative, alpha=0.3)
    ax3.plot(range(len(cumulative)), cumulative, linewidth=2)
    ax3.axhline(y=0, color='black', linewidth=0.5)
    ax3.set_title('Cumulative Returns', fontsize=12)
    ax3.set_xlabel('Days')
    ax3.set_ylabel('Cumulative Return')
    ax3.grid(True, alpha=0.3)
    
    # ----- 4. Drawdown -----
    ax4 = axes[1, 1]
    
    peak = portfolio_df['value'].cummax()
    drawdown = (portfolio_df['value'] - peak) / peak
    ax4.fill_between(portfolio_df['date'], drawdown, 0, color='red', alpha=0.3)
    ax4.plot(portfolio_df['date'], drawdown, color='red', linewidth=1)
    ax4.set_title('Drawdown', fontsize=12)
    ax4.set_xlabel('Date')
    ax4.set_ylabel('Drawdown')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


# ============================================
# 테스트
# ============================================

if __name__ == "__main__":
    print("\n[테스트] 백테스트")
    print("실제 테스트는 Colab에서 data.py와 함께 실행하세요.")
