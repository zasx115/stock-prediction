# ============================================
# 파일명: src/backtest.py
# 설명: 백테스트 (최적화 버전)
# 
# 최적화 포인트:
# - 모멘텀 점수를 한 번에 미리 계산
# - 시장 수익률을 한 번에 미리 계산
# - 백테스트 중에는 조회만 (빠름!)
# ============================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# ============================================
# 설정 (수정하기 쉽게 변수로 분리)
# ============================================

# ----- 자본금 -----
INITIAL_CAPITAL = 10000000   # 초기 자본금 (1000만원)

# ----- 수수료 -----
BUY_COMMISSION = 0.0025      # 매수 수수료 (0.25%)
SELL_COMMISSION = 0.0025     # 매도 수수료 (0.25%)

# ----- 손절 -----
STOP_LOSS = -0.05            # 손절 기준 (-5%)

# ----- 모멘텀 점수 가중치 -----
WEIGHT_2DAY = 3.5            # 2일전 수익률 가중치
WEIGHT_4DAY = 2.5            # 4일전 수익률 가중치
WEIGHT_6DAY = 1.5            # 6일전 수익률 가중치

# ----- 종목 선정 -----
TOP_N = 3                    # 상위 몇 개 종목 선정
ALLOCATIONS = [0.4, 0.3, 0.3]  # 투자 비중 (1위, 2위, 3위)

# ----- 필터링 조건 -----
MIN_SCORE = 0.01             # 최소 점수 (이 점수 이상이어야 매수)
MARKET_FILTER = True         # 시장 필터 사용 여부


# ============================================
# 1. 모멘텀 점수 사전 계산
# ============================================

def calc_all_momentum_scores(df):
    """
    모든 날짜의 모멘텀 점수를 한 번에 계산합니다.
    
    왜 필요한가?
    - 기존: 백테스트 매일 점수 계산 (느림)
    - 최적화: 미리 전부 계산해두고 조회만 (빠름)
    
    Args:
        df: 전체 주가 데이터
    
    Returns:
        DataFrame: 날짜, 종목, 종가, 점수 포함
    """
    print("모멘텀 점수 사전 계산 중...")
    
    df = df.copy()
    df = df.sort_values(['symbol', 'date']).reset_index(drop=True)
    
    results = []
    
    # 각 종목별로 처리
    for symbol in df['symbol'].unique():
        stock = df[df['symbol'] == symbol].copy().reset_index(drop=True)
        
        # 최소 7일 데이터 필요 (6일전 수익률 계산하려면)
        if len(stock) < 7:
            continue
        
        # 7일차부터 마지막 날까지 점수 계산
        for i in range(6, len(stock)):
            today = stock.iloc[i]
            today_close = today['close']
            today_date = today['date']
            
            # N일전 종가
            close_2d = stock.iloc[i-2]['close']
            close_4d = stock.iloc[i-4]['close']
            close_6d = stock.iloc[i-6]['close']
            
            # 수익률 계산: (오늘 - N일전) / N일전
            return_2d = (today_close - close_2d) / close_2d
            return_4d = (today_close - close_4d) / close_4d
            return_6d = (today_close - close_6d) / close_6d
            
            # 모멘텀 점수 = 가중 합계
            score = (return_2d * WEIGHT_2DAY) + (return_4d * WEIGHT_4DAY) + (return_6d * WEIGHT_6DAY)
            
            results.append({
                'date': today_date,
                'symbol': symbol,
                'close': today_close,
                'score': score
            })
    
    result_df = pd.DataFrame(results)
    print(f"✅ {len(result_df):,}개 점수 계산 완료!")
    
    return result_df


# ============================================
# 2. 시장 수익률 사전 계산
# ============================================

def calc_daily_market_returns(df):
    """
    모든 날짜의 시장 평균 수익률을 한 번에 계산합니다.
    
    시장 평균 수익률 = 전체 종목의 일일 수익률 평균
    이 값이 양수면 시장이 좋은 상태 → 매수 가능
    
    Args:
        df: 전체 주가 데이터
    
    Returns:
        DataFrame: 날짜, 시장 평균 수익률
    """
    print("시장 수익률 사전 계산 중...")
    
    df = df.copy()
    df = df.sort_values(['symbol', 'date']).reset_index(drop=True)
    
    # 종목별 일일 수익률 계산
    df['daily_return'] = df.groupby('symbol')['close'].pct_change()
    
    # 날짜별 평균 수익률
    market_returns = df.groupby('date')['daily_return'].mean().reset_index()
    market_returns.columns = ['date', 'market_return']
    
    print(f"✅ {len(market_returns)}일 시장 수익률 계산 완료!")
    
    return market_returns


# ============================================
# 3. 백테스트 실행 (메인)
# ============================================

def run_backtest(df, rebalance_days=5):
    """
    백테스트를 실행합니다.
    
    실행 순서:
    1. 모멘텀 점수 사전 계산 (한 번만)
    2. 시장 수익률 사전 계산 (한 번만)
    3. 날짜별 시뮬레이션
       - 포트폴리오 가치 계산
       - 손절 체크
       - 리밸런싱 (매수/매도)
    4. 성과 지표 계산
    
    Args:
        df: 주가 데이터
        rebalance_days: 리밸런싱 주기 (일)
    
    Returns:
        dict: portfolio(일별 가치), trades(거래 내역), metrics(성과 지표)
    """
    print("=" * 50)
    print("[백테스트 실행]")
    print(f"초기 자본금: {INITIAL_CAPITAL:,}원")
    print(f"리밸런싱 주기: {rebalance_days}일")
    print(f"매수 수수료: {BUY_COMMISSION*100:.2f}%")
    print(f"매도 수수료: {SELL_COMMISSION*100:.2f}%")
    print(f"손절 기준: {STOP_LOSS*100:.1f}%")
    print("=" * 50)
    
    # 데이터 정렬
    df = df.sort_values('date').reset_index(drop=True)
    dates = sorted(df['date'].unique())
    
    # ----- 핵심: 점수와 시장 수익률 미리 계산 -----
    all_scores = calc_all_momentum_scores(df)
    market_returns = calc_daily_market_returns(df)
    
    # 빠른 조회를 위해 딕셔너리로 변환
    market_dict = dict(zip(market_returns['date'], market_returns['market_return']))
    
    # 결과 저장용
    portfolio_values = []    # 일별 포트폴리오 가치
    trades = []              # 거래 내역
    
    # 현재 상태
    cash = INITIAL_CAPITAL   # 현금
    holdings = {}            # 보유 종목 {symbol: {shares, avg_price}}
    last_rebalance = None    # 마지막 리밸런싱 날짜
    
    print(f"\n{len(dates)}일 시뮬레이션 시작...")
    
    # ----- 날짜별 시뮬레이션 -----
    for i, date in enumerate(dates):
        
        # 진행 상황 출력 (50일마다)
        if (i + 1) % 50 == 0:
            print(f"  진행중... {i+1}/{len(dates)} ({(i+1)/len(dates)*100:.1f}%)")
        
        # 오늘 주가 데이터
        today_data = df[df['date'] == date]
        
        # ----- 포트폴리오 가치 계산 -----
        portfolio_value = cash
        for symbol, info in holdings.items():
            stock = today_data[today_data['symbol'] == symbol]
            if not stock.empty:
                current_price = stock.iloc[0]['close']
                portfolio_value += info['shares'] * current_price
        
        portfolio_values.append({
            'date': date,
            'value': portfolio_value,
            'cash': cash
        })
        
        # ----- 손절 체크 (매일) -----
        for symbol, info in list(holdings.items()):
            stock = today_data[today_data['symbol'] == symbol]
            if stock.empty:
                continue
            
            current_price = stock.iloc[0]['close']
            return_rate = (current_price - info['avg_price']) / info['avg_price']
            
            # 손절 기준 이하면 매도
            if return_rate <= STOP_LOSS:
                sell_amount = info['shares'] * current_price
                commission = sell_amount * SELL_COMMISSION
                cash += sell_amount - commission
                
                trades.append({
                    'date': date,
                    'symbol': symbol,
                    'action': 'STOP_LOSS',
                    'shares': info['shares'],
                    'price': current_price,
                    'amount': sell_amount,
                    'commission': commission,
                    'return_rate': return_rate
                })
                
                del holdings[symbol]
        
        # ----- 리밸런싱 체크 -----
        # 마지막 리밸런싱 후 N일 지났는지 확인
        if last_rebalance is not None:
            days_since = (date - last_rebalance).days
            if days_since < rebalance_days:
                continue  # 아직 리밸런싱 시기 아님
        
        # ----- 오늘 점수 조회 (미리 계산된 테이블에서) -----
        today_scores = all_scores[all_scores['date'] == date].copy()
        
        if today_scores.empty:
            continue  # 점수 없으면 스킵
        
        # ----- 시장 필터링 -----
        if MARKET_FILTER:
            market_ret = market_dict.get(date, 0)
            if market_ret <= 0:
                continue  # 시장 안 좋으면 매수 안 함
        
        # ----- 상위 종목 선정 -----
        today_scores = today_scores.sort_values('score', ascending=False)
        qualified = today_scores.head(TOP_N)
        qualified = qualified[qualified['score'] >= MIN_SCORE]
        
        if len(qualified) == 0:
            continue  # 조건 충족 종목 없음
        
        # 종목 리스트와 비중
        picks = qualified['symbol'].tolist()
        n_picks = len(picks)
        
        if n_picks >= 3:
            allocations = ALLOCATIONS[:3]
        elif n_picks == 2:
            allocations = [0.5, 0.5]
        else:
            allocations = [1.0]
        
        # ----- 기존 보유 종목 매도 -----
        for symbol, info in list(holdings.items()):
            stock = today_data[today_data['symbol'] == symbol]
            if not stock.empty:
                sell_price = stock.iloc[0]['close']
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
                    'return_rate': return_rate
                })
        
        holdings = {}
        
        # ----- 새 종목 매수 -----
        for symbol, allocation in zip(picks, allocations):
            stock = today_data[today_data['symbol'] == symbol]
            if stock.empty:
                continue
            
            buy_price = stock.iloc[0]['close']
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
    metrics = calculate_metrics(portfolio_df, trades_df, df)
    
    print("\n" + "=" * 50)
    print("✅ 백테스트 완료!")
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
    
    계산 지표:
    - 총 수익률, 연환산 수익률 (CAGR)
    - 변동성, 샤프 비율
    - 최대 낙폭 (MDD)
    - 승률
    - SPY 대비 초과 수익 (Alpha)
    - 거래 통계
    """
    values = portfolio_df['value'].values
    dates = portfolio_df['date']
    
    # 기본 수익률
    initial = values[0]
    final = values[-1]
    total_return = (final - initial) / initial
    
    # 일별 수익률
    daily_returns = pd.Series(values).pct_change().dropna()
    
    # 연환산 수익률 (CAGR)
    days = (dates.iloc[-1] - dates.iloc[0]).days
    years = days / 365
    cagr = (final / initial) ** (1 / years) - 1 if years > 0 else 0
    
    # 변동성 (연환산)
    volatility = daily_returns.std() * np.sqrt(252)
    
    # 샤프 비율 (무위험 수익률 3% 가정)
    sharpe = (cagr - 0.03) / volatility if volatility > 0 else 0
    
    # 최대 낙폭 (MDD)
    peak = pd.Series(values).cummax()
    drawdown = (pd.Series(values) - peak) / peak
    mdd = drawdown.min()
    
    # 승률 (일 기준)
    win_rate = (daily_returns > 0).mean()
    
    # SPY 수익률 (벤치마크)
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
    stop_loss_count = len(trades_df[trades_df['action'] == 'STOP_LOSS']) if not trades_df.empty else 0
    
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
        'total_commission': total_commission,
        'stop_loss_count': stop_loss_count
    }


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
    
    print(f"\n⚠️ 위험 지표")
    print(f"  변동성: {metrics['volatility']*100:.2f}%")
    print(f"  최대 낙폭 (MDD): {metrics['mdd']*100:.2f}%")
    print(f"  샤프 비율: {metrics['sharpe_ratio']:.2f}")
    
    print(f"\n🎯 거래 통계")
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
    
    그래프 4개:
    1. 포트폴리오 vs SPY (정규화)
    2. 일별 수익률
    3. 누적 수익률
    4. Drawdown (낙폭)
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    # ----- 1. 포트폴리오 vs SPY -----
    ax1 = axes[0, 0]
    
    # 정규화 (시작점 = 100)
    portfolio_df['normalized'] = portfolio_df['value'] / portfolio_df['value'].iloc[0] * 100
    ax1.plot(portfolio_df['date'], portfolio_df['normalized'], label='Portfolio', linewidth=2)
    
    # SPY도 같이 표시
    if 'SPY' in df['symbol'].unique():
        spy = df[df['symbol'] == 'SPY'].sort_values('date')
        spy['normalized'] = spy['close'] / spy['close'].iloc[0] * 100
        ax1.plot(spy['date'], spy['normalized'], label='SPY', linewidth=2, alpha=0.7)
    
    ax1.set_title('Portfolio vs SPY (시작=100 기준)', fontsize=12)
    ax1.set_xlabel('날짜')
    ax1.set_ylabel('가치')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # ----- 2. 일별 수익률 -----
    ax2 = axes[0, 1]
    
    daily_returns = portfolio_df['value'].pct_change().dropna()
    colors = ['green' if r > 0 else 'red' for r in daily_returns]
    ax2.bar(range(len(daily_returns)), daily_returns, color=colors, alpha=0.7)
    ax2.axhline(y=0, color='black', linewidth=0.5)
    ax2.set_title('일별 수익률', fontsize=12)
    ax2.set_xlabel('일수')
    ax2.set_ylabel('수익률')
    ax2.grid(True, alpha=0.3)
    
    # ----- 3. 누적 수익률 -----
    ax3 = axes[1, 0]
    
    cumulative = (1 + daily_returns).cumprod() - 1
    ax3.fill_between(range(len(cumulative)), cumulative, alpha=0.3)
    ax3.plot(range(len(cumulative)), cumulative, linewidth=2)
    ax3.axhline(y=0, color='black', linewidth=0.5)
    ax3.set_title('누적 수익률', fontsize=12)
    ax3.set_xlabel('일수')
    ax3.set_ylabel('누적 수익률')
    ax3.grid(True, alpha=0.3)
    
    # ----- 4. Drawdown -----
    ax4 = axes[1, 1]
    
    peak = portfolio_df['value'].cummax()
    drawdown = (portfolio_df['value'] - peak) / peak
    ax4.fill_between(portfolio_df['date'], drawdown, 0, color='red', alpha=0.3)
    ax4.plot(portfolio_df['date'], drawdown, color='red', linewidth=1)
    ax4.set_title('Drawdown (낙폭)', fontsize=12)
    ax4.set_xlabel('날짜')
    ax4.set_ylabel('낙폭')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


# ============================================
# 테스트
# ============================================

if __name__ == "__main__":
    print("\n[테스트] 백테스트")
    print("Colab에서 data.py와 함께 실행하세요.")
