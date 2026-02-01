# ============================================
# 파일명: src/backtest.py
# 설명: 백테스트 (최적화 버전)
# 
# 전략:
# - 모멘텀 점수 기반 상위 3종목 선정
# - 시장 필터링 (평균 수익률 > 0)
# - RSI 필터링 (80 이상 과매수 제외)
# - 화요일/목요일만 매수
# - 손절은 매일 체크 (-5%)
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
STOP_LOSS = -0.07            # 손절 기준 (-5%)

# ----- 모멘텀 점수 가중치 -----
WEIGHT_2DAY = 3.5            # 2일전 수익률 가중치
WEIGHT_4DAY = 2.5            # 4일전 수익률 가중치
WEIGHT_6DAY = 1.5            # 6일전 수익률 가중치

# ----- 종목 선정 -----
TOP_N = 3                    # 상위 몇 개 종목 선정
ALLOCATIONS = [0.4, 0.3, 0.3]  # 투자 비중 (1위, 2위, 3위)

# ----- 필터링 조건 -----
MIN_SCORE = 0.004             # 최소 점수 (이 점수 이상이어야 매수)
MARKET_FILTER = True         # 시장 필터 사용 여부

# ----- 리밸런싱 조건 -----
REBALANCE_DAYS = ['Tuesday', 'Thursday']  # 화요일, 목요일만 매수

# ----- RSI 필터 -----
RSI_PERIOD = 14              # RSI 계산 기간
RSI_OVERBOUGHT = 90          # RSI 80 이상이면 과매수 → 매수 제외


# ============================================
# 1. 모멘텀 점수 사전 계산
# ============================================

def calc_all_momentum_scores(df):
    """
    모든 날짜의 모멘텀 점수를 한 번에 계산합니다.
    
    공식:
    score = (2일전 수익률 × 3.5) + (4일전 수익률 × 2.5) + (6일전 수익률 × 1.5)
    
    Args:
        df: 전체 주가 데이터
    
    Returns:
        DataFrame: 날짜, 종목, 종가, 점수
    """
    print("모멘텀 점수 사전 계산 중...")
    
    df = df.copy()
    df = df.sort_values(['symbol', 'date']).reset_index(drop=True)
    
    results = []
    
    for symbol in df['symbol'].unique():
        stock = df[df['symbol'] == symbol].copy().reset_index(drop=True)
        
        # 최소 7일 데이터 필요
        if len(stock) < 7:
            continue
        
        # 7일차부터 점수 계산
        for i in range(6, len(stock)):
            today = stock.iloc[i]
            today_close = today['close']
            today_date = today['date']
            
            close_2d = stock.iloc[i-2]['close']
            close_4d = stock.iloc[i-4]['close']
            close_6d = stock.iloc[i-6]['close']
            
            return_2d = (today_close - close_2d) / close_2d
            return_4d = (today_close - close_4d) / close_4d
            return_6d = (today_close - close_6d) / close_6d
            
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
    
    Args:
        df: 전체 주가 데이터
    
    Returns:
        DataFrame: 날짜, 시장 평균 수익률
    """
    print("시장 수익률 사전 계산 중...")
    
    df = df.copy()
    df = df.sort_values(['symbol', 'date']).reset_index(drop=True)
    
    df['daily_return'] = df.groupby('symbol')['close'].pct_change()
    
    market_returns = df.groupby('date')['daily_return'].mean().reset_index()
    market_returns.columns = ['date', 'market_return']
    
    print(f"✅ {len(market_returns)}일 시장 수익률 계산 완료!")
    
    return market_returns


# ============================================
# 3. RSI 사전 계산
# ============================================

def calc_rsi(df, period=14):
    """
    모든 종목의 RSI를 한 번에 계산합니다.
    
    RSI = 100 - (100 / (1 + RS))
    RS = 평균 상승폭 / 평균 하락폭
    
    80 이상: 과매수 (매수 피하기)
    20 이하: 과매도 (매수 기회)
    
    Args:
        df: 주가 데이터
        period: RSI 기간 (기본 14일)
    
    Returns:
        DataFrame: 날짜, 종목, RSI
    """
    print("RSI 사전 계산 중...")
    
    df = df.copy()
    df = df.sort_values(['symbol', 'date']).reset_index(drop=True)
    
    results = []
    
    for symbol in df['symbol'].unique():
        stock = df[df['symbol'] == symbol].copy().reset_index(drop=True)
        
        if len(stock) < period + 1:
            continue
        
        # 일일 변화량
        delta = stock['close'].diff()
        
        # 상승/하락 분리
        gain = delta.where(delta > 0, 0)
        loss = (-delta).where(delta < 0, 0)
        
        # 평균 계산
        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()
        
        # RSI 계산
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        # 결과 저장
        for i in range(period, len(stock)):
            results.append({
                'date': stock.iloc[i]['date'],
                'symbol': symbol,
                'rsi': rsi.iloc[i]
            })
    
    result_df = pd.DataFrame(results)
    print(f"✅ {len(result_df):,}개 RSI 계산 완료!")
    
    return result_df


# ============================================
# 4. 백테스트 실행 (메인)
# ============================================

def run_backtest(df):
    """
    백테스트를 실행합니다.
    
    매일 체크:
    - 손절 (-5% 이하면 매도)
    
    화요일/목요일만:
    - 시장 필터 (평균 수익률 > 0)
    - 종목 선정 (모멘텀 상위 + RSI < 80)
    - 매수/매도 실행
    
    Args:
        df: 주가 데이터
    
    Returns:
        dict: portfolio(일별 가치), trades(거래 내역), metrics(성과 지표)
    """
    print("=" * 50)
    print("[백테스트 실행]")
    print(f"초기 자본금: {INITIAL_CAPITAL:,}원")
    print(f"매수 요일: {', '.join(REBALANCE_DAYS)}")
    print(f"매수 수수료: {BUY_COMMISSION*100:.2f}%")
    print(f"매도 수수료: {SELL_COMMISSION*100:.2f}%")
    print(f"손절 기준: {STOP_LOSS*100:.1f}%")
    print(f"RSI 과매수 기준: {RSI_OVERBOUGHT}")
    print("=" * 50)
    
    # 데이터 정렬
    df = df.sort_values('date').reset_index(drop=True)
    dates = sorted(df['date'].unique())
    
    # ----- 사전 계산 (한 번만) -----
    all_scores = calc_all_momentum_scores(df)
    market_returns = calc_daily_market_returns(df)
    all_rsi = calc_rsi(df, RSI_PERIOD)
    
    # 빠른 조회용 딕셔너리
    market_dict = dict(zip(market_returns['date'], market_returns['market_return']))
    
    # 결과 저장
    portfolio_values = []
    trades = []
    
    # 현재 상태
    cash = INITIAL_CAPITAL
    holdings = {}
    
    print(f"\n{len(dates)}일 시뮬레이션 시작...")
    
    # ----- 날짜별 시뮬레이션 -----
    for i, date in enumerate(dates):
        
        # 진행 상황 (50일마다)
        if (i + 1) % 50 == 0:
            print(f"  진행중... {i+1}/{len(dates)} ({(i+1)/len(dates)*100:.1f}%)")
        
        today_data = df[df['date'] == date]
        
        # ----- 포트폴리오 가치 계산 (매일) -----
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
        
        # ----- 매수는 화요일/목요일만 -----
        day_name = date.strftime('%A')
        
        if day_name not in REBALANCE_DAYS:
            continue  # 화요일, 목요일 아니면 매수 스킵
        
        # ----- 오늘 점수 조회 -----
        today_scores = all_scores[all_scores['date'] == date].copy()
        
        if today_scores.empty:
            continue
        
        # ----- 시장 필터링 -----
        if MARKET_FILTER:
            market_ret = market_dict.get(date, 0)
            if market_ret <= 0:
                continue  # 시장 안 좋으면 매수 안 함
        
        # ----- 상위 종목 선정 -----
        today_scores = today_scores.sort_values('score', ascending=False)
        qualified = today_scores.head(TOP_N)
        qualified = qualified[qualified['score'] >= MIN_SCORE]
        
        # ----- RSI 필터 (과매수 제외) -----
        today_rsi = all_rsi[all_rsi['date'] == date]
        
        if not today_rsi.empty:
            overbought = today_rsi[today_rsi['rsi'] >= RSI_OVERBOUGHT]['symbol'].tolist()
            qualified = qualified[~qualified['symbol'].isin(overbought)]
        
        # ----- 조건 맞는 종목 없으면 스킵 -----
        if len(qualified) == 0:
            continue
        
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
        picks = qualified['symbol'].tolist()
        n_picks = len(picks)
        
        if n_picks >= 3:
            allocations = ALLOCATIONS[:3]
        elif n_picks == 2:
            allocations = [0.5, 0.5]
        else:
            allocations = [1.0]
        
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
# 5. 성과 지표 계산
# ============================================

def calculate_metrics(portfolio_df, trades_df, df):
    """
    백테스트 성과 지표를 계산합니다.
    """
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
            spy_initial = spy.iloc[0]['close']
            spy_final = spy.iloc[-1]['close']
            spy_return = (spy_final - spy_initial) / spy_initial
    
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
# 6. 결과 출력
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
# 7. 그래프 출력 (Colab용)
# ============================================

def plot_results(portfolio_df, trades_df, df, figsize=(14, 12)):
    """
    백테스트 결과를 그래프로 출력합니다.
    
    표시 내용:
    - 빨간 점: 매매 시점
    - 회색 구간: 홀딩 기간 (매수 종목 없음)
    - 파란 구간: 보유 기간
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    # ----- 1. 포트폴리오 vs SPY + 매매 시점 표시 -----
    ax1 = axes[0, 0]
    
    # 포트폴리오 정규화
    portfolio_df = portfolio_df.copy()
    portfolio_df['normalized'] = portfolio_df['value'] / portfolio_df['value'].iloc[0] * 100
    
    # 홀딩 구간 표시 (매수 종목 없는 기간)
    if not trades_df.empty:
        # 매수 날짜 리스트
        buy_dates = trades_df[trades_df['action'] == 'BUY']['date'].unique()
        sell_dates = trades_df[trades_df['action'].isin(['SELL', 'STOP_LOSS'])]['date'].unique()
        trade_dates = set(buy_dates) | set(sell_dates)
        
        # 보유 중인지 추적
        holding = False
        hold_start = None
        
        for i, row in portfolio_df.iterrows():
            date = row['date']
            
            # 매수하면 보유 시작
            if date in buy_dates:
                holding = True
                if hold_start is not None:
                    # 이전 홀딩 구간 표시 (회색)
                    ax1.axvspan(hold_start, date, alpha=0.2, color='gray', label='_nolegend_')
                hold_start = None
            
            # 전부 매도하면 홀딩 시작
            if date in sell_dates and date not in buy_dates:
                holding = False
                hold_start = date
        
        # 마지막 홀딩 구간
        if hold_start is not None:
            ax1.axvspan(hold_start, portfolio_df['date'].iloc[-1], alpha=0.2, color='gray', label='_nolegend_')
    
    # 포트폴리오 라인
    ax1.plot(portfolio_df['date'], portfolio_df['normalized'], 
             label='Portfolio', linewidth=2, color='blue')
    
    # SPY 라인
    if 'SPY' in df['symbol'].unique():
        spy = df[df['symbol'] == 'SPY'].sort_values('date').copy()
        spy['normalized'] = spy['close'] / spy['close'].iloc[0] * 100
        ax1.plot(spy['date'], spy['normalized'], 
                 label='SPY', linewidth=2, alpha=0.7, color='orange')
    
    # 매매 시점 빨간 점 표시
    if not trades_df.empty:
        buy_trades = trades_df[trades_df['action'] == 'BUY']
        for _, trade in buy_trades.iterrows():
            trade_date = trade['date']
            # 해당 날짜의 포트폴리오 가치
            port_value = portfolio_df[portfolio_df['date'] == trade_date]['normalized']
            if not port_value.empty:
                ax1.scatter(trade_date, port_value.values[0], 
                           color='red', s=30, zorder=5, label='_nolegend_')
    
    ax1.set_title('Portfolio vs SPY (빨간점=매수, 회색=홀딩)', fontsize=12)
    ax1.set_xlabel('날짜')
    ax1.set_ylabel('가치 (시작=100)')
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
    ax3.fill_between(range(len(cumulative)), cumulative, alpha=0.3, color='blue')
    ax3.plot(range(len(cumulative)), cumulative, linewidth=2, color='blue')
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
    
    # ----- 범례 설명 출력 -----
    print("\n📊 그래프 범례:")
    print("  🔴 빨간 점: 매수 시점")
    print("  ⬜ 회색 구간: 홀딩 (보유 종목 없음)")
    print("  🔵 파란 라인: 포트폴리오 가치")
    print("  🟠 주황 라인: SPY (벤치마크)")

