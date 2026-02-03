# ============================================
# 파일명: src/backtest.py
# 설명: 백테스트 (현실적인 매매 타이밍)
# 
# 전략:
# - 월요일 종가로 점수 계산 → 화요일 종가로 매수
# - 목요일 종가로 점수 계산 → 금요일 종가로 매수
# - 시장 필터: 1주 수익률 평균 > 0
# - 손절은 매일 체크 (-7%)
# ============================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# ============================================
# 설정
# ============================================

INITIAL_CAPITAL = 2000       # 초기 자본금 ($2000)

BUY_COMMISSION = 0.0025      # 매수 수수료 (0.25%)
SELL_COMMISSION = 0.0025     # 매도 수수료 (0.25%)

STOP_LOSS = -0.07            # 손절 기준 (-7%)

WEIGHT_1W = 3.5              # 1주 수익률 가중치
WEIGHT_2W = 2.5              # 2주 수익률 가중치
WEIGHT_3W = 1.5              # 3주 수익률 가중치

TOP_N = 3                    # 상위 종목 수
ALLOCATIONS = [0.4, 0.3, 0.3]  # 투자 비중


# ============================================
# 1. 데이터 전처리 (월/목 필터링)
# ============================================

def prepare_biweekly_data(df):
    """
    월요일, 목요일 데이터만 필터링합니다.
    (resample 대신 정확한 날짜 필터링)
    """
    print("월/목 데이터 필터링 중...")
    
    df = df.copy()
    
    # 피벗: 날짜 × 종목 형태로 변환
    price_df = df.pivot(index='date', columns='symbol', values='close')
    
    # 요일 추가
    price_df['weekday'] = price_df.index.day_name()
    
    # 월요일, 목요일만 필터링
    biweekly_prices = price_df[price_df['weekday'].isin(['Monday', 'Thursday'])].copy()
    biweekly_prices = biweekly_prices.drop(columns=['weekday'])
    
    # SPY 있는 날만
    if 'SPY' in biweekly_prices.columns:
        biweekly_prices = biweekly_prices.dropna(subset=['SPY'])
    
    print(f"✅ {len(biweekly_prices)}개 날짜 필터링 완료!")
    
    return biweekly_prices


# ============================================
# 2. 모멘텀 점수 계산
# ============================================

def calc_momentum_scores(biweekly_prices):
    """
    모멘텀 점수 계산
    
    ret_1w = 2회 전 대비 (약 1주)
    ret_2w = 4회 전 대비 (약 2주)
    ret_3w = 6회 전 대비 (약 3주)
    
    score = (ret_1w × 3.5) + (ret_2w × 2.5) + (ret_3w × 1.5)
    """
    print("모멘텀 점수 계산 중...")
    
    ret_1w = biweekly_prices.pct_change(2)
    ret_2w = biweekly_prices.pct_change(4)
    ret_3w = biweekly_prices.pct_change(6)
    
    score_df = (ret_1w * WEIGHT_1W) + (ret_2w * WEIGHT_2W) + (ret_3w * WEIGHT_3W)
    
    print(f"✅ 점수 계산 완료!")
    
    return score_df, ret_1w


# ============================================
# 3. 매수일 매핑 생성
# ============================================

def create_trade_mapping(df):
    """
    점수 계산일 → 실제 매수일 매핑
    
    월요일 종가로 점수 → 화요일 종가로 매수
    목요일 종가로 점수 → 금요일 종가로 매수
    """
    print("매수일 매핑 생성 중...")
    
    df = df.copy()
    dates = sorted(df['date'].unique())
    
    # 날짜별 요일
    date_weekday = {d: pd.Timestamp(d).day_name() for d in dates}
    
    # 매핑: 점수계산일 → 매수일
    trade_map = {}
    
    for i, date in enumerate(dates):
        weekday = date_weekday[date]
        
        # 월요일 → 다음 화요일 찾기
        if weekday == 'Monday':
            for j in range(i+1, len(dates)):
                if date_weekday[dates[j]] == 'Tuesday':
                    trade_map[date] = dates[j]
                    break
        
        # 목요일 → 다음 금요일 찾기
        elif weekday == 'Thursday':
            for j in range(i+1, len(dates)):
                if date_weekday[dates[j]] == 'Friday':
                    trade_map[date] = dates[j]
                    break
    
    print(f"✅ {len(trade_map)}개 매핑 생성 완료!")
    
    return trade_map


# ============================================
# 4. 백테스트 실행 (메인)
# ============================================

def run_backtest(df):
    """
    백테스트 실행
    
    - 월요일 종가로 점수 계산 → 화요일 종가로 매수
    - 목요일 종가로 점수 계산 → 금요일 종가로 매수
    - 손절은 매일 체크
    """
    print("=" * 50)
    print("[백테스트 실행]")
    print(f"초기 자본금: ${INITIAL_CAPITAL:,}")
    print(f"손절 기준: {STOP_LOSS*100:.1f}%")
    print("점수: 월요일/목요일 종가")
    print("매수: 화요일/금요일 종가")
    print("=" * 50)
    
    # 원본 데이터 보관
    df_daily = df.copy()
    df_daily = df_daily.sort_values('date').reset_index(drop=True)
    daily_dates = sorted(df_daily['date'].unique())
    
    # 월/목 데이터 준비 (점수 계산용)
    biweekly_prices = prepare_biweekly_data(df)
    score_df, ret_1w = calc_momentum_scores(biweekly_prices)
    
    # 점수계산일 → 매수일 매핑
    trade_map = create_trade_mapping(df)
    
    # 점수 계산 날짜 (월/목)
    score_dates = biweekly_prices.index.tolist()
    
    # 결과 저장
    portfolio_values = []
    trades = []
    
    # 현재 상태
    cash = INITIAL_CAPITAL
    holdings = {}
    
    # 대기 중인 매수 주문 (점수계산 후 다음날 매수)
    pending_order = None
    
    print(f"\n{len(daily_dates)}일 시뮬레이션 시작...")
    
    # ----- 매일 시뮬레이션 -----
    for i, date in enumerate(daily_dates):
        
        if (i + 1) % 50 == 0:
            print(f"  진행중... {i+1}/{len(daily_dates)} ({(i+1)/len(daily_dates)*100:.1f}%)")
        
        today_data = df_daily[df_daily['date'] == date]
        date_ts = pd.Timestamp(date)
        
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
        
        # ----- 대기 중인 매수 주문 실행 (화요일/금요일) -----
        if pending_order is not None and pending_order['trade_date'] == date:
            order = pending_order
            pending_order = None
            
            # 기존 보유 종목 매도
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
            
            # 새 종목 매수
            picks = order['picks']
            scores = order['scores']
            n_picks = len(picks)
            
            if n_picks >= 3:
                allocations = ALLOCATIONS[:3]
            elif n_picks == 2:
                allocations = [0.5, 0.5]
            elif n_picks == 1:
                allocations = [1.0]
            else:
                allocations = []
            
            for j, (symbol, allocation) in enumerate(zip(picks, allocations)):
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
                        'return_rate': 0,
                        'score': scores[j] if j < len(scores) else 0
                    })
        
        # ----- 점수 계산일인지 확인 (월요일/목요일) -----
        if date_ts not in score_dates:
            continue
        
        # 매수일 확인
        if date not in trade_map:
            continue
        
        trade_date = trade_map[date]
        
        # ----- 시장 필터 (ret_1w 평균 > 0) -----
        market_momentum = ret_1w.loc[date_ts].mean()
        
        if market_momentum <= 0:
            continue
        
        # ----- 상위 종목 선정 -----
        current_scores = score_df.loc[date_ts].drop(labels=['SPY'], errors='ignore').dropna()
        
        if current_scores.empty:
            continue
        
        top_n = current_scores.nlargest(TOP_N)
        
        # ----- 매수 주문 대기 -----
        pending_order = {
            'score_date': date,
            'trade_date': trade_date,
            'picks': top_n.index.tolist(),
            'scores': top_n.values.tolist()
        }
    
    # ----- 결과 정리 -----
    portfolio_df = pd.DataFrame(portfolio_values)
    trades_df = pd.DataFrame(trades) if trades else pd.DataFrame()
    metrics = calculate_metrics(portfolio_df, trades_df, df_daily)
    
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

def print_metrics(metrics, trades_df=None):
    """
    성과 지표를 보기 좋게 출력합니다.
    """
    print("\n" + "=" * 50)
    print("📊 백테스트 성과")
    print("=" * 50)
    
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
    print(f"  총 수수료: ${metrics['total_commission']:,.2f}")
    print(f"  손절 횟수: {metrics['stop_loss_count']}회")
    
    print(f"\n📅 기타")
    print(f"  승률 (일 기준): {metrics['win_rate']*100:.2f}%")
    
    # 최근 매수 10회 표시
    if trades_df is not None and not trades_df.empty:
        buy_trades = trades_df[trades_df['action'] == 'BUY'].copy()
        
        if not buy_trades.empty:
            recent_dates = buy_trades['date'].drop_duplicates().sort_values(ascending=False).head(10)
            
            print(f"\n🛒 최근 매수 내역 (최근 10회)")
            print("-" * 50)
            
            for buy_date in recent_dates:
                date_buys = buy_trades[buy_trades['date'] == buy_date].sort_values('score', ascending=False)
                print(f"\n📅 {buy_date.strftime('%Y-%m-%d')}")
                
                for i, (_, row) in enumerate(date_buys.iterrows()):
                    score = row.get('score', 0)
                    print(f"  {i+1}위: {row['symbol']:5} | 점수: {score:.4f} | 가격: ${row['price']:.2f} | 금액: ${row['amount']:,.2f}")
    
    print("\n" + "=" * 50)


# ============================================
# 7. 그래프 출력
# ============================================

def plot_results(portfolio_df, trades_df, df, figsize=(14, 12)):
    """
    백테스트 결과를 그래프로 출력합니다.
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    # 1. 포트폴리오 vs SPY
    ax1 = axes[0, 0]
    
    portfolio_df = portfolio_df.copy()
    portfolio_df['normalized'] = portfolio_df['value'] / portfolio_df['value'].iloc[0] * 100
    
    if not trades_df.empty:
        buy_dates = trades_df[trades_df['action'] == 'BUY']['date'].unique()
        sell_dates = trades_df[trades_df['action'].isin(['SELL', 'STOP_LOSS'])]['date'].unique()
        
        hold_start = None
        
        for i, row in portfolio_df.iterrows():
            date = row['date']
            
            if date in buy_dates:
                if hold_start is not None:
                    ax1.axvspan(hold_start, date, alpha=0.2, color='gray', label='_nolegend_')
                hold_start = None
            
            if date in sell_dates and date not in buy_dates:
                hold_start = date
        
        if hold_start is not None:
            ax1.axvspan(hold_start, portfolio_df['date'].iloc[-1], alpha=0.2, color='gray', label='_nolegend_')
    
    ax1.plot(portfolio_df['date'], portfolio_df['normalized'], 
             label='Portfolio', linewidth=2, color='blue')
    
    if 'SPY' in df['symbol'].unique():
        spy = df[df['symbol'] == 'SPY'].sort_values('date').copy()
        spy['normalized'] = spy['close'] / spy['close'].iloc[0] * 100
        ax1.plot(spy['date'], spy['normalized'], 
                 label='SPY', linewidth=2, alpha=0.7, color='orange')
    
    if not trades_df.empty:
        buy_trades = trades_df[trades_df['action'] == 'BUY']
        for _, trade in buy_trades.iterrows():
            trade_date = trade['date']
            port_value = portfolio_df[portfolio_df['date'] == trade_date]['normalized']
            if not port_value.empty:
                ax1.scatter(trade_date, port_value.values[0], 
                           color='red', s=30, zorder=5, label='_nolegend_')
    
    ax1.set_title('Portfolio vs SPY (빨간점=매수, 회색=홀딩)', fontsize=12)
    ax1.set_xlabel('날짜')
    ax1.set_ylabel('가치 (시작=100)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. 일별 수익률
    ax2 = axes[0, 1]
    daily_returns = portfolio_df['value'].pct_change().dropna()
    colors = ['green' if r > 0 else 'red' for r in daily_returns]
    ax2.bar(range(len(daily_returns)), daily_returns, color=colors, alpha=0.7)
    ax2.axhline(y=0, color='black', linewidth=0.5)
    ax2.set_title('일별 수익률', fontsize=12)
    ax2.grid(True, alpha=0.3)
    
    # 3. 누적 수익률
    ax3 = axes[1, 0]
    cumulative = (1 + daily_returns).cumprod() - 1
    ax3.fill_between(range(len(cumulative)), cumulative, alpha=0.3, color='blue')
    ax3.plot(range(len(cumulative)), cumulative, linewidth=2, color='blue')
    ax3.axhline(y=0, color='black', linewidth=0.5)
    ax3.set_title('누적 수익률', fontsize=12)
    ax3.grid(True, alpha=0.3)
    
    # 4. Drawdown
    ax4 = axes[1, 1]
    peak = portfolio_df['value'].cummax()
    drawdown = (portfolio_df['value'] - peak) / peak
    ax4.fill_between(portfolio_df['date'], drawdown, 0, color='red', alpha=0.3)
    ax4.plot(portfolio_df['date'], drawdown, color='red', linewidth=1)
    ax4.set_title('Drawdown (낙폭)', fontsize=12)
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print("\n📊 그래프 범례:")
    print("  🔴 빨간 점: 매수 시점")
    print("  ⬜ 회색 구간: 홀딩")
    print("  🔵 파란 라인: 포트폴리오")
    print("  🟠 주황 라인: SPY")
