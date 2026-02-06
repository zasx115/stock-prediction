# ============================================
# 파일명: src/backtest.py
# 설명: 백테스트 (상관관계 + 중장기 모멘텀 4가지 버전)
# 
# 버전 A: SPY 상관관계 > 0.5
# 버전 B: 중장기 모멘텀 (1개월, 3개월, 6개월)
# 버전 C: 상관관계 + 중장기 모멘텀
# 버전 D: C + 섹터필터 + RSI + 섹터당 1종목
# ============================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf


# ============================================
# 설정
# ============================================

INITIAL_CAPITAL = 2000
BUY_COMMISSION = 0.0025
SELL_COMMISSION = 0.0025
SLIPPAGE = 0.001
STOP_LOSS = -0.07

# 단기 모멘텀 (기존)
WEIGHT_1W = 3.5
WEIGHT_2W = 2.5
WEIGHT_3W = 1.5

# 중장기 모멘텀 (새로운)
WEIGHT_1M = 3.5   # 1개월
WEIGHT_3M = 2.5   # 3개월
WEIGHT_6M = 1.5   # 6개월

TOP_N = 3
ALLOCATIONS = [0.4, 0.3, 0.3]

# 상관관계 설정
CORRELATION_PERIOD = 60      # 상관관계 계산 기간 (60일)
CORRELATION_THRESHOLD = 0.5  # 최소 상관관계

# 섹터 필터 설정
SECTOR_MOMENTUM_PERIOD = 21
SECTOR_RSI_PERIOD = 14
SECTOR_RSI_UPPER = 70

# 섹터 ETF 매핑
SECTOR_ETFS = {
    'Technology': 'XLK',
    'Information Technology': 'XLK',
    'Health Care': 'XLV',
    'Financials': 'XLF',
    'Consumer Discretionary': 'XLY',
    'Consumer Staples': 'XLP',
    'Energy': 'XLE',
    'Industrials': 'XLI',
    'Materials': 'XLB',
    'Utilities': 'XLU',
    'Real Estate': 'XLRE',
    'Communication Services': 'XLC'
}


# ============================================
# 1. 데이터 전처리
# ============================================

def prepare_price_data(df):
    """피벗 테이블로 변환"""
    price_df = df.pivot(index='date', columns='symbol', values='close')
    return price_df


def filter_tuesday(price_df):
    """화요일만 필터링"""
    mask = price_df.index.day_name() == 'Tuesday'
    return price_df[mask]


# ============================================
# 2. SPY 상관관계 계산
# ============================================

def calc_spy_correlation(price_df, period=CORRELATION_PERIOD):
    """
    각 종목과 SPY의 상관관계 계산
    
    Returns:
        DataFrame: 날짜별 종목별 상관관계
    """
    if 'SPY' not in price_df.columns:
        return pd.DataFrame()
    
    # 일별 수익률
    returns = price_df.pct_change()
    spy_returns = returns['SPY']
    
    # 롤링 상관관계
    correlation_df = pd.DataFrame(index=price_df.index)
    
    for col in returns.columns:
        if col == 'SPY':
            continue
        correlation_df[col] = returns[col].rolling(period).corr(spy_returns)
    
    return correlation_df


def get_high_correlation_stocks(date, correlation_df, threshold=CORRELATION_THRESHOLD):
    """
    SPY와 상관관계 높은 종목 리스트 반환
    """
    if date not in correlation_df.index:
        return []
    
    corr_values = correlation_df.loc[date].dropna()
    high_corr = corr_values[corr_values > threshold]
    
    return high_corr.index.tolist()


# ============================================
# 3. 모멘텀 점수 계산
# ============================================

def calc_momentum_short(weekly_df):
    """
    단기 모멘텀 (기존)
    (1주×3.5) + (2주×2.5) + (3주×1.5)
    """
    ret_1w = weekly_df.pct_change(1)
    ret_2w = weekly_df.pct_change(2)
    ret_3w = weekly_df.pct_change(3)
    
    score_df = (ret_1w * WEIGHT_1W) + (ret_2w * WEIGHT_2W) + (ret_3w * WEIGHT_3W)
    
    return score_df, ret_1w


def calc_momentum_long(weekly_df):
    """
    중장기 모멘텀 (새로운)
    (1개월×3.5) + (3개월×2.5) + (6개월×1.5)
    
    주 1회 데이터 기준:
    - 4회 전 = 1개월
    - 12회 전 = 3개월
    - 24회 전 = 6개월
    """
    ret_1m = weekly_df.pct_change(4)    # 1개월
    ret_3m = weekly_df.pct_change(12)   # 3개월
    ret_6m = weekly_df.pct_change(24)   # 6개월
    
    score_df = (ret_1m * WEIGHT_1M) + (ret_3m * WEIGHT_3M) + (ret_6m * WEIGHT_6M)
    
    return score_df, ret_1m


# ============================================
# 4. 섹터 ETF 데이터
# ============================================

def get_sector_etf_data(start_date, end_date):
    """섹터 ETF 데이터 다운로드"""
    etfs = list(set(SECTOR_ETFS.values())) + ['SPY']
    
    print(f"섹터 ETF 데이터 다운로드 중... ({len(etfs)}개)")
    
    data = yf.download(
        etfs,
        start=start_date,
        end=end_date,
        auto_adjust=True,
        threads=True
    )
    
    if 'Close' in data.columns.get_level_values(0):
        price_df = data['Close']
    else:
        price_df = data
    
    return price_df


def calc_sector_performance(sector_df, period=SECTOR_MOMENTUM_PERIOD):
    """섹터 SPY 대비 수익률"""
    returns = sector_df.pct_change(period)
    
    if 'SPY' not in returns.columns:
        return pd.DataFrame()
    
    spy_return = returns['SPY']
    excess_returns = returns.sub(spy_return, axis=0)
    excess_returns = excess_returns.drop(columns=['SPY'], errors='ignore')
    
    return excess_returns


def calc_sector_rsi(sector_df, period=SECTOR_RSI_PERIOD):
    """섹터 RSI 계산"""
    rsi_df = pd.DataFrame(index=sector_df.index)
    
    for col in sector_df.columns:
        if col == 'SPY':
            continue
        
        delta = sector_df[col].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        rsi_df[col] = rsi
    
    return rsi_df


def get_valid_sectors(date, excess_returns, sector_rsi):
    """SPY 대비 좋고 RSI < 70인 섹터"""
    if date not in excess_returns.index:
        return []
    
    sector_perf = excess_returns.loc[date].dropna()
    good_sectors = sector_perf[sector_perf > 0].index.tolist()
    
    if date not in sector_rsi.index:
        return good_sectors
    
    rsi_values = sector_rsi.loc[date]
    
    valid_sectors = []
    for sector in good_sectors:
        if sector in rsi_values.index:
            if rsi_values[sector] < SECTOR_RSI_UPPER:
                valid_sectors.append(sector)
        else:
            valid_sectors.append(sector)
    
    return valid_sectors


def get_etf_to_sector():
    """ETF → 섹터 매핑"""
    return {v: k for k, v in SECTOR_ETFS.items()}


# ============================================
# 5. 매수일 매핑
# ============================================

def create_trade_mapping(df):
    """화요일 → 수요일 매핑"""
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
# 6. 백테스트 핵심 로직
# ============================================

def run_backtest_core(df, version='A', sector_df=None, sector_map=None):
    """
    백테스트 핵심 로직
    
    version:
    - 'A': SPY 상관관계 > 0.5
    - 'B': 중장기 모멘텀
    - 'C': 상관관계 + 중장기 모멘텀
    - 'D': C + 섹터필터
    """
    
    df_daily = df.copy().sort_values('date').reset_index(drop=True)
    daily_dates = sorted(df_daily['date'].unique())
    
    price_df = prepare_price_data(df)
    tuesday_df = filter_tuesday(price_df)
    
    if 'SPY' in tuesday_df.columns:
        tuesday_df = tuesday_df.dropna(subset=['SPY'])
    
    # 버전별 모멘텀 계산
    if version in ['B', 'C', 'D']:
        score_df, ret_1m = calc_momentum_long(tuesday_df)
    else:
        score_df, ret_1m = calc_momentum_short(tuesday_df)
    
    # 상관관계 계산 (버전 A, C, D)
    if version in ['A', 'C', 'D']:
        correlation_df = calc_spy_correlation(price_df)
    else:
        correlation_df = pd.DataFrame()
    
    # 섹터 필터 (버전 D)
    if version == 'D' and sector_df is not None:
        excess_returns = calc_sector_performance(sector_df)
        sector_rsi = calc_sector_rsi(sector_df)
        etf_to_sector = get_etf_to_sector()
    else:
        excess_returns = pd.DataFrame()
        sector_rsi = pd.DataFrame()
        etf_to_sector = {}
    
    trade_map = create_trade_mapping(df)
    score_dates = score_df.dropna(how='all').index.tolist()
    
    portfolio_values = []
    trades = []
    
    cash = INITIAL_CAPITAL
    holdings = {}
    pending_order = None
    
    for i, date in enumerate(daily_dates):
        today_data = df_daily[df_daily['date'] == date]
        date_ts = pd.Timestamp(date)
        
        # 포트폴리오 가치
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
        
        # 손절 체크
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
        
        # 매수 주문 실행
        if pending_order is not None and pending_order['trade_date'] == date:
            order = pending_order
            pending_order = None
            
            new_picks = order['picks']
            new_scores = order['scores']
            
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
            
            # 비중 계산
            n_picks = len(new_picks)
            if n_picks >= 3:
                allocations = ALLOCATIONS[:3]
            elif n_picks == 2:
                allocations = [0.5, 0.5]
            elif n_picks == 1:
                allocations = [1.0]
            else:
                allocations = []
            
            target_allocations = {}
            for j, symbol in enumerate(new_picks):
                if j < len(allocations):
                    target_allocations[symbol] = allocations[j]
            
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
                
                if abs(diff_value) / portfolio_value > 0.05 and diff_shares > 0:
                    if diff_value > 0:
                        buy_price = current_price * (1 + SLIPPAGE)
                        buy_amount = diff_shares * buy_price
                        commission = buy_amount * BUY_COMMISSION
                        
                        if cash >= buy_amount + commission:
                            cash -= (buy_amount + commission)
                            holdings[symbol]['shares'] += diff_shares
                            total_cost = holdings[symbol]['avg_price'] * (holdings[symbol]['shares'] - diff_shares) + buy_amount
                            holdings[symbol]['avg_price'] = total_cost / holdings[symbol]['shares']
                            
                            trades.append({
                                'date': date,
                                'symbol': symbol,
                                'action': 'ADD',
                                'shares': diff_shares,
                                'price': buy_price,
                                'amount': buy_amount,
                                'commission': commission,
                                'slippage': current_price * SLIPPAGE * diff_shares,
                                'return_rate': 0
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
                            'return_rate': 0
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
                    score = new_scores[score_idx] if score_idx >= 0 and score_idx < len(new_scores) else 0
                    
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
        
        # 화요일: 종목 선정
        if date_ts not in score_dates:
            continue
        
        if date not in trade_map:
            continue
        
        trade_date = trade_map[date]
        
        # 시장 필터
        if date_ts not in ret_1m.index:
            continue
        
        market_momentum = ret_1m.loc[date_ts].mean()
        if market_momentum <= 0:
            continue
        
        # 점수
        if date_ts not in score_df.index:
            continue
        
        current_scores = score_df.loc[date_ts].drop(labels=['SPY'], errors='ignore').dropna()
        if current_scores.empty:
            continue
        
        # ----- 필터 적용 -----
        filtered_scores = current_scores.copy()
        
        # 상관관계 필터 (버전 A, C, D)
        if version in ['A', 'C', 'D'] and not correlation_df.empty:
            high_corr_stocks = get_high_correlation_stocks(date_ts, correlation_df)
            if high_corr_stocks:
                filtered_scores = filtered_scores[filtered_scores.index.isin(high_corr_stocks)]
        
        # 섹터 필터 (버전 D)
        if version == 'D' and sector_map is not None:
            valid_etfs = get_valid_sectors(date_ts, excess_returns, sector_rsi)
            valid_sectors = [etf_to_sector.get(etf, etf) for etf in valid_etfs]
            
            # 섹터당 1종목
            sector_picked = set()
            final_scores = pd.Series(dtype=float)
            
            for symbol in filtered_scores.sort_values(ascending=False).index:
                if symbol not in sector_map:
                    continue
                
                stock_sector = sector_map[symbol]
                
                if stock_sector not in valid_sectors:
                    continue
                
                if stock_sector in sector_picked:
                    continue
                
                final_scores[symbol] = filtered_scores[symbol]
                sector_picked.add(stock_sector)
                
                if len(final_scores) >= TOP_N:
                    break
            
            filtered_scores = final_scores
        
        if filtered_scores.empty:
            continue
        
        top_n = filtered_scores.nlargest(min(TOP_N, len(filtered_scores)))
        
        pending_order = {
            'score_date': date,
            'trade_date': trade_date,
            'picks': top_n.index.tolist(),
            'scores': top_n.values.tolist()
        }
    
    portfolio_df = pd.DataFrame(portfolio_values)
    trades_df = pd.DataFrame(trades) if trades else pd.DataFrame()
    
    return portfolio_df, trades_df


# ============================================
# 7. 버전별 백테스트
# ============================================

def run_backtest_A(df):
    """버전 A: SPY 상관관계 > 0.5"""
    print("[버전 A] SPY 상관관계 > 0.5 (단기 모멘텀)")
    portfolio_df, trades_df = run_backtest_core(df, version='A')
    metrics = calculate_metrics(portfolio_df, trades_df, df)
    return {'portfolio': portfolio_df, 'trades': trades_df, 'metrics': metrics}


def run_backtest_B(df):
    """버전 B: 중장기 모멘텀"""
    print("[버전 B] 중장기 모멘텀 (1개월, 3개월, 6개월)")
    portfolio_df, trades_df = run_backtest_core(df, version='B')
    metrics = calculate_metrics(portfolio_df, trades_df, df)
    return {'portfolio': portfolio_df, 'trades': trades_df, 'metrics': metrics}


def run_backtest_C(df):
    """버전 C: 상관관계 + 중장기 모멘텀"""
    print("[버전 C] 상관관계 + 중장기 모멘텀")
    portfolio_df, trades_df = run_backtest_core(df, version='C')
    metrics = calculate_metrics(portfolio_df, trades_df, df)
    return {'portfolio': portfolio_df, 'trades': trades_df, 'metrics': metrics}


def run_backtest_D(df, sector_df, sector_map):
    """버전 D: C + 섹터필터"""
    print("[버전 D] 상관관계 + 중장기 모멘텀 + 섹터필터")
    portfolio_df, trades_df = run_backtest_core(df, version='D', sector_df=sector_df, sector_map=sector_map)
    metrics = calculate_metrics(portfolio_df, trades_df, df)
    return {'portfolio': portfolio_df, 'trades': trades_df, 'metrics': metrics}


# ============================================
# 8. 전체 비교 실행
# ============================================

def run_all_versions(df):
    """4가지 버전 비교"""
    print("\n" + "=" * 80)
    print("🧪 상관관계 + 중장기 모멘텀 백테스트 비교")
    print(f"   상관관계 기간: {CORRELATION_PERIOD}일")
    print(f"   상관관계 기준: > {CORRELATION_THRESHOLD}")
    print(f"   중장기 모멘텀: 1개월, 3개월, 6개월")
    print("=" * 80 + "\n")
    
    # 섹터 정보 준비
    from src.data import get_sp500_list
    sp500 = get_sp500_list()
    sector_map = dict(zip(sp500['symbol'], sp500['sector']))
    
    # 섹터 ETF 데이터
    start_date = df['date'].min()
    end_date = df['date'].max()
    sector_df = get_sector_etf_data(start_date, end_date)
    
    results = {}
    
    results['A'] = run_backtest_A(df)
    results['B'] = run_backtest_B(df)
    results['C'] = run_backtest_C(df)
    results['D'] = run_backtest_D(df, sector_df, sector_map)
    
    # 비교 테이블
    print("\n" + "=" * 90)
    print("📊 결과 비교")
    print("=" * 90)
    
    print(f"\n{'버전':<6} {'설명':<40} {'총수익률':>12} {'CAGR':>10} {'MDD':>10} {'샤프':>8}")
    print("-" * 90)
    
    descriptions = {
        'A': 'SPY 상관관계 > 0.5 (단기 모멘텀)',
        'B': '중장기 모멘텀 (1개월, 3개월, 6개월)',
        'C': '상관관계 + 중장기 모멘텀',
        'D': 'C + 섹터필터 + RSI + 섹터당 1종목'
    }
    
    for ver in ['A', 'B', 'C', 'D']:
        m = results[ver]['metrics']
        desc = descriptions[ver]
        print(f"{ver:<6} {desc:<40} {m['total_return']*100:>11.2f}% {m['cagr']*100:>9.2f}% {m['mdd']*100:>9.2f}% {m['sharpe_ratio']:>8.2f}")
    
    print("-" * 90)
    
    spy_ret = results['A']['metrics']['spy_return']
    print(f"{'SPY':<6} {'벤치마크':<40} {spy_ret*100:>11.2f}%")
    
    print("=" * 90)
    
    # 거래 통계
    print("\n💸 거래 통계")
    print("-" * 90)
    print(f"{'버전':<6} {'거래횟수':>10} {'수수료':>15} {'슬리피지':>15} {'손절횟수':>10}")
    print("-" * 90)
    
    for ver in ['A', 'B', 'C', 'D']:
        m = results[ver]['metrics']
        print(f"{ver:<6} {m['total_trades']:>10} ${m['total_commission']:>14.2f} ${m['total_slippage']:>14.2f} {m['stop_loss_count']:>10}")
    
    print("=" * 90)
    
    return results


# ============================================
# 9. 성과 지표 계산
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
            spy_initial = spy.iloc[0]['close']
            spy_final = spy.iloc[-1]['close']
            spy_return = (spy_final - spy_initial) / spy_initial
    
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
# 10. 결과 출력
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
    print(f"  총 비용: ${metrics['total_commission'] + metrics['total_slippage']:,.2f}")
    
    print("\n" + "=" * 60)


# ============================================
# 11. 그래프
# ============================================

def plot_comparison(results, df):
    """4가지 버전 비교 그래프"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    colors = {'A': 'blue', 'B': 'green', 'C': 'red', 'D': 'purple'}
    
    # 1. 포트폴리오 가치
    ax1 = axes[0, 0]
    
    for ver, res in results.items():
        portfolio = res['portfolio'].copy()
        portfolio['normalized'] = portfolio['value'] / portfolio['value'].iloc[0] * 100
        ax1.plot(portfolio['date'], portfolio['normalized'], 
                 label=f'{ver}', linewidth=2, color=colors[ver])
    
    if 'SPY' in df['symbol'].unique():
        spy = df[df['symbol'] == 'SPY'].sort_values('date').copy()
        spy['normalized'] = spy['close'] / spy['close'].iloc[0] * 100
        ax1.plot(spy['date'], spy['normalized'], 
                 label='SPY', linewidth=2, linestyle='--', color='orange')
    
    ax1.set_title('Portfolio Value (Start=100)', fontsize=12)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # 2. 수익률 비교
    ax2 = axes[0, 1]
    versions = list(results.keys())
    x = np.arange(len(versions))
    width = 0.35
    
    returns = [results[v]['metrics']['total_return'] * 100 for v in versions]
    cagrs = [results[v]['metrics']['cagr'] * 100 for v in versions]
    
    ax2.bar(x - width/2, returns, width, label='Total Return', color='steelblue')
    ax2.bar(x + width/2, cagrs, width, label='CAGR', color='lightsteelblue')
    ax2.axhline(y=results['A']['metrics']['spy_return']*100, color='orange', linestyle='--', label='SPY')
    ax2.set_xticks(x)
    ax2.set_xticklabels(versions)
    ax2.set_title('Total Return & CAGR (%)', fontsize=12)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. MDD
    ax3 = axes[1, 0]
    mdds = [results[v]['metrics']['mdd'] * 100 for v in versions]
    ax3.bar(versions, mdds, color=[colors[v] for v in versions])
    ax3.set_title('Maximum Drawdown (%)', fontsize=12)
    ax3.grid(True, alpha=0.3)
    
    # 4. 샤프
    ax4 = axes[1, 1]
    sharpes = [results[v]['metrics']['sharpe_ratio'] for v in versions]
    ax4.bar(versions, sharpes, color=[colors[v] for v in versions])
    ax4.set_title('Sharpe Ratio', fontsize=12)
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print("\n📋 버전 설명:")
    print("  A: SPY 상관관계 > 0.5 (단기 모멘텀 유지)")
    print("  B: 중장기 모멘텀 (1개월, 3개월, 6개월)")
    print("  C: 상관관계 + 중장기 모멘텀")
    print("  D: C + 섹터필터 + RSI + 섹터당 1종목")
