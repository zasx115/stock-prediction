# ============================================
# 파일명: src/backtest.py
# 설명: 백테스트 (섹터 필터 4가지 버전)
# 
# 버전 A: 섹터 필터만 (SPY 대비)
# 버전 B: 섹터 필터 + RSI
# 버전 C: 섹터 필터 + RSI + 섹터당 1종목
# 버전 D: SPY 대비 1위 섹터에서 모멘텀 3종목
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

WEIGHT_1W = 3.5
WEIGHT_2W = 2.5
WEIGHT_3W = 1.5

TOP_N = 3
ALLOCATIONS = [0.4, 0.3, 0.3]

# 섹터 필터 설정
SECTOR_MOMENTUM_PERIOD = 21  # 섹터 모멘텀 기간 (약 1개월)
SECTOR_RSI_PERIOD = 14       # RSI 기간
SECTOR_RSI_UPPER = 70        # RSI 과열 기준

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
# 1. 섹터 ETF 데이터 다운로드
# ============================================

def get_sector_etf_data(start_date, end_date):
    """
    섹터 ETF 데이터 다운로드
    """
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


# ============================================
# 2. 섹터 성과 계산 (SPY 대비)
# ============================================

def calc_sector_performance(sector_df, period=SECTOR_MOMENTUM_PERIOD):
    """
    각 섹터의 SPY 대비 수익률 계산
    
    Returns:
        DataFrame: 날짜별 섹터 초과 수익률
    """
    # 수익률 계산
    returns = sector_df.pct_change(period)
    
    if 'SPY' not in returns.columns:
        return pd.DataFrame()
    
    spy_return = returns['SPY']
    
    # SPY 대비 초과 수익률
    excess_returns = returns.sub(spy_return, axis=0)
    excess_returns = excess_returns.drop(columns=['SPY'], errors='ignore')
    
    return excess_returns


# ============================================
# 3. 섹터 RSI 계산
# ============================================

def calc_sector_rsi(sector_df, period=SECTOR_RSI_PERIOD):
    """
    각 섹터 ETF의 RSI 계산
    """
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


# ============================================
# 4. 투자 가능 섹터 선정
# ============================================

def get_valid_sectors(date, excess_returns, sector_rsi, version='A'):
    """
    투자 가능한 섹터 리스트 반환
    
    version:
    - 'A': SPY 대비 수익률 > 0인 섹터
    - 'B': A + RSI < 70
    - 'C': B와 동일 (종목 선정에서 차이)
    - 'D': SPY 대비 수익률 1위 섹터만
    """
    if date not in excess_returns.index:
        return []
    
    # SPY 대비 수익률
    sector_perf = excess_returns.loc[date].dropna()
    
    if version == 'D':
        # 1위 섹터만
        if sector_perf.empty:
            return []
        best_sector = sector_perf.idxmax()
        return [best_sector]
    
    # SPY보다 좋은 섹터
    good_sectors = sector_perf[sector_perf > 0].index.tolist()
    
    if version == 'A':
        return good_sectors
    
    # RSI 필터 (버전 B, C)
    if version in ['B', 'C']:
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
    
    return good_sectors


# ============================================
# 5. ETF → 섹터 이름 역매핑
# ============================================

def get_etf_to_sector():
    """ETF 심볼 → 섹터 이름 매핑"""
    return {v: k for k, v in SECTOR_ETFS.items()}


# ============================================
# 6. 데이터 전처리
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
# 7. 모멘텀 점수 계산
# ============================================

def calc_momentum_scores(weekly_df):
    """모멘텀 점수 계산"""
    ret_1w = weekly_df.pct_change(1)
    ret_2w = weekly_df.pct_change(2)
    ret_3w = weekly_df.pct_change(3)
    
    score_df = (ret_1w * WEIGHT_1W) + (ret_2w * WEIGHT_2W) + (ret_3w * WEIGHT_3W)
    
    return score_df, ret_1w


# ============================================
# 8. 매수일 매핑
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
# 9. 백테스트 핵심 로직
# ============================================

def run_backtest_core(df, sector_df, sector_map, version='A'):
    """
    백테스트 핵심 로직
    
    version:
    - 'A': 섹터 필터만
    - 'B': 섹터 필터 + RSI
    - 'C': 섹터 필터 + RSI + 섹터당 1종목
    - 'D': 1위 섹터에서 Top 3
    """
    
    df_daily = df.copy().sort_values('date').reset_index(drop=True)
    daily_dates = sorted(df_daily['date'].unique())
    
    price_df = prepare_price_data(df)
    tuesday_df = filter_tuesday(price_df)
    
    if 'SPY' in tuesday_df.columns:
        tuesday_df = tuesday_df.dropna(subset=['SPY'])
    
    score_df, ret_1w = calc_momentum_scores(tuesday_df)
    
    # 섹터 성과 & RSI 계산
    excess_returns = calc_sector_performance(sector_df)
    sector_rsi = calc_sector_rsi(sector_df)
    
    # ETF → 섹터 매핑
    etf_to_sector = get_etf_to_sector()
    
    trade_map = create_trade_mapping(df)
    score_dates = score_df.dropna(how='all').index.tolist()
    
    portfolio_values = []
    trades = []
    
    cash = INITIAL_CAPITAL
    holdings = {}
    pending_order = None
    
    # 통계
    skipped_by_sector = 0
    skipped_by_rsi = 0
    
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
        if date_ts not in ret_1w.index:
            continue
        
        market_momentum = ret_1w.loc[date_ts].mean()
        if market_momentum <= 0:
            continue
        
        # 점수
        if date_ts not in score_df.index:
            continue
        
        current_scores = score_df.loc[date_ts].drop(labels=['SPY'], errors='ignore').dropna()
        if current_scores.empty:
            continue
        
        # ----- 섹터 필터 적용 -----
        valid_etfs = get_valid_sectors(date_ts, excess_returns, sector_rsi, version)
        
        # ETF → 섹터 이름 변환
        valid_sectors = []
        for etf in valid_etfs:
            if etf in etf_to_sector:
                valid_sectors.append(etf_to_sector[etf])
        
        # 종목별 섹터 확인
        filtered_scores = pd.Series(dtype=float)
        
        if version == 'C':
            # 섹터당 1종목
            sector_picked = set()
            
            for symbol in current_scores.sort_values(ascending=False).index:
                if symbol not in sector_map:
                    continue
                
                stock_sector = sector_map[symbol]
                
                if stock_sector not in valid_sectors:
                    continue
                
                if stock_sector in sector_picked:
                    continue
                
                filtered_scores[symbol] = current_scores[symbol]
                sector_picked.add(stock_sector)
                
                if len(filtered_scores) >= TOP_N:
                    break
        
        elif version == 'D':
            # 1위 섹터에서 Top 3
            for symbol in current_scores.sort_values(ascending=False).index:
                if symbol not in sector_map:
                    continue
                
                stock_sector = sector_map[symbol]
                
                if stock_sector not in valid_sectors:
                    continue
                
                filtered_scores[symbol] = current_scores[symbol]
                
                if len(filtered_scores) >= TOP_N:
                    break
        
        else:
            # 버전 A, B: valid_sectors에 속한 종목만
            for symbol in current_scores.sort_values(ascending=False).index:
                if symbol not in sector_map:
                    continue
                
                stock_sector = sector_map[symbol]
                
                if stock_sector not in valid_sectors:
                    continue
                
                filtered_scores[symbol] = current_scores[symbol]
                
                if len(filtered_scores) >= TOP_N:
                    break
        
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
# 10. 버전별 백테스트
# ============================================

def run_backtest_A(df, sector_df, sector_map):
    """버전 A: 섹터 필터만"""
    print("[버전 A] 섹터 필터 (SPY 대비)")
    portfolio_df, trades_df = run_backtest_core(df, sector_df, sector_map, version='A')
    metrics = calculate_metrics(portfolio_df, trades_df, df)
    return {'portfolio': portfolio_df, 'trades': trades_df, 'metrics': metrics}


def run_backtest_B(df, sector_df, sector_map):
    """버전 B: 섹터 필터 + RSI"""
    print("[버전 B] 섹터 필터 + RSI")
    portfolio_df, trades_df = run_backtest_core(df, sector_df, sector_map, version='B')
    metrics = calculate_metrics(portfolio_df, trades_df, df)
    return {'portfolio': portfolio_df, 'trades': trades_df, 'metrics': metrics}


def run_backtest_C(df, sector_df, sector_map):
    """버전 C: 섹터 필터 + RSI + 섹터당 1종목"""
    print("[버전 C] 섹터 필터 + RSI + 섹터당 1종목")
    portfolio_df, trades_df = run_backtest_core(df, sector_df, sector_map, version='C')
    metrics = calculate_metrics(portfolio_df, trades_df, df)
    return {'portfolio': portfolio_df, 'trades': trades_df, 'metrics': metrics}


def run_backtest_D(df, sector_df, sector_map):
    """버전 D: 1위 섹터에서 Top 3"""
    print("[버전 D] 1위 섹터에서 Top 3")
    portfolio_df, trades_df = run_backtest_core(df, sector_df, sector_map, version='D')
    metrics = calculate_metrics(portfolio_df, trades_df, df)
    return {'portfolio': portfolio_df, 'trades': trades_df, 'metrics': metrics}


# ============================================
# 11. 전체 비교 실행
# ============================================

def run_all_versions(df):
    """4가지 버전 비교"""
    print("\n" + "=" * 80)
    print("🧪 섹터 필터 백테스트 비교")
    print(f"   섹터 모멘텀 기간: {SECTOR_MOMENTUM_PERIOD}일")
    print(f"   섹터 RSI 상한: {SECTOR_RSI_UPPER}")
    print("=" * 80 + "\n")
    
    # 섹터 정보 준비
    from src.data import get_sp500_list
    sp500 = get_sp500_list()
    sector_map = dict(zip(sp500['symbol'], sp500['sector']))
    
    # 섹터 ETF 데이터 다운로드
    start_date = df['date'].min()
    end_date = df['date'].max()
    sector_df = get_sector_etf_data(start_date, end_date)
    
    results = {}
    
    results['A'] = run_backtest_A(df, sector_df, sector_map)
    results['B'] = run_backtest_B(df, sector_df, sector_map)
    results['C'] = run_backtest_C(df, sector_df, sector_map)
    results['D'] = run_backtest_D(df, sector_df, sector_map)
    
    # 비교 테이블
    print("\n" + "=" * 90)
    print("📊 결과 비교")
    print("=" * 90)
    
    print(f"\n{'버전':<6} {'설명':<35} {'총수익률':>12} {'CAGR':>10} {'MDD':>10} {'샤프':>8}")
    print("-" * 90)
    
    descriptions = {
        'A': '섹터 필터 (SPY 대비)',
        'B': '섹터 필터 + RSI',
        'C': '섹터 필터 + RSI + 섹터당 1종목',
        'D': '1위 섹터에서 Top 3'
    }
    
    for ver in ['A', 'B', 'C', 'D']:
        m = results[ver]['metrics']
        desc = descriptions[ver]
        print(f"{ver:<6} {desc:<35} {m['total_return']*100:>11.2f}% {m['cagr']*100:>9.2f}% {m['mdd']*100:>9.2f}% {m['sharpe_ratio']:>8.2f}")
    
    print("-" * 90)
    
    spy_ret = results['A']['metrics']['spy_return']
    print(f"{'SPY':<6} {'벤치마크':<35} {spy_ret*100:>11.2f}%")
    
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
# 12. 성과 지표 계산
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
# 13. 결과 출력
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
# 14. 그래프
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
    print("  A: 섹터 필터만 (SPY 대비 수익률 > 0인 섹터)")
    print("  B: 섹터 필터 + RSI (RSI < 70)")
    print("  C: 섹터 필터 + RSI + 섹터당 1종목")
    print("  D: SPY 대비 1위 섹터에서 Top 3 종목")
