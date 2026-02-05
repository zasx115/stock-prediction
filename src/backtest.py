# ============================================
# 파일명: src/backtest.py
# 설명: 백테스트 (3가지 버전 비교)
# 
# 버전 A: 듀얼 모멘텀 (절대 + 상대)
# 버전 B: 변동성 조절
# 버전 C: 듀얼 모멘텀 + 변동성 조절
# 
# 공통:
# - 화요일 점수 → 수요일 종가 매수
# - 같은 종목이면 비중만 조절
# - 손절 -7%
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

SLIPPAGE = 0.001             # 슬리피지 (0.1%)

STOP_LOSS = -0.07            # 손절 기준 (-7%)

# 모멘텀 점수 가중치
WEIGHT_1W = 3.5
WEIGHT_2W = 2.5
WEIGHT_3W = 1.5

TOP_N = 3
ALLOCATIONS = [0.4, 0.3, 0.3]

# 듀얼 모멘텀 설정
ABSOLUTE_MOMENTUM_PERIOD = 63  # 3개월 (약 63 거래일)

# 변동성 조절 설정
TARGET_VOLATILITY = 0.15      # 목표 변동성 15%
VOLATILITY_LOOKBACK = 20      # 변동성 계산 기간 (20일)
MIN_WEIGHT = 0.2              # 최소 투자 비중 20%
MAX_WEIGHT = 1.0              # 최대 투자 비중 100%


# ============================================
# 1. 데이터 전처리
# ============================================

def prepare_price_data(df):
    """피벗 테이블로 변환"""
    price_df = df.pivot(index='date', columns='symbol', values='close')
    return price_df


def filter_tuesday(price_df):
    """화요일만 필터링"""
    price_df = price_df.copy()
    mask = price_df.index.day_name() == 'Tuesday'
    return price_df[mask]


# ============================================
# 2. 모멘텀 점수 계산
# ============================================

def calc_momentum_scores(weekly_df):
    """모멘텀 점수 계산"""
    ret_1w = weekly_df.pct_change(1)
    ret_2w = weekly_df.pct_change(2)
    ret_3w = weekly_df.pct_change(3)
    
    score_df = (ret_1w * WEIGHT_1W) + (ret_2w * WEIGHT_2W) + (ret_3w * WEIGHT_3W)
    
    return score_df, ret_1w


# ============================================
# 3. 절대 모멘텀 계산 (버전 A, C)
# ============================================

def calc_absolute_momentum(price_df, period=ABSOLUTE_MOMENTUM_PERIOD):
    """
    SPY 절대 모멘텀 계산
    
    SPY의 N일 수익률 > 0 이면 True
    """
    if 'SPY' not in price_df.columns:
        return pd.Series(True, index=price_df.index)
    
    spy = price_df['SPY']
    spy_return = spy.pct_change(period)
    
    # True = 상승장, False = 하락장
    absolute_momentum = spy_return > 0
    
    return absolute_momentum


# ============================================
# 4. 변동성 계산 (버전 B, C)
# ============================================

def calc_volatility_weight(price_df, lookback=VOLATILITY_LOOKBACK):
    """
    변동성 기반 투자 비중 계산
    
    투자 비중 = 목표 변동성 / 현재 변동성
    """
    if 'SPY' not in price_df.columns:
        return pd.Series(1.0, index=price_df.index)
    
    spy = price_df['SPY']
    daily_returns = spy.pct_change()
    
    # 20일 롤링 변동성 (연율화)
    rolling_vol = daily_returns.rolling(lookback).std() * np.sqrt(252)
    
    # 투자 비중 계산
    weight = TARGET_VOLATILITY / rolling_vol
    
    # 최소/최대 제한
    weight = weight.clip(MIN_WEIGHT, MAX_WEIGHT)
    
    return weight


# ============================================
# 5. 매수일 매핑 생성
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

def run_backtest_core(df, version='A'):
    """
    백테스트 핵심 로직
    
    version:
    - 'A': 듀얼 모멘텀만
    - 'B': 변동성 조절만
    - 'C': 둘 다 적용
    - 'BASE': 기본 (비교용)
    """
    
    # 데이터 준비
    df_daily = df.copy().sort_values('date').reset_index(drop=True)
    daily_dates = sorted(df_daily['date'].unique())
    
    price_df = prepare_price_data(df)
    tuesday_df = filter_tuesday(price_df)
    
    if 'SPY' in tuesday_df.columns:
        tuesday_df = tuesday_df.dropna(subset=['SPY'])
    
    score_df, ret_1w = calc_momentum_scores(tuesday_df)
    
    # 버전별 추가 계산
    if version in ['A', 'C']:
        absolute_momentum = calc_absolute_momentum(price_df)
    else:
        absolute_momentum = pd.Series(True, index=price_df.index)
    
    if version in ['B', 'C']:
        volatility_weight = calc_volatility_weight(price_df)
    else:
        volatility_weight = pd.Series(1.0, index=price_df.index)
    
    trade_map = create_trade_mapping(df)
    score_dates = score_df.dropna(how='all').index.tolist()
    
    # 결과 저장
    portfolio_values = []
    trades = []
    
    # 현재 상태
    cash = INITIAL_CAPITAL
    holdings = {}
    pending_order = None
    
    # ----- 매일 시뮬레이션 -----
    for i, date in enumerate(daily_dates):
        today_data = df_daily[df_daily['date'] == date]
        date_ts = pd.Timestamp(date)
        
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
        
        # ----- 대기 중인 매수 주문 실행 -----
        if pending_order is not None and pending_order['trade_date'] == date:
            order = pending_order
            pending_order = None
            
            # 절대 모멘텀 체크 (버전 A, C)
            use_absolute = order.get('absolute_momentum', True)
            
            # 변동성 비중 (버전 B, C)
            vol_weight = order.get('volatility_weight', 1.0)
            
            # 절대 모멘텀 실패 → 전량 매도 후 현금 보유
            if not use_absolute:
                for symbol, info in list(holdings.items()):
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
                            'action': 'SELL_CASH',
                            'shares': info['shares'],
                            'price': sell_price,
                            'amount': sell_amount,
                            'commission': commission,
                            'slippage': base_price * SLIPPAGE * info['shares'],
                            'return_rate': return_rate
                        })
                
                holdings = {}
                continue  # 현금 보유, 매수 안 함
            
            new_picks = order['picks']
            new_scores = order['scores']
            
            current_holdings = set(holdings.keys())
            new_holdings_set = set(new_picks)
            
            to_sell = current_holdings - new_holdings_set
            to_buy = new_holdings_set - current_holdings
            to_keep = current_holdings & new_holdings_set
            
            # ----- 1. 매도할 종목 -----
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
            
            # ----- 2. 비중 계산 (변동성 조절 적용) -----
            n_picks = len(new_picks)
            if n_picks >= 3:
                base_allocations = ALLOCATIONS[:3]
            elif n_picks == 2:
                base_allocations = [0.5, 0.5]
            elif n_picks == 1:
                base_allocations = [1.0]
            else:
                base_allocations = []
            
            # 변동성 비중 적용
            adjusted_allocations = [a * vol_weight for a in base_allocations]
            
            target_allocations = {}
            for j, symbol in enumerate(new_picks):
                if j < len(adjusted_allocations):
                    target_allocations[symbol] = adjusted_allocations[j]
            
            # ----- 3. 유지 종목 비중 조절 -----
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
                                'return_rate': 0,
                                'score': target_allocations.get(symbol, 0)
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
            
            # ----- 4. 신규 매수 -----
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
        
        # ----- 화요일: 점수 계산 & 종목 선정 -----
        if date_ts not in score_dates:
            continue
        
        if date not in trade_map:
            continue
        
        trade_date = trade_map[date]
        
        # ----- 시장 필터 -----
        if date_ts not in ret_1w.index:
            continue
        
        market_momentum = ret_1w.loc[date_ts].mean()
        
        if market_momentum <= 0:
            continue
        
        # ----- 절대 모멘텀 체크 (버전 A, C) -----
        use_absolute = True
        if version in ['A', 'C']:
            if date_ts in absolute_momentum.index:
                use_absolute = absolute_momentum.loc[date_ts]
            else:
                # 가장 가까운 날짜 찾기
                closest_date = absolute_momentum.index[absolute_momentum.index <= date_ts]
                if len(closest_date) > 0:
                    use_absolute = absolute_momentum.loc[closest_date[-1]]
        
        # ----- 변동성 비중 (버전 B, C) -----
        vol_weight = 1.0
        if version in ['B', 'C']:
            if date_ts in volatility_weight.index:
                vol_weight = volatility_weight.loc[date_ts]
            else:
                closest_date = volatility_weight.index[volatility_weight.index <= date_ts]
                if len(closest_date) > 0:
                    vol_weight = volatility_weight.loc[closest_date[-1]]
            
            if pd.isna(vol_weight):
                vol_weight = 1.0
        
        # ----- 상위 종목 선정 -----
        if date_ts not in score_df.index:
            continue
        
        current_scores = score_df.loc[date_ts].drop(labels=['SPY'], errors='ignore').dropna()
        
        if current_scores.empty:
            continue
        
        top_n = current_scores.nlargest(TOP_N)
        
        # ----- 매수 주문 대기 -----
        pending_order = {
            'score_date': date,
            'trade_date': trade_date,
            'picks': top_n.index.tolist(),
            'scores': top_n.values.tolist(),
            'absolute_momentum': use_absolute,
            'volatility_weight': vol_weight
        }
    
    portfolio_df = pd.DataFrame(portfolio_values)
    trades_df = pd.DataFrame(trades) if trades else pd.DataFrame()
    
    return portfolio_df, trades_df


# ============================================
# 7. 버전별 백테스트 실행
# ============================================

def run_backtest_A(df):
    """버전 A: 듀얼 모멘텀만"""
    print("[버전 A] 듀얼 모멘텀 (절대 + 상대)")
    portfolio_df, trades_df = run_backtest_core(df, version='A')
    metrics = calculate_metrics(portfolio_df, trades_df, df)
    return {'portfolio': portfolio_df, 'trades': trades_df, 'metrics': metrics}


def run_backtest_B(df):
    """버전 B: 변동성 조절만"""
    print("[버전 B] 변동성 조절")
    portfolio_df, trades_df = run_backtest_core(df, version='B')
    metrics = calculate_metrics(portfolio_df, trades_df, df)
    return {'portfolio': portfolio_df, 'trades': trades_df, 'metrics': metrics}


def run_backtest_C(df):
    """버전 C: 둘 다 적용"""
    print("[버전 C] 듀얼 모멘텀 + 변동성 조절")
    portfolio_df, trades_df = run_backtest_core(df, version='C')
    metrics = calculate_metrics(portfolio_df, trades_df, df)
    return {'portfolio': portfolio_df, 'trades': trades_df, 'metrics': metrics}


def run_backtest_base(df):
    """기본 버전 (비교용)"""
    print("[BASE] 기본 전략")
    portfolio_df, trades_df = run_backtest_core(df, version='BASE')
    metrics = calculate_metrics(portfolio_df, trades_df, df)
    return {'portfolio': portfolio_df, 'trades': trades_df, 'metrics': metrics}


# ============================================
# 8. 전체 비교 실행
# ============================================

def run_all_versions(df):
    """3가지 버전 + 기본 비교"""
    print("\n" + "=" * 80)
    print("🧪 백테스트 버전 비교")
    print(f"   목표 변동성: {TARGET_VOLATILITY*100:.0f}%")
    print(f"   절대 모멘텀 기간: {ABSOLUTE_MOMENTUM_PERIOD}일 (약 3개월)")
    print("=" * 80 + "\n")
    
    results = {}
    
    results['BASE'] = run_backtest_base(df)
    results['A'] = run_backtest_A(df)
    results['B'] = run_backtest_B(df)
    results['C'] = run_backtest_C(df)
    
    # 비교 테이블
    print("\n" + "=" * 90)
    print("📊 결과 비교")
    print("=" * 90)
    
    print(f"\n{'버전':<8} {'설명':<30} {'총수익률':>12} {'CAGR':>10} {'MDD':>10} {'샤프':>8}")
    print("-" * 90)
    
    descriptions = {
        'BASE': '기본 (비교용)',
        'A': '듀얼 모멘텀 (절대+상대)',
        'B': '변동성 조절',
        'C': '듀얼 모멘텀 + 변동성 조절'
    }
    
    for ver in ['BASE', 'A', 'B', 'C']:
        m = results[ver]['metrics']
        desc = descriptions[ver]
        print(f"{ver:<8} {desc:<30} {m['total_return']*100:>11.2f}% {m['cagr']*100:>9.2f}% {m['mdd']*100:>9.2f}% {m['sharpe_ratio']:>8.2f}")
    
    print("-" * 90)
    
    spy_ret = results['BASE']['metrics']['spy_return']
    print(f"{'SPY':<8} {'벤치마크':<30} {spy_ret*100:>11.2f}%")
    
    print("=" * 90)
    
    # 거래 통계
    print("\n💸 거래 & 비용 통계")
    print("-" * 90)
    print(f"{'버전':<8} {'거래횟수':>10} {'수수료':>15} {'슬리피지':>15} {'현금보유일':>12}")
    print("-" * 90)
    
    for ver in ['BASE', 'A', 'B', 'C']:
        m = results[ver]['metrics']
        t = results[ver]['trades']
        
        # 현금 보유 일수 계산
        cash_days = len(t[t['action'] == 'SELL_CASH']) if not t.empty and 'action' in t.columns else 0
        
        print(f"{ver:<8} {m['total_trades']:>10} ${m['total_commission']:>14.2f} ${m['total_slippage']:>14.2f} {cash_days:>12}")
    
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
    cash_count = len(trades_df[trades_df['action'] == 'SELL_CASH']) if not trades_df.empty else 0
    
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
        'cash_count': cash_count,
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
    print(f"    - 현금 전환 (CASH): {metrics['cash_count']}회")
    print(f"    - 손절 (STOP_LOSS): {metrics['stop_loss_count']}회")
    print(f"  총 수수료: ${metrics['total_commission']:,.2f}")
    print(f"  총 슬리피지: ${metrics['total_slippage']:,.2f}")
    print(f"  총 비용: ${metrics['total_commission'] + metrics['total_slippage']:,.2f}")
    
    print("\n" + "=" * 60)


# ============================================
# 11. 그래프 (버전 비교)
# ============================================

def plot_comparison(results, df):
    """4가지 버전 비교 그래프"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    colors = {'BASE': 'gray', 'A': 'blue', 'B': 'green', 'C': 'red'}
    
    # 1. 포트폴리오 가치 비교
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
    
    ax1.set_title('Portfolio Value Comparison (Start=100)', fontsize=12)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # 2. 총 수익률 & CAGR 비교
    ax2 = axes[0, 1]
    versions = list(results.keys())
    x = np.arange(len(versions))
    width = 0.35
    
    returns = [results[v]['metrics']['total_return'] * 100 for v in versions]
    cagrs = [results[v]['metrics']['cagr'] * 100 for v in versions]
    
    bars1 = ax2.bar(x - width/2, returns, width, label='Total Return', color='steelblue')
    bars2 = ax2.bar(x + width/2, cagrs, width, label='CAGR', color='lightsteelblue')
    
    ax2.axhline(y=results['BASE']['metrics']['spy_return']*100, color='orange', linestyle='--', label='SPY')
    ax2.set_xticks(x)
    ax2.set_xticklabels(versions)
    ax2.set_title('Total Return & CAGR (%)', fontsize=12)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. MDD 비교
    ax3 = axes[1, 0]
    mdds = [results[v]['metrics']['mdd'] * 100 for v in versions]
    ax3.bar(versions, mdds, color=[colors[v] for v in versions])
    ax3.set_title('Maximum Drawdown (%)', fontsize=12)
    ax3.grid(True, alpha=0.3)
    
    # 4. 샤프 비율 비교
    ax4 = axes[1, 1]
    sharpes = [results[v]['metrics']['sharpe_ratio'] for v in versions]
    ax4.bar(versions, sharpes, color=[colors[v] for v in versions])
    ax4.set_title('Sharpe Ratio', fontsize=12)
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print("\n📋 버전 설명:")
    print("  BASE: 기본 전략 (비교용)")
    print("  A: 듀얼 모멘텀 (SPY 3개월 > 0 일 때만 매수)")
    print("  B: 변동성 조절 (변동성 높으면 비중 축소)")
    print("  C: 듀얼 모멘텀 + 변동성 조절 (둘 다 적용)")
