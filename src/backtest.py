# ============================================
# 파일명: src/backtest.py
# 설명: 백테스트 (5가지 버전 비교)
# 
# 버전 A: 월/목 점수 → 화/금 종가 (기본)
# 버전 B: 월/목 점수 → 화/금 시초가
# 버전 C: 화 점수 → 수 종가
# 버전 D: 장기 모멘텀 (1주, 1달, 2달)
# 버전 E: 장기 모멘텀 (가중치 변경)
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

TOP_N = 3                    # 상위 종목 수
ALLOCATIONS = [0.4, 0.3, 0.3]  # 투자 비중


# ============================================
# 1. 데이터 전처리
# ============================================

def prepare_price_data(df):
    """
    피벗 테이블로 변환 (날짜 × 종목)
    """
    price_df = df.pivot(index='date', columns='symbol', values='close')
    return price_df


def prepare_open_data(df):
    """
    시초가 피벗 테이블 (버전 B용)
    """
    if 'open' not in df.columns:
        # open 없으면 close 사용
        return df.pivot(index='date', columns='symbol', values='close')
    return df.pivot(index='date', columns='symbol', values='open')


def filter_by_weekday(price_df, weekdays):
    """
    특정 요일만 필터링
    """
    price_df = price_df.copy()
    mask = price_df.index.day_name().isin(weekdays)
    return price_df[mask]


# ============================================
# 2. 모멘텀 점수 계산 (버전별)
# ============================================

def calc_scores_ABC(price_df):
    """
    버전 A, B, C용 점수 계산
    (1주×3.5) + (2주×2.5) + (3주×1.5)
    
    주 2회 데이터 기준:
    - 2회 전 = 1주
    - 4회 전 = 2주  
    - 6회 전 = 3주
    """
    ret_1w = price_df.pct_change(2)   # 1주 (2회 전)
    ret_2w = price_df.pct_change(4)   # 2주 (4회 전)
    ret_3w = price_df.pct_change(6)   # 3주 (6회 전)
    
    score_df = (ret_1w * 3.5) + (ret_2w * 2.5) + (ret_3w * 1.5)
    
    return score_df, ret_1w


def calc_scores_D(price_df):
    """
    버전 D용 점수 계산 (장기 모멘텀)
    (1주×3.5) + (1달×2.5) + (2달×1.5)
    
    주 2회 데이터 기준:
    - 2회 전 = 1주
    - 8회 전 = 1달 (4주)
    - 16회 전 = 2달 (8주)
    """
    ret_1w = price_df.pct_change(2)    # 1주
    ret_1m = price_df.pct_change(8)    # 1달
    ret_2m = price_df.pct_change(16)   # 2달
    
    score_df = (ret_1w * 3.5) + (ret_1m * 2.5) + (ret_2m * 1.5)
    
    return score_df, ret_1w


def calc_scores_E(price_df):
    """
    버전 E용 점수 계산 (장기 모멘텀 + 가중치 변경)
    (1주×5.5) + (1달×3) + (2달×1.5)
    """
    ret_1w = price_df.pct_change(2)    # 1주
    ret_1m = price_df.pct_change(8)    # 1달
    ret_2m = price_df.pct_change(16)   # 2달
    
    score_df = (ret_1w * 5.5) + (ret_1m * 3.0) + (ret_2m * 1.5)
    
    return score_df, ret_1w


# ============================================
# 3. 매수일 매핑 생성
# ============================================

def create_trade_mapping(df, score_day, trade_day):
    """
    점수 계산일 → 매수일 매핑
    
    score_day: 점수 계산 요일 (예: 'Monday')
    trade_day: 매수 요일 (예: 'Tuesday')
    """
    dates = sorted(df['date'].unique())
    date_weekday = {d: pd.Timestamp(d).day_name() for d in dates}
    
    trade_map = {}
    
    for i, date in enumerate(dates):
        if date_weekday[date] == score_day:
            # 다음 trade_day 찾기
            for j in range(i+1, len(dates)):
                if date_weekday[dates[j]] == trade_day:
                    trade_map[date] = dates[j]
                    break
    
    return trade_map


# ============================================
# 4. 백테스트 실행 (공통)
# ============================================

def run_backtest_core(df, price_df, score_df, ret_1w, trade_map, use_open=False):
    """
    백테스트 핵심 로직
    
    use_open: True면 시초가 매수, False면 종가 매수
    """
    df_daily = df.copy().sort_values('date').reset_index(drop=True)
    daily_dates = sorted(df_daily['date'].unique())
    
    # 시초가 데이터 (버전 B용)
    if use_open:
        open_df = prepare_open_data(df)
    
    score_dates = score_df.dropna(how='all').index.tolist()
    
    portfolio_values = []
    trades = []
    
    cash = INITIAL_CAPITAL
    holdings = {}
    pending_order = None
    
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
        
        # ----- 대기 중인 매수 주문 실행 -----
        if pending_order is not None and pending_order['trade_date'] == date:
            order = pending_order
            pending_order = None
            
            # 기존 매도
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
                # 시초가 또는 종가 선택
                if use_open and date_ts in open_df.index and symbol in open_df.columns:
                    buy_price = open_df.loc[date_ts, symbol]
                else:
                    stock = today_data[today_data['symbol'] == symbol]
                    if stock.empty:
                        continue
                    buy_price = stock.iloc[0]['close']
                
                if pd.isna(buy_price):
                    continue
                
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
        
        # ----- 점수 계산일 확인 -----
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
        
        # ----- 종목 선정 -----
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
            'scores': top_n.values.tolist()
        }
    
    portfolio_df = pd.DataFrame(portfolio_values)
    trades_df = pd.DataFrame(trades) if trades else pd.DataFrame()
    
    return portfolio_df, trades_df


# ============================================
# 5. 버전별 백테스트 실행
# ============================================

def run_backtest_A(df):
    """
    버전 A: 월/목 점수 → 화/금 종가 매수
    """
    print("=" * 50)
    print("[버전 A] 월/목 점수 → 화/금 종가")
    print("=" * 50)
    
    price_df = prepare_price_data(df)
    
    # 월/목 필터링
    biweekly = filter_by_weekday(price_df, ['Monday', 'Thursday'])
    biweekly = biweekly.dropna(subset=['SPY']) if 'SPY' in biweekly.columns else biweekly
    
    score_df, ret_1w = calc_scores_ABC(biweekly)
    
    # 매핑: 월→화, 목→금
    trade_map_mon = create_trade_mapping(df, 'Monday', 'Tuesday')
    trade_map_thu = create_trade_mapping(df, 'Thursday', 'Friday')
    trade_map = {**trade_map_mon, **trade_map_thu}
    
    portfolio_df, trades_df = run_backtest_core(df, price_df, score_df, ret_1w, trade_map, use_open=False)
    metrics = calculate_metrics(portfolio_df, trades_df, df)
    
    return {'portfolio': portfolio_df, 'trades': trades_df, 'metrics': metrics}


def run_backtest_B(df):
    """
    버전 B: 월/목 점수 → 화/금 시초가 매수
    """
    print("=" * 50)
    print("[버전 B] 월/목 점수 → 화/금 시초가")
    print("=" * 50)
    
    price_df = prepare_price_data(df)
    
    biweekly = filter_by_weekday(price_df, ['Monday', 'Thursday'])
    biweekly = biweekly.dropna(subset=['SPY']) if 'SPY' in biweekly.columns else biweekly
    
    score_df, ret_1w = calc_scores_ABC(biweekly)
    
    trade_map_mon = create_trade_mapping(df, 'Monday', 'Tuesday')
    trade_map_thu = create_trade_mapping(df, 'Thursday', 'Friday')
    trade_map = {**trade_map_mon, **trade_map_thu}
    
    portfolio_df, trades_df = run_backtest_core(df, price_df, score_df, ret_1w, trade_map, use_open=True)
    metrics = calculate_metrics(portfolio_df, trades_df, df)
    
    return {'portfolio': portfolio_df, 'trades': trades_df, 'metrics': metrics}


def run_backtest_C(df):
    """
    버전 C: 화 점수 → 수 종가 매수
    """
    print("=" * 50)
    print("[버전 C] 화 점수 → 수 종가")
    print("=" * 50)
    
    price_df = prepare_price_data(df)
    
    # 화요일만 필터링
    weekly = filter_by_weekday(price_df, ['Tuesday'])
    weekly = weekly.dropna(subset=['SPY']) if 'SPY' in weekly.columns else weekly
    
    score_df, ret_1w = calc_scores_ABC(weekly)
    
    # 매핑: 화→수
    trade_map = create_trade_mapping(df, 'Tuesday', 'Wednesday')
    
    portfolio_df, trades_df = run_backtest_core(df, price_df, score_df, ret_1w, trade_map, use_open=False)
    metrics = calculate_metrics(portfolio_df, trades_df, df)
    
    return {'portfolio': portfolio_df, 'trades': trades_df, 'metrics': metrics}


def run_backtest_D(df):
    """
    버전 D: 장기 모멘텀 (1주, 1달, 2달)
    (1주×3.5) + (1달×2.5) + (2달×1.5)
    """
    print("=" * 50)
    print("[버전 D] 장기 모멘텀 (1주×3.5 + 1달×2.5 + 2달×1.5)")
    print("=" * 50)
    
    price_df = prepare_price_data(df)
    
    biweekly = filter_by_weekday(price_df, ['Monday', 'Thursday'])
    biweekly = biweekly.dropna(subset=['SPY']) if 'SPY' in biweekly.columns else biweekly
    
    score_df, ret_1w = calc_scores_D(biweekly)
    
    trade_map_mon = create_trade_mapping(df, 'Monday', 'Tuesday')
    trade_map_thu = create_trade_mapping(df, 'Thursday', 'Friday')
    trade_map = {**trade_map_mon, **trade_map_thu}
    
    portfolio_df, trades_df = run_backtest_core(df, price_df, score_df, ret_1w, trade_map, use_open=False)
    metrics = calculate_metrics(portfolio_df, trades_df, df)
    
    return {'portfolio': portfolio_df, 'trades': trades_df, 'metrics': metrics}


def run_backtest_E(df):
    """
    버전 E: 장기 모멘텀 + 가중치 변경
    (1주×5.5) + (1달×3) + (2달×1.5)
    """
    print("=" * 50)
    print("[버전 E] 장기 모멘텀 (1주×5.5 + 1달×3 + 2달×1.5)")
    print("=" * 50)
    
    price_df = prepare_price_data(df)
    
    biweekly = filter_by_weekday(price_df, ['Monday', 'Thursday'])
    biweekly = biweekly.dropna(subset=['SPY']) if 'SPY' in biweekly.columns else biweekly
    
    score_df, ret_1w = calc_scores_E(biweekly)
    
    trade_map_mon = create_trade_mapping(df, 'Monday', 'Tuesday')
    trade_map_thu = create_trade_mapping(df, 'Thursday', 'Friday')
    trade_map = {**trade_map_mon, **trade_map_thu}
    
    portfolio_df, trades_df = run_backtest_core(df, price_df, score_df, ret_1w, trade_map, use_open=False)
    metrics = calculate_metrics(portfolio_df, trades_df, df)
    
    return {'portfolio': portfolio_df, 'trades': trades_df, 'metrics': metrics}


# ============================================
# 6. 전체 비교 실행
# ============================================

def run_all_versions(df):
    """
    5가지 버전 모두 실행하고 비교
    """
    print("\n" + "=" * 60)
    print("🧪 5가지 버전 백테스트 비교")
    print("=" * 60)
    
    results = {}
    
    results['A'] = run_backtest_A(df)
    results['B'] = run_backtest_B(df)
    results['C'] = run_backtest_C(df)
    results['D'] = run_backtest_D(df)
    results['E'] = run_backtest_E(df)
    
    # 비교 테이블 출력
    print("\n" + "=" * 60)
    print("📊 결과 비교")
    print("=" * 60)
    
    print(f"\n{'버전':<8} {'총수익률':>10} {'CAGR':>10} {'MDD':>10} {'샤프':>8} {'거래수':>8}")
    print("-" * 60)
    
    for ver, res in results.items():
        m = res['metrics']
        print(f"{ver:<8} {m['total_return']*100:>9.2f}% {m['cagr']*100:>9.2f}% {m['mdd']*100:>9.2f}% {m['sharpe_ratio']:>8.2f} {m['total_trades']:>8}")
    
    print("-" * 60)
    
    # SPY 수익률
    spy_ret = results['A']['metrics']['spy_return']
    print(f"{'SPY':<8} {spy_ret*100:>9.2f}%")
    
    return results


# ============================================
# 7. 성과 지표 계산
# ============================================

def calculate_metrics(portfolio_df, trades_df, df):
    """
    성과 지표 계산
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
# 8. 결과 출력
# ============================================

def print_metrics(metrics, trades_df=None):
    """
    성과 지표 출력
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
    
    # 최근 매수 내역
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
                    print(f"  {i+1}위: {row['symbol']:5} | 점수: {score:.4f} | 가격: ${row['price']:.2f}")
    
    print("\n" + "=" * 50)


# ============================================
# 9. 그래프 (버전 비교)
# ============================================

def plot_comparison(results, df):
    """
    5가지 버전 성과 비교 그래프
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. 포트폴리오 가치 비교
    ax1 = axes[0, 0]
    
    colors = {'A': 'blue', 'B': 'green', 'C': 'red', 'D': 'purple', 'E': 'orange'}
    
    for ver, res in results.items():
        portfolio = res['portfolio'].copy()
        portfolio['normalized'] = portfolio['value'] / portfolio['value'].iloc[0] * 100
        ax1.plot(portfolio['date'], portfolio['normalized'], 
                 label=f'버전 {ver}', linewidth=2, color=colors[ver])
    
    # SPY
    if 'SPY' in df['symbol'].unique():
        spy = df[df['symbol'] == 'SPY'].sort_values('date').copy()
        spy['normalized'] = spy['close'] / spy['close'].iloc[0] * 100
        ax1.plot(spy['date'], spy['normalized'], 
                 label='SPY', linewidth=2, linestyle='--', color='gray')
    
    ax1.set_title('포트폴리오 가치 비교 (시작=100)', fontsize=12)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. 총 수익률 비교
    ax2 = axes[0, 1]
    versions = list(results.keys())
    returns = [results[v]['metrics']['total_return'] * 100 for v in versions]
    bars = ax2.bar(versions, returns, color=[colors[v] for v in versions])
    ax2.axhline(y=results['A']['metrics']['spy_return']*100, color='gray', linestyle='--', label='SPY')
    ax2.set_title('총 수익률 비교 (%)', fontsize=12)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. MDD 비교
    ax3 = axes[1, 0]
    mdds = [results[v]['metrics']['mdd'] * 100 for v in versions]
    ax3.bar(versions, mdds, color=[colors[v] for v in versions])
    ax3.set_title('최대 낙폭 (MDD) 비교 (%)', fontsize=12)
    ax3.grid(True, alpha=0.3)
    
    # 4. 샤프 비율 비교
    ax4 = axes[1, 1]
    sharpes = [results[v]['metrics']['sharpe_ratio'] for v in versions]
    ax4.bar(versions, sharpes, color=[colors[v] for v in versions])
    ax4.set_title('샤프 비율 비교', fontsize=12)
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
