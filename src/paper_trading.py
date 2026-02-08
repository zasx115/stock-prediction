# ============================================

# 파일명: src/paper_trading.py

# 설명: 페이퍼 트레이딩 (가상 매매) 시스템

# 

# 기능:

# - 실시간 매수/매도 신호 생성

# - 가상 포트폴리오 관리

# - 거래 기록 저장 (CSV, Google Sheets)

# - 백테스트 결과와 비교

# 

# 사용법:

# python paper_trading.py signal     # 오늘 신호 확인

# python paper_trading.py portfolio  # 포트폴리오 상태

# python paper_trading.py execute    # 신호대로 가상 매매 실행

# ============================================

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import os
import json

from src.strategy import (
CustomStrategy,
prepare_price_data,
filter_tuesday,
create_trade_mapping
)

# ============================================

# [설정] 페이퍼 트레이딩 파라미터

# ============================================

# —– 자본금 —–

INITIAL_CAPITAL = 2000       # 초기 가상 자본금

# —– 거래 비용 —–

BUY_COMMISSION = 0.0025      # 매수 수수료 (0.25%)
SELL_COMMISSION = 0.0025     # 매도 수수료 (0.25%)
SLIPPAGE = 0.001             # 슬리피지 (0.1%)

# —– 리스크 관리 —–

STOP_LOSS = -0.07            # 손절 기준 (-7%)

# —– 데이터 기간 —–

LOOKBACK_DAYS = 200          # 모멘텀 계산에 필요한 과거 데이터 (약 6개월 + 여유)

# —– 저장 경로 —–

DATA_DIR = “paper_trading_data”
PORTFOLIO_FILE = f”{DATA_DIR}/portfolio.json”
TRADES_FILE = f”{DATA_DIR}/trades.csv”
SIGNALS_FILE = f”{DATA_DIR}/signals.csv”

# ============================================

# [1] 데이터 다운로드

# ============================================

def get_sp500_list():
“””
S&P 500 종목 리스트 가져오기
“””
url = ‘https://en.wikipedia.org/wiki/List_of_S%26P_500_companies’
tables = pd.read_html(url)
df = tables[0]
df = df[[‘Symbol’, ‘Security’, ‘GICS Sector’]].copy()
df.columns = [‘symbol’, ‘company’, ‘sector’]
df[‘symbol’] = df[‘symbol’].str.replace(’.’, ‘-’, regex=False)
return df

def download_recent_data(symbols, days=LOOKBACK_DAYS):
“””
최근 N일 데이터 다운로드

```
Args:
    symbols: 종목 리스트
    days: 다운로드할 일수

Returns:
    DataFrame: 주가 데이터
"""
end_date = datetime.now()
start_date = end_date - timedelta(days=days)

print(f"데이터 다운로드 중...")
print(f"  종목 수: {len(symbols)}개")
print(f"  기간: {start_date.strftime('%Y-%m-%d')} ~ {end_date.strftime('%Y-%m-%d')}")

data = yf.download(
    symbols,
    start=start_date.strftime('%Y-%m-%d'),
    end=end_date.strftime('%Y-%m-%d'),
    auto_adjust=True,
    threads=True,
    progress=False
)

if data.empty:
    print("❌ 데이터 다운로드 실패")
    return pd.DataFrame()

# 멀티인덱스 → 일반 DataFrame
result = []

if len(symbols) == 1:
    df = data.copy()
    df['symbol'] = symbols[0]
    df = df.reset_index()
    df.columns = ['date', 'close', 'high', 'low', 'open', 'volume', 'symbol']
    result.append(df)
else:
    for symbol in symbols:
        try:
            if symbol not in data['Close'].columns:
                continue
            
            df = pd.DataFrame({
                'date': data.index,
                'open': data['Open'][symbol].values,
                'high': data['High'][symbol].values,
                'low': data['Low'][symbol].values,
                'close': data['Close'][symbol].values,
                'volume': data['Volume'][symbol].values,
                'symbol': symbol
            })
            
            df = df.dropna(subset=['close'])
            if not df.empty:
                result.append(df)
        except:
            continue

if result:
    final_df = pd.concat(result, ignore_index=True)
    final_df['date'] = pd.to_datetime(final_df['date'])
    print(f"✅ 다운로드 완료! ({final_df['symbol'].nunique()}개 종목)")
    return final_df

return pd.DataFrame()
```

# ============================================

# [2] 매수 신호 생성

# ============================================

def get_today_signal(strategy=None, target_date=None):
“””
오늘(또는 특정 날짜) 기준 매수 신호 생성

```
Args:
    strategy: CustomStrategy 인스턴스 (없으면 기본값)
    target_date: 목표 날짜 (None이면 오늘)

Returns:
    dict: {
        'date': 날짜,
        'signal': 'BUY' 또는 'HOLD',
        'picks': 종목 리스트,
        'scores': 점수 리스트,
        'allocations': 비중 리스트,
        'prices': 현재가 딕셔너리
    }
"""

if strategy is None:
    strategy = CustomStrategy()

if target_date is None:
    target_date = datetime.now()

print("=" * 60)
print(f"📡 매수 신호 생성")
print(f"   기준 날짜: {target_date.strftime('%Y-%m-%d %H:%M')}")
print("=" * 60)

# ----- 데이터 다운로드 -----
sp500 = get_sp500_list()
symbols = sp500['symbol'].tolist()
if 'SPY' not in symbols:
    symbols.append('SPY')

df = download_recent_data(symbols)

if df.empty:
    return {'signal': 'ERROR', 'message': '데이터 다운로드 실패'}

# ----- 전략 데이터 준비 -----
price_df = prepare_price_data(df)
tuesday_df = filter_tuesday(price_df)

if 'SPY' in tuesday_df.columns:
    tuesday_df = tuesday_df.dropna(subset=['SPY'])

if tuesday_df.empty:
    return {'signal': 'ERROR', 'message': '화요일 데이터 없음'}

# 전략 준비
score_df, correlation_df, ret_1m = strategy.prepare(price_df, tuesday_df)

# ----- 가장 최근 화요일 찾기 -----
score_dates = score_df.dropna(how='all').index.tolist()

if not score_dates:
    return {'signal': 'ERROR', 'message': '점수 계산 불가'}

# target_date 이전의 가장 최근 화요일
target_ts = pd.Timestamp(target_date)
valid_dates = [d for d in score_dates if d <= target_ts]

if not valid_dates:
    return {'signal': 'ERROR', 'message': '유효한 화요일 없음'}

last_tuesday = valid_dates[-1]

print(f"\n📅 분석 기준일: {last_tuesday.strftime('%Y-%m-%d')} (화요일)")

# ----- 종목 선정 -----
result = strategy.select_stocks(score_df, correlation_df, last_tuesday, ret_1m)

if result is None:
    print("\n⚠️ 시장 하락 추세 - 매수 신호 없음")
    return {
        'date': last_tuesday,
        'signal': 'HOLD',
        'message': '시장 모멘텀 <= 0',
        'picks': [],
        'scores': [],
        'allocations': [],
        'prices': {}
    }

# ----- 현재가 조회 -----
picks = result['picks']
prices = {}

for symbol in picks:
    try:
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period='1d')
        if not hist.empty:
            prices[symbol] = round(hist['Close'].iloc[-1], 2)
    except:
        prices[symbol] = None

# ----- 결과 출력 -----
print(f"\n🎯 매수 신호: {len(picks)}개 종목")
print("-" * 50)

for i, (symbol, score, alloc) in enumerate(zip(picks, result['scores'], result['allocations'])):
    price = prices.get(symbol, 'N/A')
    price_str = f"${price:.2f}" if isinstance(price, float) else price
    print(f"  {i+1}. {symbol:5} | 점수: {score:.4f} | 비중: {alloc*100:.0f}% | 현재가: {price_str}")

print("-" * 50)

return {
    'date': last_tuesday,
    'signal': 'BUY',
    'picks': picks,
    'scores': result['scores'],
    'allocations': result['allocations'],
    'prices': prices
}
```

# ============================================

# [3] 가상 포트폴리오 관리

# ============================================

def init_data_dir():
“”“데이터 디렉토리 생성”””
if not os.path.exists(DATA_DIR):
os.makedirs(DATA_DIR)
print(f”📁 디렉토리 생성: {DATA_DIR}”)

def load_portfolio():
“””
저장된 포트폴리오 불러오기

```
Returns:
    dict: {
        'cash': 현금,
        'holdings': {symbol: {'shares': int, 'avg_price': float}},
        'created_at': 생성일
    }
"""
init_data_dir()

if os.path.exists(PORTFOLIO_FILE):
    with open(PORTFOLIO_FILE, 'r') as f:
        return json.load(f)

# 신규 포트폴리오
portfolio = {
    'cash': INITIAL_CAPITAL,
    'holdings': {},
    'created_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    'last_updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
}

save_portfolio(portfolio)
return portfolio
```

def save_portfolio(portfolio):
“”“포트폴리오 저장”””
init_data_dir()

```
portfolio['last_updated'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

with open(PORTFOLIO_FILE, 'w') as f:
    json.dump(portfolio, f, indent=2)
```

def get_portfolio_value(portfolio):
“””
포트폴리오 현재 가치 계산

```
Args:
    portfolio: 포트폴리오 딕셔너리

Returns:
    dict: {
        'total': 총 가치,
        'cash': 현금,
        'stocks': 주식 가치,
        'holdings_detail': 종목별 상세
    }
"""
cash = portfolio['cash']
holdings = portfolio['holdings']

stocks_value = 0
holdings_detail = []

for symbol, info in holdings.items():
    try:
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period='1d')
        
        if not hist.empty:
            current_price = hist['Close'].iloc[-1]
            value = info['shares'] * current_price
            return_rate = (current_price - info['avg_price']) / info['avg_price']
            
            stocks_value += value
            holdings_detail.append({
                'symbol': symbol,
                'shares': info['shares'],
                'avg_price': info['avg_price'],
                'current_price': round(current_price, 2),
                'value': round(value, 2),
                'return_rate': round(return_rate * 100, 2),
                'profit': round(value - info['shares'] * info['avg_price'], 2)
            })
    except:
        continue

return {
    'total': round(cash + stocks_value, 2),
    'cash': round(cash, 2),
    'stocks': round(stocks_value, 2),
    'holdings_detail': holdings_detail
}
```

def print_portfolio(portfolio=None):
“””
포트폴리오 상태 출력

```
Args:
    portfolio: 포트폴리오 딕셔너리 (없으면 파일에서 로드)
"""
if portfolio is None:
    portfolio = load_portfolio()

value = get_portfolio_value(portfolio)

print("=" * 60)
print("💼 가상 포트폴리오 현황")
print("=" * 60)

print(f"\n💰 자산 요약")
print(f"  총 자산: ${value['total']:,.2f}")
print(f"  현금: ${value['cash']:,.2f}")
print(f"  주식: ${value['stocks']:,.2f}")

# 수익률 계산
total_return = (value['total'] - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100
print(f"\n📈 총 수익률: {total_return:+.2f}%")

if value['holdings_detail']:
    print(f"\n📊 보유 종목 ({len(value['holdings_detail'])}개)")
    print("-" * 60)
    print(f"  {'종목':5} | {'수량':>5} | {'평단가':>8} | {'현재가':>8} | {'수익률':>8} | {'평가액':>10}")
    print("-" * 60)
    
    for h in value['holdings_detail']:
        print(f"  {h['symbol']:5} | {h['shares']:>5} | ${h['avg_price']:>7.2f} | ${h['current_price']:>7.2f} | {h['return_rate']:>+7.2f}% | ${h['value']:>9.2f}")
    
    print("-" * 60)
else:
    print("\n📊 보유 종목 없음")

print(f"\n📅 생성일: {portfolio['created_at']}")
print(f"📅 최종 업데이트: {portfolio['last_updated']}")
print("=" * 60)
```

# ============================================

# [4] 가상 매매 실행

# ============================================

def execute_signal(signal, portfolio=None):
“””
신호에 따라 가상 매매 실행

```
Args:
    signal: get_today_signal() 반환값
    portfolio: 포트폴리오 (없으면 파일에서 로드)

Returns:
    list: 실행된 거래 내역
"""
if portfolio is None:
    portfolio = load_portfolio()

if signal['signal'] != 'BUY':
    print("⚠️ 매수 신호 없음 - 실행할 거래 없음")
    return []

trades = []
new_picks = set(signal['picks'])
current_holdings = set(portfolio['holdings'].keys())

to_sell = current_holdings - new_picks
to_buy = new_picks - current_holdings
to_keep = current_holdings & new_picks

print("\n" + "=" * 60)
print("🔄 가상 매매 실행")
print("=" * 60)

# 현재 포트폴리오 가치
port_value = get_portfolio_value(portfolio)
total_value = port_value['total']

# ----- 매도 처리 -----
for symbol in to_sell:
    if symbol not in portfolio['holdings']:
        continue
    
    info = portfolio['holdings'][symbol]
    
    try:
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period='1d')
        
        if not hist.empty:
            current_price = hist['Close'].iloc[-1]
            sell_price = current_price * (1 - SLIPPAGE)
            sell_amount = info['shares'] * sell_price
            commission = sell_amount * SELL_COMMISSION
            
            portfolio['cash'] += sell_amount - commission
            return_rate = (sell_price - info['avg_price']) / info['avg_price']
            
            trade = {
                'date': datetime.now().strftime('%Y-%m-%d %H:%M'),
                'symbol': symbol,
                'action': 'SELL',
                'shares': info['shares'],
                'price': round(sell_price, 2),
                'amount': round(sell_amount, 2),
                'commission': round(commission, 2),
                'return_rate': round(return_rate * 100, 2)
            }
            trades.append(trade)
            
            print(f"  📤 매도: {symbol} | {info['shares']}주 | ${sell_price:.2f} | 수익률: {return_rate*100:+.2f}%")
            
            del portfolio['holdings'][symbol]
    except Exception as e:
        print(f"  ❌ {symbol} 매도 실패: {e}")

# ----- 목표 비중 계산 -----
target_allocations = {}
for i, symbol in enumerate(signal['picks']):
    if i < len(signal['allocations']):
        target_allocations[symbol] = signal['allocations'][i]

# ----- 신규 매수 -----
for symbol in to_buy:
    if symbol not in target_allocations:
        continue
    
    try:
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period='1d')
        
        if not hist.empty:
            current_price = hist['Close'].iloc[-1]
            buy_price = current_price * (1 + SLIPPAGE)
            
            allocation = target_allocations[symbol]
            invest_amount = total_value * allocation
            shares = int(invest_amount / buy_price)
            
            if shares <= 0:
                continue
            
            buy_amount = shares * buy_price
            commission = buy_amount * BUY_COMMISSION
            
            if portfolio['cash'] >= buy_amount + commission:
                portfolio['cash'] -= (buy_amount + commission)
                
                portfolio['holdings'][symbol] = {
                    'shares': shares,
                    'avg_price': round(buy_price, 2)
                }
                
                trade = {
                    'date': datetime.now().strftime('%Y-%m-%d %H:%M'),
                    'symbol': symbol,
                    'action': 'BUY',
                    'shares': shares,
                    'price': round(buy_price, 2),
                    'amount': round(buy_amount, 2),
                    'commission': round(commission, 2),
                    'return_rate': 0
                }
                trades.append(trade)
                
                print(f"  📥 매수: {symbol} | {shares}주 | ${buy_price:.2f} | 비중: {allocation*100:.0f}%")
            else:
                print(f"  ⚠️ {symbol} 현금 부족: 필요 ${buy_amount+commission:.2f}, 보유 ${portfolio['cash']:.2f}")
    except Exception as e:
        print(f"  ❌ {symbol} 매수 실패: {e}")

# ----- 유지 종목 비중 조절 -----
for symbol in to_keep:
    if symbol not in portfolio['holdings'] or symbol not in target_allocations:
        continue
    
    info = portfolio['holdings'][symbol]
    
    try:
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period='1d')
        
        if hist.empty:
            continue
        
        current_price = hist['Close'].iloc[-1]
        current_value = info['shares'] * current_price
        target_value = total_value * target_allocations[symbol]
        
        diff_value = target_value - current_value
        diff_shares = int(abs(diff_value) / current_price)
        
        # 비중 차이가 5% 이상일 때만 조절
        if abs(diff_value) / total_value > 0.05 and diff_shares > 0:
            
            if diff_value > 0:
                # 추가 매수
                buy_price = current_price * (1 + SLIPPAGE)
                buy_amount = diff_shares * buy_price
                commission = buy_amount * BUY_COMMISSION
                
                if portfolio['cash'] >= buy_amount + commission:
                    portfolio['cash'] -= (buy_amount + commission)
                    
                    old_shares = info['shares']
                    old_avg = info['avg_price']
                    new_shares = old_shares + diff_shares
                    new_avg = (old_avg * old_shares + buy_amount) / new_shares
                    
                    portfolio['holdings'][symbol] = {
                        'shares': new_shares,
                        'avg_price': round(new_avg, 2)
                    }
                    
                    trade = {
                        'date': datetime.now().strftime('%Y-%m-%d %H:%M'),
                        'symbol': symbol,
                        'action': 'ADD',
                        'shares': diff_shares,
                        'price': round(buy_price, 2),
                        'amount': round(buy_amount, 2),
                        'commission': round(commission, 2),
                        'return_rate': 0
                    }
                    trades.append(trade)
                    
                    print(f"  📥 추가매수: {symbol} | +{diff_shares}주 | ${buy_price:.2f}")
            
            else:
                # 일부 매도
                sell_price = current_price * (1 - SLIPPAGE)
                sell_amount = diff_shares * sell_price
                commission = sell_amount * SELL_COMMISSION
                
                portfolio['cash'] += sell_amount - commission
                portfolio['holdings'][symbol]['shares'] -= diff_shares
                
                trade = {
                    'date': datetime.now().strftime('%Y-%m-%d %H:%M'),
                    'symbol': symbol,
                    'action': 'REDUCE',
                    'shares': diff_shares,
                    'price': round(sell_price, 2),
                    'amount': round(sell_amount, 2),
                    'commission': round(commission, 2),
                    'return_rate': 0
                }
                trades.append(trade)
                
                print(f"  📤 일부매도: {symbol} | -{diff_shares}주 | ${sell_price:.2f}")
        else:
            print(f"  ⏸️ 유지: {symbol} | {info['shares']}주")
    
    except Exception as e:
        print(f"  ❌ {symbol} 비중 조절 실패: {e}")

# ----- 저장 -----
save_portfolio(portfolio)
save_trades(trades)

print("\n" + "-" * 60)
print(f"✅ 거래 완료: {len(trades)}건")
print("-" * 60)

# 업데이트된 포트폴리오 출력
print_portfolio(portfolio)

return trades
```

# ============================================

# [5] 손절 체크

# ============================================

def check_stop_loss(portfolio=None):
“””
손절 조건 체크 및 실행

```
Args:
    portfolio: 포트폴리오 (없으면 파일에서 로드)

Returns:
    list: 손절된 종목 리스트
"""
if portfolio is None:
    portfolio = load_portfolio()

trades = []

print("=" * 60)
print("🛡️ 손절 체크")
print("=" * 60)

for symbol, info in list(portfolio['holdings'].items()):
    try:
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period='1d')
        
        if hist.empty:
            continue
        
        current_price = hist['Close'].iloc[-1]
        return_rate = (current_price - info['avg_price']) / info['avg_price']
        
        print(f"  {symbol}: 평단가 ${info['avg_price']:.2f} | 현재가 ${current_price:.2f} | 수익률 {return_rate*100:+.2f}%")
        
        if return_rate <= STOP_LOSS:
            # 손절 실행
            sell_price = current_price * (1 - SLIPPAGE)
            sell_amount = info['shares'] * sell_price
            commission = sell_amount * SELL_COMMISSION
            
            portfolio['cash'] += sell_amount - commission
            
            trade = {
                'date': datetime.now().strftime('%Y-%m-%d %H:%M'),
                'symbol': symbol,
                'action': 'STOP_LOSS',
                'shares': info['shares'],
                'price': round(sell_price, 2),
                'amount': round(sell_amount, 2),
                'commission': round(commission, 2),
                'return_rate': round(return_rate * 100, 2)
            }
            trades.append(trade)
            
            print(f"  🚨 손절 실행: {symbol} | {return_rate*100:.2f}% <= {STOP_LOSS*100:.0f}%")
            
            del portfolio['holdings'][symbol]
    
    except Exception as e:
        print(f"  ❌ {symbol} 체크 실패: {e}")

if trades:
    save_portfolio(portfolio)
    save_trades(trades)
    print(f"\n⚠️ {len(trades)}개 종목 손절 완료")
else:
    print("\n✅ 손절 대상 없음")

return trades
```

# ============================================

# [6] 거래 기록 저장

# ============================================

def save_trades(trades):
“””
거래 내역을 CSV에 추가

```
Args:
    trades: 거래 내역 리스트
"""
init_data_dir()

if not trades:
    return

trades_df = pd.DataFrame(trades)

if os.path.exists(TRADES_FILE):
    existing = pd.read_csv(TRADES_FILE)
    trades_df = pd.concat([existing, trades_df], ignore_index=True)

trades_df.to_csv(TRADES_FILE, index=False)
print(f"💾 거래 기록 저장: {TRADES_FILE}")
```

def save_signal(signal):
“””
신호를 CSV에 저장

```
Args:
    signal: get_today_signal() 반환값
"""
init_data_dir()

record = {
    'date': signal['date'].strftime('%Y-%m-%d') if hasattr(signal['date'], 'strftime') else str(signal['date']),
    'signal': signal['signal'],
    'picks': ','.join(signal.get('picks', [])),
    'scores': ','.join([f"{s:.4f}" for s in signal.get('scores', [])]),
    'allocations': ','.join([f"{a:.2f}" for a in signal.get('allocations', [])]),
    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
}

signals_df = pd.DataFrame([record])

if os.path.exists(SIGNALS_FILE):
    existing = pd.read_csv(SIGNALS_FILE)
    signals_df = pd.concat([existing, signals_df], ignore_index=True)

signals_df.to_csv(SIGNALS_FILE, index=False)
print(f"💾 신호 기록 저장: {SIGNALS_FILE}")
```

def load_trades():
“”“저장된 거래 내역 불러오기”””
if os.path.exists(TRADES_FILE):
return pd.read_csv(TRADES_FILE)
return pd.DataFrame()

def print_trade_history():
“”“거래 내역 출력”””
trades_df = load_trades()

```
print("=" * 60)
print("📋 거래 내역")
print("=" * 60)

if trades_df.empty:
    print("거래 내역 없음")
    return

print(f"총 {len(trades_df)}건")
print("-" * 60)
print(trades_df.to_string())
print("-" * 60)
```

# ============================================

# [7] 포트폴리오 초기화

# ============================================

def reset_portfolio():
“””
포트폴리오 초기화 (처음부터 다시 시작)
“””
init_data_dir()

```
# 확인
confirm = input("⚠️ 정말 초기화하시겠습니까? 모든 데이터가 삭제됩니다. (y/N): ")

if confirm.lower() != 'y':
    print("취소됨")
    return

# 파일 삭제
for f in [PORTFOLIO_FILE, TRADES_FILE, SIGNALS_FILE]:
    if os.path.exists(f):
        os.remove(f)
        print(f"🗑️ 삭제: {f}")

# 새 포트폴리오 생성
portfolio = load_portfolio()
print(f"\n✅ 초기화 완료! 초기 자본금: ${INITIAL_CAPITAL:,}")
```

# ============================================

# [8] 백테스트 결과와 비교

# ============================================

def compare_with_backtest(backtest_result):
“””
Paper Trading 결과와 백테스트 결과 비교

```
Args:
    backtest_result: run_backtest() 반환값
"""
portfolio = load_portfolio()
port_value = get_portfolio_value(portfolio)

# Paper Trading 수익률
pt_return = (port_value['total'] - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100

# 백테스트 수익률 (같은 기간)
bt_metrics = backtest_result['metrics']
bt_return = bt_metrics['total_return'] * 100

print("=" * 60)
print("📊 백테스트 vs Paper Trading 비교")
print("=" * 60)

print(f"\n백테스트 (5년 전체)")
print(f"  총 수익률: {bt_return:.2f}%")
print(f"  MDD: {bt_metrics['mdd']*100:.2f}%")
print(f"  샤프: {bt_metrics['sharpe_ratio']:.2f}")

print(f"\nPaper Trading (진행 중)")
print(f"  총 수익률: {pt_return:.2f}%")
print(f"  현재 자산: ${port_value['total']:,.2f}")
print(f"  시작일: {portfolio['created_at']}")

print("=" * 60)
```

# ============================================

# [9] 메인 실행

# ============================================

def main():
“””
메인 함수 - CLI 명령어 처리

```
사용법:
    python paper_trading.py signal     # 오늘 신호 확인
    python paper_trading.py portfolio  # 포트폴리오 상태
    python paper_trading.py execute    # 신호대로 가상 매매 실행
    python paper_trading.py stoploss   # 손절 체크
    python paper_trading.py history    # 거래 내역
    python paper_trading.py reset      # 초기화
"""
import sys

if len(sys.argv) < 2:
    print("사용법: python paper_trading.py [signal|portfolio|execute|stoploss|history|reset]")
    return

command = sys.argv[1].lower()

if command == 'signal':
    signal = get_today_signal()
    save_signal(signal)

elif command == 'portfolio':
    print_portfolio()

elif command == 'execute':
    signal = get_today_signal()
    if signal['signal'] == 'BUY':
        execute_signal(signal)
    save_signal(signal)

elif command == 'stoploss':
    check_stop_loss()

elif command == 'history':
    print_trade_history()

elif command == 'reset':
    reset_portfolio()

else:
    print(f"알 수 없는 명령어: {command}")
    print("사용법: python paper_trading.py [signal|portfolio|execute|stoploss|history|reset]")
```

if **name** == “**main**”:
main()