# ============================================

# src/paper_trading.py

# Paper Trading System

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

# Settings

# ============================================

INITIAL_CAPITAL = 2000
BUY_COMMISSION = 0.0025
SELL_COMMISSION = 0.0025
SLIPPAGE = 0.001
STOP_LOSS = -0.07
LOOKBACK_DAYS = 200

DATA_DIR = “paper_trading_data”
PORTFOLIO_FILE = DATA_DIR + “/portfolio.json”
TRADES_FILE = DATA_DIR + “/trades.csv”
SIGNALS_FILE = DATA_DIR + “/signals.csv”

# ============================================

# Data Download

# ============================================

def get_sp500_list():
“”“Get S&P 500 stock list from Wikipedia”””
url = “https://en.wikipedia.org/wiki/List_of_S%26P_500_companies”
tables = pd.read_html(url)
df = tables[0]
df = df[[“Symbol”, “Security”, “GICS Sector”]].copy()
df.columns = [“symbol”, “company”, “sector”]
df[“symbol”] = df[“symbol”].str.replace(”.”, “-”, regex=False)
return df

def download_recent_data(symbols, days=LOOKBACK_DAYS):
“”“Download recent N days of data”””
end_date = datetime.now()
start_date = end_date - timedelta(days=days)

```
print(f"Downloading data...")
print(f"  Symbols: {len(symbols)}")
print(f"  Period: {start_date.strftime('%Y-%m-%d')} ~ {end_date.strftime('%Y-%m-%d')}")

data = yf.download(
    symbols,
    start=start_date.strftime("%Y-%m-%d"),
    end=end_date.strftime("%Y-%m-%d"),
    auto_adjust=True,
    threads=True,
    progress=False
)

if data.empty:
    print("Download failed")
    return pd.DataFrame()

result = []

if len(symbols) == 1:
    df = data.copy()
    df["symbol"] = symbols[0]
    df = df.reset_index()
    df.columns = ["date", "close", "high", "low", "open", "volume", "symbol"]
    result.append(df)
else:
    for symbol in symbols:
        try:
            if symbol not in data["Close"].columns:
                continue
            
            df = pd.DataFrame({
                "date": data.index,
                "open": data["Open"][symbol].values,
                "high": data["High"][symbol].values,
                "low": data["Low"][symbol].values,
                "close": data["Close"][symbol].values,
                "volume": data["Volume"][symbol].values,
                "symbol": symbol
            })
            
            df = df.dropna(subset=["close"])
            if not df.empty:
                result.append(df)
        except:
            continue

if result:
    final_df = pd.concat(result, ignore_index=True)
    final_df["date"] = pd.to_datetime(final_df["date"])
    print(f"Download complete! ({final_df['symbol'].nunique()} symbols)")
    return final_df

return pd.DataFrame()
```

# ============================================

# Signal Generation

# ============================================

def get_today_signal(strategy=None, target_date=None):
“”“Generate buy signal for today”””

```
if strategy is None:
    strategy = CustomStrategy()

if target_date is None:
    target_date = datetime.now()

print("=" * 60)
print(f"Generating Signal")
print(f"   Date: {target_date.strftime('%Y-%m-%d %H:%M')}")
print("=" * 60)

# Download data
sp500 = get_sp500_list()
symbols = sp500["symbol"].tolist()
if "SPY" not in symbols:
    symbols.append("SPY")

df = download_recent_data(symbols)

if df.empty:
    return {"signal": "ERROR", "message": "Download failed"}

# Prepare strategy data
price_df = prepare_price_data(df)
tuesday_df = filter_tuesday(price_df)

if "SPY" in tuesday_df.columns:
    tuesday_df = tuesday_df.dropna(subset=["SPY"])

if tuesday_df.empty:
    return {"signal": "ERROR", "message": "No Tuesday data"}

score_df, correlation_df, ret_1m = strategy.prepare(price_df, tuesday_df)

# Find latest Tuesday
score_dates = score_df.dropna(how="all").index.tolist()

if not score_dates:
    return {"signal": "ERROR", "message": "Cannot calculate scores"}

target_ts = pd.Timestamp(target_date)
valid_dates = [d for d in score_dates if d <= target_ts]

if not valid_dates:
    return {"signal": "ERROR", "message": "No valid Tuesday"}

last_tuesday = valid_dates[-1]

print(f"\nAnalysis date: {last_tuesday.strftime('%Y-%m-%d')} (Tuesday)")

# Select stocks
result = strategy.select_stocks(score_df, correlation_df, last_tuesday, ret_1m)

if result is None:
    print("\nMarket downtrend - No buy signal")
    return {
        "date": last_tuesday,
        "signal": "HOLD",
        "message": "Market momentum <= 0",
        "picks": [],
        "scores": [],
        "allocations": [],
        "prices": {}
    }

# Get current prices
picks = result["picks"]
prices = {}

for symbol in picks:
    try:
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period="1d")
        if not hist.empty:
            prices[symbol] = round(hist["Close"].iloc[-1], 2)
    except:
        prices[symbol] = None

# Print results
print(f"\nBuy Signal: {len(picks)} stocks")
print("-" * 50)

for i, (symbol, score, alloc) in enumerate(zip(picks, result["scores"], result["allocations"])):
    price = prices.get(symbol, "N/A")
    price_str = f"${price:.2f}" if isinstance(price, float) else price
    print(f"  {i+1}. {symbol:5} | Score: {score:.4f} | Weight: {alloc*100:.0f}% | Price: {price_str}")

print("-" * 50)

return {
    "date": last_tuesday,
    "signal": "BUY",
    "picks": picks,
    "scores": result["scores"],
    "allocations": result["allocations"],
    "prices": prices
}
```

# ============================================

# Portfolio Management

# ============================================

def init_data_dir():
“”“Create data directory”””
if not os.path.exists(DATA_DIR):
os.makedirs(DATA_DIR)
print(f”Created directory: {DATA_DIR}”)

def load_portfolio():
“”“Load saved portfolio”””
init_data_dir()

```
if os.path.exists(PORTFOLIO_FILE):
    with open(PORTFOLIO_FILE, "r") as f:
        return json.load(f)

portfolio = {
    "cash": INITIAL_CAPITAL,
    "holdings": {},
    "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
}

save_portfolio(portfolio)
return portfolio
```

def save_portfolio(portfolio):
“”“Save portfolio”””
init_data_dir()

```
portfolio["last_updated"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

with open(PORTFOLIO_FILE, "w") as f:
    json.dump(portfolio, f, indent=2)
```

def get_portfolio_value(portfolio):
“”“Calculate current portfolio value”””
cash = portfolio[“cash”]
holdings = portfolio[“holdings”]

```
stocks_value = 0
holdings_detail = []

for symbol, info in holdings.items():
    try:
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period="1d")
        
        if not hist.empty:
            current_price = hist["Close"].iloc[-1]
            value = info["shares"] * current_price
            return_rate = (current_price - info["avg_price"]) / info["avg_price"]
            
            stocks_value += value
            holdings_detail.append({
                "symbol": symbol,
                "shares": info["shares"],
                "avg_price": info["avg_price"],
                "current_price": round(current_price, 2),
                "value": round(value, 2),
                "return_rate": round(return_rate * 100, 2),
                "profit": round(value - info["shares"] * info["avg_price"], 2)
            })
    except:
        continue

return {
    "total": round(cash + stocks_value, 2),
    "cash": round(cash, 2),
    "stocks": round(stocks_value, 2),
    "holdings_detail": holdings_detail
}
```

def print_portfolio(portfolio=None):
“”“Print portfolio status”””
if portfolio is None:
portfolio = load_portfolio()

```
value = get_portfolio_value(portfolio)

print("=" * 60)
print("Portfolio Status")
print("=" * 60)

print(f"\nAsset Summary")
print(f"  Total: ${value['total']:,.2f}")
print(f"  Cash: ${value['cash']:,.2f}")
print(f"  Stocks: ${value['stocks']:,.2f}")

total_return = (value["total"] - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100
print(f"\nTotal Return: {total_return:+.2f}%")

if value["holdings_detail"]:
    print(f"\nHoldings ({len(value['holdings_detail'])})")
    print("-" * 60)
    print(f"  {'Symbol':5} | {'Qty':>5} | {'AvgPrc':>8} | {'CurPrc':>8} | {'Return':>8} | {'Value':>10}")
    print("-" * 60)
    
    for h in value["holdings_detail"]:
        print(f"  {h['symbol']:5} | {h['shares']:>5} | ${h['avg_price']:>7.2f} | ${h['current_price']:>7.2f} | {h['return_rate']:>+7.2f}% | ${h['value']:>9.2f}")
    
    print("-" * 60)
else:
    print("\nNo holdings")

print(f"\nCreated: {portfolio['created_at']}")
print(f"Updated: {portfolio['last_updated']}")
print("=" * 60)
```

# ============================================

# Execute Trades

# ============================================

def execute_signal(signal, portfolio=None):
“”“Execute virtual trades based on signal”””
if portfolio is None:
portfolio = load_portfolio()

```
if signal["signal"] != "BUY":
    print("No buy signal - nothing to execute")
    return []

trades = []
new_picks = set(signal["picks"])
current_holdings = set(portfolio["holdings"].keys())

to_sell = current_holdings - new_picks
to_buy = new_picks - current_holdings
to_keep = current_holdings & new_picks

print("\n" + "=" * 60)
print("Executing Virtual Trades")
print("=" * 60)

port_value = get_portfolio_value(portfolio)
total_value = port_value["total"]

# Sell
for symbol in to_sell:
    if symbol not in portfolio["holdings"]:
        continue
    
    info = portfolio["holdings"][symbol]
    
    try:
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period="1d")
        
        if not hist.empty:
            current_price = hist["Close"].iloc[-1]
            sell_price = current_price * (1 - SLIPPAGE)
            sell_amount = info["shares"] * sell_price
            commission = sell_amount * SELL_COMMISSION
            
            portfolio["cash"] += sell_amount - commission
            return_rate = (sell_price - info["avg_price"]) / info["avg_price"]
            
            trade = {
                "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
                "symbol": symbol,
                "action": "SELL",
                "shares": info["shares"],
                "price": round(sell_price, 2),
                "amount": round(sell_amount, 2),
                "commission": round(commission, 2),
                "return_rate": round(return_rate * 100, 2)
            }
            trades.append(trade)
            
            print(f"  SELL: {symbol} | {info['shares']} shares | ${sell_price:.2f} | Return: {return_rate*100:+.2f}%")
            
            del portfolio["holdings"][symbol]
    except Exception as e:
        print(f"  SELL failed {symbol}: {e}")

# Target allocations
target_allocations = {}
for i, symbol in enumerate(signal["picks"]):
    if i < len(signal["allocations"]):
        target_allocations[symbol] = signal["allocations"][i]

# Buy new
for symbol in to_buy:
    if symbol not in target_allocations:
        continue
    
    try:
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period="1d")
        
        if not hist.empty:
            current_price = hist["Close"].iloc[-1]
            buy_price = current_price * (1 + SLIPPAGE)
            
            allocation = target_allocations[symbol]
            invest_amount = total_value * allocation
            shares = int(invest_amount / buy_price)
            
            if shares <= 0:
                continue
            
            buy_amount = shares * buy_price
            commission = buy_amount * BUY_COMMISSION
            
            if portfolio["cash"] >= buy_amount + commission:
                portfolio["cash"] -= (buy_amount + commission)
                
                portfolio["holdings"][symbol] = {
                    "shares": shares,
                    "avg_price": round(buy_price, 2)
                }
                
                trade = {
                    "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
                    "symbol": symbol,
                    "action": "BUY",
                    "shares": shares,
                    "price": round(buy_price, 2),
                    "amount": round(buy_amount, 2),
                    "commission": round(commission, 2),
                    "return_rate": 0
                }
                trades.append(trade)
                
                print(f"  BUY: {symbol} | {shares} shares | ${buy_price:.2f} | Weight: {allocation*100:.0f}%")
            else:
                print(f"  Insufficient cash for {symbol}")
    except Exception as e:
        print(f"  BUY failed {symbol}: {e}")

# Adjust existing holdings
for symbol in to_keep:
    if symbol not in portfolio["holdings"] or symbol not in target_allocations:
        continue
    
    info = portfolio["holdings"][symbol]
    
    try:
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period="1d")
        
        if hist.empty:
            continue
        
        current_price = hist["Close"].iloc[-1]
        current_value = info["shares"] * current_price
        target_value = total_value * target_allocations[symbol]
        
        diff_value = target_value - current_value
        diff_shares = int(abs(diff_value) / current_price)
        
        if abs(diff_value) / total_value > 0.05 and diff_shares > 0:
            
            if diff_value > 0:
                buy_price = current_price * (1 + SLIPPAGE)
                buy_amount = diff_shares * buy_price
                commission = buy_amount * BUY_COMMISSION
                
                if portfolio["cash"] >= buy_amount + commission:
                    portfolio["cash"] -= (buy_amount + commission)
                    
                    old_shares = info["shares"]
                    old_avg = info["avg_price"]
                    new_shares = old_shares + diff_shares
                    new_avg = (old_avg * old_shares + buy_amount) / new_shares
                    
                    portfolio["holdings"][symbol] = {
                        "shares": new_shares,
                        "avg_price": round(new_avg, 2)
                    }
                    
                    trade = {
                        "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
                        "symbol": symbol,
                        "action": "ADD",
                        "shares": diff_shares,
                        "price": round(buy_price, 2),
                        "amount": round(buy_amount, 2),
                        "commission": round(commission, 2),
                        "return_rate": 0
                    }
                    trades.append(trade)
                    
                    print(f"  ADD: {symbol} | +{diff_shares} shares | ${buy_price:.2f}")
            
            else:
                sell_price = current_price * (1 - SLIPPAGE)
                sell_amount = diff_shares * sell_price
                commission = sell_amount * SELL_COMMISSION
                
                portfolio["cash"] += sell_amount - commission
                portfolio["holdings"][symbol]["shares"] -= diff_shares
                
                trade = {
                    "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
                    "symbol": symbol,
                    "action": "REDUCE",
                    "shares": diff_shares,
                    "price": round(sell_price, 2),
                    "amount": round(sell_amount, 2),
                    "commission": round(commission, 2),
                    "return_rate": 0
                }
                trades.append(trade)
                
                print(f"  REDUCE: {symbol} | -{diff_shares} shares | ${sell_price:.2f}")
        else:
            print(f"  HOLD: {symbol} | {info['shares']} shares")
    
    except Exception as e:
        print(f"  Adjust failed {symbol}: {e}")

save_portfolio(portfolio)
save_trades(trades)

print("\n" + "-" * 60)
print(f"Completed: {len(trades)} trades")
print("-" * 60)

print_portfolio(portfolio)

return trades
```

# ============================================

# Stop Loss

# ============================================

def check_stop_loss(portfolio=None):
“”“Check and execute stop loss”””
if portfolio is None:
portfolio = load_portfolio()

```
trades = []

print("=" * 60)
print("Stop Loss Check")
print("=" * 60)

for symbol, info in list(portfolio["holdings"].items()):
    try:
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period="1d")
        
        if hist.empty:
            continue
        
        current_price = hist["Close"].iloc[-1]
        return_rate = (current_price - info["avg_price"]) / info["avg_price"]
        
        print(f"  {symbol}: Avg ${info['avg_price']:.2f} | Current ${current_price:.2f} | Return {return_rate*100:+.2f}%")
        
        if return_rate <= STOP_LOSS:
            sell_price = current_price * (1 - SLIPPAGE)
            sell_amount = info["shares"] * sell_price
            commission = sell_amount * SELL_COMMISSION
            
            portfolio["cash"] += sell_amount - commission
            
            trade = {
                "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
                "symbol": symbol,
                "action": "STOP_LOSS",
                "shares": info["shares"],
                "price": round(sell_price, 2),
                "amount": round(sell_amount, 2),
                "commission": round(commission, 2),
                "return_rate": round(return_rate * 100, 2)
            }
            trades.append(trade)
            
            print(f"  STOP LOSS: {symbol} | {return_rate*100:.2f}% <= {STOP_LOSS*100:.0f}%")
            
            del portfolio["holdings"][symbol]
    
    except Exception as e:
        print(f"  Check failed {symbol}: {e}")

if trades:
    save_portfolio(portfolio)
    save_trades(trades)
    print(f"\n{len(trades)} stop loss executed")
else:
    print("\nNo stop loss needed")

return trades
```

# ============================================

# Trade History

# ============================================

def save_trades(trades):
“”“Save trades to CSV”””
init_data_dir()

```
if not trades:
    return

trades_df = pd.DataFrame(trades)

if os.path.exists(TRADES_FILE):
    existing = pd.read_csv(TRADES_FILE)
    trades_df = pd.concat([existing, trades_df], ignore_index=True)

trades_df.to_csv(TRADES_FILE, index=False)
print(f"Saved: {TRADES_FILE}")
```

def save_signal(signal):
“”“Save signal to CSV”””
init_data_dir()

```
record = {
    "date": signal["date"].strftime("%Y-%m-%d") if hasattr(signal["date"], "strftime") else str(signal["date"]),
    "signal": signal["signal"],
    "picks": ",".join(signal.get("picks", [])),
    "scores": ",".join([f"{s:.4f}" for s in signal.get("scores", [])]),
    "allocations": ",".join([f"{a:.2f}" for a in signal.get("allocations", [])]),
    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
}

signals_df = pd.DataFrame([record])

if os.path.exists(SIGNALS_FILE):
    existing = pd.read_csv(SIGNALS_FILE)
    signals_df = pd.concat([existing, signals_df], ignore_index=True)

signals_df.to_csv(SIGNALS_FILE, index=False)
print(f"Saved: {SIGNALS_FILE}")
```

def load_trades():
“”“Load trade history”””
if os.path.exists(TRADES_FILE):
return pd.read_csv(TRADES_FILE)
return pd.DataFrame()

def print_trade_history():
“”“Print trade history”””
trades_df = load_trades()

```
print("=" * 60)
print("Trade History")
print("=" * 60)

if trades_df.empty:
    print("No trades")
    return

print(f"Total: {len(trades_df)} trades")
print("-" * 60)
print(trades_df.to_string())
print("-" * 60)
```

# ============================================

# Reset

# ============================================

def reset_portfolio():
“”“Reset portfolio”””
init_data_dir()

```
confirm = input("Reset all data? (y/N): ")

if confirm.lower() != "y":
    print("Cancelled")
    return

for f in [PORTFOLIO_FILE, TRADES_FILE, SIGNALS_FILE]:
    if os.path.exists(f):
        os.remove(f)
        print(f"Deleted: {f}")

portfolio = load_portfolio()
print(f"\nReset complete! Initial capital: ${INITIAL_CAPITAL:,}")
```

# ============================================

# Compare with Backtest

# ============================================

def compare_with_backtest(backtest_result):
“”“Compare paper trading with backtest”””
portfolio = load_portfolio()
port_value = get_portfolio_value(portfolio)

```
pt_return = (port_value["total"] - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100

bt_metrics = backtest_result["metrics"]
bt_return = bt_metrics["total_return"] * 100

print("=" * 60)
print("Backtest vs Paper Trading")
print("=" * 60)

print(f"\nBacktest (5 years)")
print(f"  Total Return: {bt_return:.2f}%")
print(f"  MDD: {bt_metrics['mdd']*100:.2f}%")
print(f"  Sharpe: {bt_metrics['sharpe_ratio']:.2f}")

print(f"\nPaper Trading")
print(f"  Total Return: {pt_return:.2f}%")
print(f"  Current Value: ${port_value['total']:,.2f}")
print(f"  Started: {portfolio['created_at']}")

print("=" * 60)
```

# ============================================

# Main

# ============================================

def main():
“”“Main function”””
import sys

```
if len(sys.argv) < 2:
    print("Usage: python paper_trading.py [signal|portfolio|execute|stoploss|history|reset]")
    return

command = sys.argv[1].lower()

if command == "signal":
    signal = get_today_signal()
    save_signal(signal)

elif command == "portfolio":
    print_portfolio()

elif command == "execute":
    signal = get_today_signal()
    if signal["signal"] == "BUY":
        execute_signal(signal)
    save_signal(signal)

elif command == "stoploss":
    check_stop_loss()

elif command == "history":
    print_trade_history()

elif command == "reset":
    reset_portfolio()

else:
    print(f"Unknown command: {command}")
```

if **name** == “**main**”:
main()