# ============================================
# src/paper_trading.py
# Paper Trading System (with Google Sheets)
# ============================================

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta

from src.strategy import (
    CustomStrategy,
    prepare_price_data,
    filter_tuesday,
    create_trade_mapping
)
from src.sheets import SheetsManager

# ============================================
# Settings
# ============================================

INITIAL_CAPITAL = 2000
BUY_COMMISSION = 0.0025
SELL_COMMISSION = 0.0025
SLIPPAGE = 0.001
STOP_LOSS = -0.07
LOOKBACK_DAYS = 200

# ============================================
# Data Download
# ============================================

def get_sp500_list():
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    headers = {'User-Agent': 'Mozilla/5.0'}
    response = requests.get(url, headers=headers)
    #tables = pd.read_html(url)
    table = pd.read_html(StringIO(response.text))
    
    df = tables[0]
    df = df[["Symbol", "Security", "GICS Sector"]].copy()
    df.columns = ["symbol", "company", "sector"]
    df["symbol"] = df["symbol"].str.replace(".", "-", regex=False)
    return df
    

def download_recent_data(symbols, days=LOOKBACK_DAYS):
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    print(f"Downloading {len(symbols)} symbols...")
    data = yf.download(symbols, start=start_date.strftime("%Y-%m-%d"), end=end_date.strftime("%Y-%m-%d"), auto_adjust=True, threads=True, progress=False)
    if data.empty:
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
                df = pd.DataFrame({"date": data.index, "open": data["Open"][symbol].values, "high": data["High"][symbol].values, "low": data["Low"][symbol].values, "close": data["Close"][symbol].values, "volume": data["Volume"][symbol].values, "symbol": symbol})
                df = df.dropna(subset=["close"])
                if not df.empty:
                    result.append(df)
            except:
                continue
    if result:
        final_df = pd.concat(result, ignore_index=True)
        final_df["date"] = pd.to_datetime(final_df["date"])
        print(f"Downloaded {final_df['symbol'].nunique()} symbols")
        return final_df
    return pd.DataFrame()


def get_current_prices(symbols):
    prices = {}
    for symbol in symbols:
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="1d")
            if not hist.empty:
                prices[symbol] = round(hist["Close"].iloc[-1], 2)
        except:
            pass
    return prices


def get_spy_price():
    try:
        ticker = yf.Ticker("SPY")
        hist = ticker.history(period="1d")
        if not hist.empty:
            return round(hist["Close"].iloc[-1], 2)
    except:
        pass
    return 0


# ============================================
# Signal Generation
# ============================================

def get_today_signal(strategy=None):
    if strategy is None:
        strategy = CustomStrategy()
    
    print("=" * 60)
    print(f"Signal Generation")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 60)
    
    # Download data
    sp500 = get_sp500_list()
    symbols = sp500["symbol"].tolist()
    if "SPY" not in symbols:
        symbols.append("SPY")
    
    df = download_recent_data(symbols)
    if df.empty:
        return {"signal": "ERROR", "message": "Download failed"}
    
    # Prepare data
    price_df = prepare_price_data(df)
    tuesday_df = filter_tuesday(price_df)
    if "SPY" in tuesday_df.columns:
        tuesday_df = tuesday_df.dropna(subset=["SPY"])
    if tuesday_df.empty:
        return {"signal": "ERROR", "message": "No Tuesday data"}
    
    # Calculate scores
    score_df, correlation_df, ret_1m = strategy.prepare(price_df, tuesday_df)
    score_dates = score_df.dropna(how="all").index.tolist()
    if not score_dates:
        return {"signal": "ERROR", "message": "Cannot calculate scores"}
    
    # Find latest Tuesday
    target_ts = pd.Timestamp(datetime.now())
    valid_dates = [d for d in score_dates if d <= target_ts]
    if not valid_dates:
        return {"signal": "ERROR", "message": "No valid Tuesday"}
    
    last_tuesday = valid_dates[-1]
    print(f"\nAnalysis date: {last_tuesday.strftime('%Y-%m-%d')} (Tuesday)")
    
    # Market momentum
    market_momentum = 0
    if last_tuesday in ret_1m.index:
        market_momentum = ret_1m.loc[last_tuesday].mean()
    
    # Select stocks
    result = strategy.select_stocks(score_df, correlation_df, last_tuesday, ret_1m)
    
    # SPY price
    spy_price = get_spy_price()
    
    if result is None:
        print("\nMarket downtrend - HOLD")
        return {
            "date": last_tuesday,
            "signal": "HOLD",
            "message": "Market momentum <= 0",
            "picks": [],
            "scores": [],
            "allocations": [],
            "prices": {},
            "market_momentum": market_momentum,
            "spy_price": spy_price,
            "market_trend": "DOWN"
        }
    
    # Get current prices
    picks = result["picks"]
    prices = get_current_prices(picks)
    
    # Get sectors
    sector_map = dict(zip(sp500["symbol"], sp500["sector"]))
    
    print(f"\nBUY Signal: {len(picks)} stocks")
    print("-" * 50)
    for i, (symbol, score, alloc) in enumerate(zip(picks, result["scores"], result["allocations"])):
        price = prices.get(symbol, "N/A")
        sector = sector_map.get(symbol, "")
        price_str = f"${price:.2f}" if isinstance(price, float) else price
        print(f"  {i+1}. {symbol:5} | {sector[:15]:15} | Score: {score:.4f} | {alloc*100:.0f}% | {price_str}")
    print("-" * 50)
    print(f"SPY: ${spy_price} | Market Momentum: {market_momentum:.4f}")
    
    return {
        "date": last_tuesday,
        "signal": "BUY",
        "picks": picks,
        "scores": result["scores"],
        "allocations": result["allocations"],
        "prices": prices,
        "sectors": {s: sector_map.get(s, "") for s in picks},
        "market_momentum": market_momentum,
        "spy_price": spy_price,
        "market_trend": "UP"
    }


# ============================================
# Portfolio Management
# ============================================

def get_portfolio_value(portfolio, current_prices=None):
    if current_prices is None:
        symbols = list(portfolio.get("holdings", {}).keys())
        if symbols:
            current_prices = get_current_prices(symbols)
        else:
            current_prices = {}
    
    cash = portfolio.get("cash", 0)
    stocks_value = 0
    holdings_detail = []
    
    for symbol, info in portfolio.get("holdings", {}).items():
        current_price = current_prices.get(symbol, 0)
        if current_price > 0:
            value = info["shares"] * current_price
            return_pct = (current_price - info["avg_price"]) / info["avg_price"] * 100
            stocks_value += value
            holdings_detail.append({
                "symbol": symbol,
                "shares": info["shares"],
                "avg_price": info["avg_price"],
                "current_price": current_price,
                "value": round(value, 2),
                "return_pct": round(return_pct, 2)
            })
    
    return {
        "total": round(cash + stocks_value, 2),
        "cash": round(cash, 2),
        "stocks": round(stocks_value, 2),
        "holdings_detail": holdings_detail
    }


def print_portfolio(sheets):
    portfolio = sheets.load_portfolio()
    value = get_portfolio_value(portfolio)
    
    print("=" * 60)
    print("Portfolio Status")
    print("=" * 60)
    print(f"\nTotal: ${value['total']:,.2f}")
    print(f"Cash: ${value['cash']:,.2f}")
    print(f"Stocks: ${value['stocks']:,.2f}")
    
    total_return = (value["total"] - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100
    print(f"\nTotal Return: {total_return:+.2f}%")
    
    if value["holdings_detail"]:
        print(f"\nHoldings ({len(value['holdings_detail'])})")
        print("-" * 60)
        for h in value["holdings_detail"]:
            print(f"  {h['symbol']:5} | {h['shares']:>4} shares | Avg: ${h['avg_price']:>7.2f} | Now: ${h['current_price']:>7.2f} | {h['return_pct']:>+6.2f}%")
        print("-" * 60)
    else:
        print("\nNo holdings")
    print("=" * 60)


# ============================================
# Trade Execution
# ============================================

def execute_signal(signal, sheets):
    portfolio = sheets.load_portfolio()
    
    if signal["signal"] != "BUY":
        print("No BUY signal - nothing to execute")
        return []
    
    trades = []
    new_picks = set(signal["picks"])
    current_holdings = set(portfolio["holdings"].keys())
    
    to_sell = current_holdings - new_picks
    to_buy = new_picks - current_holdings
    to_keep = current_holdings & new_picks
    
    print("\n" + "=" * 60)
    print("Executing Trades")
    print("=" * 60)
    
    # Current portfolio value
    port_value = get_portfolio_value(portfolio, signal.get("prices", {}))
    total_value = port_value["total"]
    
    # Target allocations
    target_allocations = {}
    for i, symbol in enumerate(signal["picks"]):
        if i < len(signal["allocations"]):
            target_allocations[symbol] = signal["allocations"][i]
    
    # SELL
    for symbol in to_sell:
        if symbol not in portfolio["holdings"]:
            continue
        info = portfolio["holdings"][symbol]
        current_price = signal.get("prices", {}).get(symbol)
        if not current_price:
            prices = get_current_prices([symbol])
            current_price = prices.get(symbol, 0)
        if current_price > 0:
            sell_price = current_price * (1 - SLIPPAGE)
            sell_amount = info["shares"] * sell_price
            commission = sell_amount * SELL_COMMISSION
            portfolio["cash"] += sell_amount - commission
            return_pct = (sell_price - info["avg_price"]) / info["avg_price"] * 100
            trade = {
                "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
                "symbol": symbol,
                "action": "SELL",
                "shares": info["shares"],
                "price": round(sell_price, 2),
                "amount": round(sell_amount, 2),
                "commission": round(commission, 2),
                "slippage": round(current_price * SLIPPAGE * info["shares"], 2),
                "return_pct": round(return_pct, 2),
                "sector": info.get("sector", ""),
                "score": "",
                "memo": "Position closed"
            }
            trades.append(trade)
            print(f"  SELL: {symbol} | {info['shares']} shares | ${sell_price:.2f} | Return: {return_pct:+.2f}%")
            del portfolio["holdings"][symbol]
    
    # BUY new
    for symbol in to_buy:
        if symbol not in target_allocations:
            continue
        current_price = signal.get("prices", {}).get(symbol)
        if not current_price:
            prices = get_current_prices([symbol])
            current_price = prices.get(symbol, 0)
        if current_price > 0:
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
                score_idx = signal["picks"].index(symbol) if symbol in signal["picks"] else -1
                score = signal["scores"][score_idx] if 0 <= score_idx < len(signal["scores"]) else 0
                portfolio["holdings"][symbol] = {
                    "shares": shares,
                    "avg_price": round(buy_price, 2),
                    "sector": signal.get("sectors", {}).get(symbol, ""),
                    "buy_date": datetime.now().strftime("%Y-%m-%d")
                }
                trade = {
                    "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
                    "symbol": symbol,
                    "action": "BUY",
                    "shares": shares,
                    "price": round(buy_price, 2),
                    "amount": round(buy_amount, 2),
                    "commission": round(commission, 2),
                    "slippage": round(current_price * SLIPPAGE * shares, 2),
                    "return_pct": 0,
                    "sector": signal.get("sectors", {}).get(symbol, ""),
                    "score": round(score, 4),
                    "memo": f"Weight: {allocation*100:.0f}%"
                }
                trades.append(trade)
                print(f"  BUY: {symbol} | {shares} shares | ${buy_price:.2f} | Weight: {allocation*100:.0f}%")
            else:
                print(f"  SKIP: {symbol} - Insufficient cash")
    
    # ADJUST existing
    for symbol in to_keep:
        if symbol not in portfolio["holdings"] or symbol not in target_allocations:
            continue
        info = portfolio["holdings"][symbol]
        current_price = signal.get("prices", {}).get(symbol)
        if not current_price:
            prices = get_current_prices([symbol])
            current_price = prices.get(symbol, 0)
        if current_price <= 0:
            continue
        current_value = info["shares"] * current_price
        target_value = total_value * target_allocations[symbol]
        diff_value = target_value - current_value
        diff_shares = int(abs(diff_value) / current_price)
        score_idx = signal["picks"].index(symbol) if symbol in signal["picks"] else -1
        score = signal["scores"][score_idx] if 0 <= score_idx < len(signal["scores"]) else 0
        if abs(diff_value) / total_value > 0.05 and diff_shares > 0:
            if diff_value > 0:
                # ADD
                buy_price = current_price * (1 + SLIPPAGE)
                buy_amount = diff_shares * buy_price
                commission = buy_amount * BUY_COMMISSION
                if portfolio["cash"] >= buy_amount + commission:
                    portfolio["cash"] -= (buy_amount + commission)
                    old_shares = info["shares"]
                    old_avg = info["avg_price"]
                    new_shares = old_shares + diff_shares
                    new_avg = (old_avg * old_shares + buy_amount) / new_shares
                    portfolio["holdings"][symbol]["shares"] = new_shares
                    portfolio["holdings"][symbol]["avg_price"] = round(new_avg, 2)
                    trade = {
                        "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
                        "symbol": symbol,
                        "action": "ADD",
                        "shares": diff_shares,
                        "price": round(buy_price, 2),
                        "amount": round(buy_amount, 2),
                        "commission": round(commission, 2),
                        "slippage": round(current_price * SLIPPAGE * diff_shares, 2),
                        "return_pct": 0,
                        "sector": info.get("sector", ""),
                        "score": round(score, 4),
                        "memo": "Position increased"
                    }
                    trades.append(trade)
                    print(f"  ADD: {symbol} | +{diff_shares} shares | ${buy_price:.2f}")
            else:
                # REDUCE
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
                    "slippage": round(current_price * SLIPPAGE * diff_shares, 2),
                    "return_pct": 0,
                    "sector": info.get("sector", ""),
                    "score": round(score, 4),
                    "memo": "Position reduced"
                }
                trades.append(trade)
                print(f"  REDUCE: {symbol} | -{diff_shares} shares | ${sell_price:.2f}")
        else:
            print(f"  HOLD: {symbol} | {info['shares']} shares")
    
    # Save
    current_prices = signal.get("prices", {})
    sheets.save_portfolio(portfolio, current_prices)
    sheets.save_trades(trades)
    
    print(f"\nCompleted: {len(trades)} trades")
    print_portfolio(sheets)
    
    return trades


# ============================================
# Stop Loss
# ============================================

def check_stop_loss(sheets):
    portfolio = sheets.load_portfolio()
    trades = []
    
    print("=" * 60)
    print("Stop Loss Check")
    print("=" * 60)
    
    symbols = list(portfolio["holdings"].keys())
    if not symbols:
        print("No holdings")
        return []
    
    current_prices = get_current_prices(symbols)
    
    for symbol, info in list(portfolio["holdings"].items()):
        current_price = current_prices.get(symbol, 0)
        if current_price <= 0:
            continue
        return_pct = (current_price - info["avg_price"]) / info["avg_price"]
        print(f"  {symbol}: Avg ${info['avg_price']:.2f} | Now ${current_price:.2f} | {return_pct*100:+.2f}%")
        
        if return_pct <= STOP_LOSS:
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
                "slippage": round(current_price * SLIPPAGE * info["shares"], 2),
                "return_pct": round(return_pct * 100, 2),
                "sector": info.get("sector", ""),
                "score": "",
                "memo": f"Stop loss at {STOP_LOSS*100:.0f}%"
            }
            trades.append(trade)
            print(f"  >>> STOP LOSS: {symbol} | {return_pct*100:.2f}%")
            del portfolio["holdings"][symbol]
    
    if trades:
        sheets.save_portfolio(portfolio)
        sheets.save_trades(trades)
        print(f"\n{len(trades)} stop loss executed")
    else:
        print("\nNo stop loss needed")
    
    return trades


# ============================================
# Daily Update
# ============================================

def record_daily_value(sheets):
    portfolio = sheets.load_portfolio()
    port_value = get_portfolio_value(portfolio)
    spy_price = get_spy_price()
    
    # Load previous daily value for return calculation
    daily_df = sheets.load_daily_values()
    prev_value = INITIAL_CAPITAL
    prev_spy = spy_price
    if len(daily_df) > 0:
        try:
            prev_value = float(daily_df.iloc[-1]["Total_Value"])
            prev_spy = float(daily_df.iloc[-1]["SPY_Price"])
        except:
            pass
    
    daily_return = (port_value["total"] - prev_value) / prev_value * 100 if prev_value > 0 else 0
    spy_return = (spy_price - prev_spy) / prev_spy * 100 if prev_spy > 0 else 0
    alpha = daily_return - spy_return
    
    daily_data = {
        "date": datetime.now().strftime("%Y-%m-%d"),
        "total_value": port_value["total"],
        "cash": port_value["cash"],
        "stocks_value": port_value["stocks"],
        "daily_return_pct": round(daily_return, 2),
        "spy_price": spy_price,
        "spy_return_pct": round(spy_return, 2),
        "alpha": round(alpha, 2)
    }
    
    sheets.save_daily_value(daily_data)
    print(f"Daily value recorded: ${port_value['total']:,.2f} ({daily_return:+.2f}%)")
    
    return daily_data


# ============================================
# Performance Update
# ============================================

def update_performance(sheets):
    portfolio = sheets.load_portfolio()
    port_value = get_portfolio_value(portfolio)
    trades_df = sheets.load_trades()
    daily_df = sheets.load_daily_values()
    
    # Calculate metrics
    total_return = (port_value["total"] - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100
    
    # Days
    start_date = portfolio.get("created_at", "")[:10]
    if start_date:
        try:
            days = (datetime.now() - datetime.strptime(start_date, "%Y-%m-%d")).days
        except:
            days = 0
    else:
        days = 0
    
    # CAGR
    years = days / 365 if days > 0 else 0
    cagr = ((port_value["total"] / INITIAL_CAPITAL) ** (1/years) - 1) * 100 if years > 0 else 0
    
    # SPY return
    spy_return = 0
    if len(daily_df) > 1:
        try:
            first_spy = float(daily_df.iloc[0]["SPY_Price"])
            last_spy = float(daily_df.iloc[-1]["SPY_Price"])
            spy_return = (last_spy - first_spy) / first_spy * 100
        except:
            pass
    
    # MDD
    mdd = 0
    if len(daily_df) > 0:
        try:
            values = daily_df["Total_Value"].astype(float)
            peak = values.cummax()
            drawdown = (values - peak) / peak * 100
            mdd = drawdown.min()
        except:
            pass
    
    # Win rate
    win_rate = 0
    total_trades = len(trades_df)
    if total_trades > 0:
        try:
            returns = trades_df["Return%"].astype(float)
            wins = (returns > 0).sum()
            win_rate = wins / total_trades * 100
        except:
            pass
    
    # Sharpe (simplified)
    sharpe = 0
    if len(daily_df) > 1:
        try:
            daily_returns = daily_df["Daily_Return%"].astype(float)
            if daily_returns.std() > 0:
                sharpe = (daily_returns.mean() * 252) / (daily_returns.std() * np.sqrt(252))
        except:
            pass
    
    metrics = {
        "initial_capital": INITIAL_CAPITAL,
        "current_value": port_value["total"],
        "total_return_pct": total_return,
        "cagr": cagr,
        "spy_return_pct": spy_return,
        "alpha": total_return - spy_return,
        "mdd": mdd,
        "sharpe_ratio": sharpe,
        "win_rate": win_rate,
        "total_trades": total_trades,
        "start_date": start_date,
        "days": days
    }
    
    sheets.save_performance(metrics)
    
    print("=" * 60)
    print("Performance Updated")
    print("=" * 60)
    print(f"Total Return: {total_return:+.2f}%")
    print(f"SPY Return: {spy_return:+.2f}%")
    print(f"Alpha: {total_return - spy_return:+.2f}%")
    print(f"MDD: {mdd:.2f}%")
    print(f"Win Rate: {win_rate:.1f}%")
    print("=" * 60)
    
    return metrics


# ============================================
# Main Functions
# ============================================

def run_daily(sheets=None):
    """Daily routine: check stop loss, record value"""
    if sheets is None:
        sheets = SheetsManager()
    
    print("\n" + "=" * 60)
    print(f"Daily Run: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 60)
    
    # Check stop loss
    check_stop_loss(sheets)
    
    # Record daily value
    record_daily_value(sheets)
    
    # Update performance
    update_performance(sheets)


def run_weekly(sheets=None):
    """Weekly routine: generate signal, execute trades"""
    if sheets is None:
        sheets = SheetsManager()
    
    print("\n" + "=" * 60)
    print(f"Weekly Run: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 60)
    
    # Generate signal
    signal = get_today_signal()
    
    # Save signal
    sheets.save_signal(signal)
    
    # Execute if BUY
    if signal["signal"] == "BUY":
        execute_signal(signal, sheets)
    
    # Record daily value
    record_daily_value(sheets)
    
    # Update performance
    update_performance(sheets)


# ============================================
# Test
# ============================================

if __name__ == "__main__":
    import sys
    
    sheets = SheetsManager()
    
    if len(sys.argv) < 2:
        print("Usage: python paper_trading.py [signal|portfolio|execute|stoploss|daily|weekly]")
        sys.exit(1)
    
    cmd = sys.argv[1].lower()
    
    if cmd == "signal":
        signal = get_today_signal()
        sheets.save_signal(signal)
    elif cmd == "portfolio":
        print_portfolio(sheets)
    elif cmd == "execute":
        signal = get_today_signal()
        sheets.save_signal(signal)
        if signal["signal"] == "BUY":
            execute_signal(signal, sheets)
    elif cmd == "stoploss":
        check_stop_loss(sheets)
    elif cmd == "daily":
        run_daily(sheets)
    elif cmd == "weekly":
        run_weekly(sheets)
    else:
        print(f"Unknown command: {cmd}")
