# ============================================
# src/paper_trading.py
# Paper Trading System
# - 한국투자증권 모의투자 API 사용
# - Google Sheets 기록
# - Telegram 알림
# ============================================

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta

from src.config import (
    INITIAL_CAPITAL,
    STOP_LOSS,
    TOP_N,
    LOOKBACK_DAYS,
    SP500_BACKUP,
    KIS_MODE
)
from src.strategy import (
    CustomStrategy,
    prepare_price_data,
    filter_tuesday
)
from src.sheets import SheetsManager
from src.telegram import (
    send_signal,
    send_trades,
    send_portfolio,
    send_stop_loss,
    send_daily_summary,
    send_error
)
from src import kis


# ============================================
# Data Functions
# ============================================

def get_sp500_list():
    """
    S&P 500 종목 리스트 가져오기
    Wikipedia에서 실패하면 백업 리스트 사용
    """
    try:
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        tables = pd.read_html(url)
        df = tables[0]
        df = df[["Symbol", "Security", "GICS Sector"]].copy()
        df.columns = ["symbol", "company", "sector"]
        df["symbol"] = df["symbol"].str.replace(".", "-", regex=False)
        print(f"Loaded {len(df)} symbols from Wikipedia")
        return df
    except Exception as e:
        print(f"Wikipedia failed: {e}")
        print(f"Using backup list ({len(SP500_BACKUP)} symbols)")
        return pd.DataFrame({
            "symbol": SP500_BACKUP,
            "company": "",
            "sector": ""
        })


def download_recent_data(symbols, days=LOOKBACK_DAYS):
    """
    yfinance로 최근 데이터 다운로드 (전략 분석용)
    """
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    print(f"Downloading {len(symbols)} symbols...")
    
    data = yf.download(
        symbols,
        start=start_date.strftime("%Y-%m-%d"),
        end=end_date.strftime("%Y-%m-%d"),
        auto_adjust=True,
        threads=True,
        progress=False
    )
    
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
        print(f"Downloaded {final_df['symbol'].nunique()} symbols")
        return final_df
    return pd.DataFrame()


def get_spy_price():
    """
    SPY 현재가 조회 (벤치마크용)
    """
    try:
        price_data = kis.get_price("SPY", "NASD")
        if price_data:
            return price_data["price"]
    except:
        pass
    
    # Fallback: yfinance
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
    """
    오늘의 매매 신호 생성
    CustomStrategy 사용
    """
    if strategy is None:
        strategy = CustomStrategy()
    
    print("=" * 60)
    print(f"Signal Generation")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"Mode: {KIS_MODE}")
    print("=" * 60)
    
    # S&P 500 종목 가져오기
    sp500 = get_sp500_list()
    symbols = sp500["symbol"].tolist()
    if "SPY" not in symbols:
        symbols.append("SPY")
    
    # 데이터 다운로드 (전략 분석용)
    df = download_recent_data(symbols)
    if df.empty:
        return {"signal": "ERROR", "message": "Download failed"}
    
    # 전략 분석
    price_df = prepare_price_data(df)
    tuesday_df = filter_tuesday(price_df)
    if "SPY" in tuesday_df.columns:
        tuesday_df = tuesday_df.dropna(subset=["SPY"])
    if tuesday_df.empty:
        return {"signal": "ERROR", "message": "No Tuesday data"}
    
    score_df, correlation_df, ret_1m = strategy.prepare(price_df, tuesday_df)
    score_dates = score_df.dropna(how="all").index.tolist()
    if not score_dates:
        return {"signal": "ERROR", "message": "Cannot calculate scores"}
    
    # 가장 최근 화요일 찾기
    target_ts = pd.Timestamp(datetime.now())
    valid_dates = [d for d in score_dates if d <= target_ts]
    if not valid_dates:
        return {"signal": "ERROR", "message": "No valid Tuesday"}
    
    last_tuesday = valid_dates[-1]
    print(f"\nAnalysis date: {last_tuesday.strftime('%Y-%m-%d')} (Tuesday)")
    
    # 시장 모멘텀
    market_momentum = 0
    if last_tuesday in ret_1m.index:
        market_momentum = ret_1m.loc[last_tuesday].mean()
    
    # 종목 선정
    result = strategy.select_stocks(score_df, correlation_df, last_tuesday, ret_1m)
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
    
    # 현재가 조회 (한투 API)
    picks = result["picks"]
    prices = kis.get_prices(picks)
    sector_map = dict(zip(sp500["symbol"], sp500["sector"]))
    
    print(f"\nBUY Signal: {len(picks)} stocks")
    print("-" * 50)
    for i, (symbol, score, alloc) in enumerate(zip(picks, result["scores"], result["allocations"])):
        price = prices.get(symbol, "N/A")
        sector = sector_map.get(symbol, "")
        price_str = f"${price:.2f}" if isinstance(price, (int, float)) else price
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
# Trade Execution (한투 API)
# ============================================

def execute_signal(signal, sheets):
    """
    신호에 따라 매매 실행 (한투 API 사용)
    """
    if signal["signal"] != "BUY":
        print("No BUY signal - nothing to execute")
        return []
    
    trades = []
    
    # 현재 잔고 조회
    balance = kis.get_balance()
    if balance is None:
        print("Failed to get balance")
        return []
    
    current_holdings = {h["symbol"]: h for h in balance["holdings"]}
    new_picks = set(signal["picks"])
    current_symbols = set(current_holdings.keys())
    
    to_sell = current_symbols - new_picks
    to_buy = new_picks - current_symbols
    
    print("\n" + "=" * 60)
    print("Executing Trades")
    print("=" * 60)
    
    total_value = balance["total_value"]
    
    # 목표 배분
    target_allocations = {}
    for i, symbol in enumerate(signal["picks"]):
        if i < len(signal["allocations"]):
            target_allocations[symbol] = signal["allocations"][i]
    
    # === SELL: 새 픽에 없는 종목 매도 ===
    for symbol in to_sell:
        if symbol not in current_holdings:
            continue
        
        holding = current_holdings[symbol]
        qty = holding["shares"]
        
        result = kis.sell(symbol, qty)
        
        if result["success"]:
            return_pct = holding["profit_loss_pct"]
            trade = {
                "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
                "symbol": symbol,
                "action": "SELL",
                "shares": qty,
                "price": holding["current_price"],
                "amount": qty * holding["current_price"],
                "commission": 0,
                "slippage": 0,
                "return_pct": return_pct,
                "sector": "",
                "score": "",
                "memo": "Position closed"
            }
            trades.append(trade)
            print(f"  SELL: {symbol} | {qty} shares | Return: {return_pct:+.2f}%")
        else:
            print(f"  SELL FAILED: {symbol} - {result.get('message', '')}")
    
    # === BUY: 새로 진입할 종목 매수 ===
    # 잔고 재조회 (매도 후)
    balance = kis.get_balance()
    if balance is None:
        return trades
    
    available_cash = balance["cash"]
    
    for symbol in to_buy:
        if symbol not in target_allocations:
            continue
        
        allocation = target_allocations[symbol]
        invest_amount = total_value * allocation
        
        current_price = signal["prices"].get(symbol, 0)
        if current_price <= 0:
            price_data = kis.get_price(symbol)
            if price_data:
                current_price = price_data["price"]
        
        if current_price <= 0:
            print(f"  SKIP: {symbol} - Cannot get price")
            continue
        
        qty = int(invest_amount / current_price)
        if qty <= 0:
            print(f"  SKIP: {symbol} - Quantity too small")
            continue
        
        required_amount = qty * current_price
        if required_amount > available_cash:
            qty = int(available_cash / current_price)
            if qty <= 0:
                print(f"  SKIP: {symbol} - Insufficient cash")
                continue
        
        result = kis.buy(symbol, qty)
        
        if result["success"]:
            score_idx = signal["picks"].index(symbol) if symbol in signal["picks"] else -1
            score = signal["scores"][score_idx] if 0 <= score_idx < len(signal["scores"]) else 0
            
            trade = {
                "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
                "symbol": symbol,
                "action": "BUY",
                "shares": qty,
                "price": current_price,
                "amount": qty * current_price,
                "commission": 0,
                "slippage": 0,
                "return_pct": 0,
                "sector": signal.get("sectors", {}).get(symbol, ""),
                "score": round(score, 4),
                "memo": f"Weight: {allocation*100:.0f}%"
            }
            trades.append(trade)
            available_cash -= required_amount
            print(f"  BUY: {symbol} | {qty} shares | ${current_price:.2f} | Weight: {allocation*100:.0f}%")
        else:
            print(f"  BUY FAILED: {symbol} - {result.get('message', '')}")
    
    # Google Sheets 저장
    if trades:
        sheets.save_trades(trades)
    
    print(f"\nCompleted: {len(trades)} trades")
    
    return trades


# ============================================
# Stop Loss Check
# ============================================

def check_stop_loss(sheets):
    """
    손절 체크 및 실행
    """
    print("=" * 60)
    print("Stop Loss Check")
    print("=" * 60)
    
    balance = kis.get_balance()
    if balance is None:
        print("Failed to get balance")
        return []
    
    trades = []
    
    for holding in balance["holdings"]:
        symbol = holding["symbol"]
        return_pct = holding["profit_loss_pct"] / 100  # 퍼센트를 비율로
        
        print(f"  {symbol}: Avg ${holding['avg_price']:.2f} | "
              f"Now ${holding['current_price']:.2f} | {return_pct*100:+.2f}%")
        
        if return_pct <= STOP_LOSS:
            qty = holding["shares"]
            result = kis.sell(symbol, qty)
            
            if result["success"]:
                trade = {
                    "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
                    "symbol": symbol,
                    "action": "STOP_LOSS",
                    "shares": qty,
                    "price": holding["current_price"],
                    "amount": qty * holding["current_price"],
                    "commission": 0,
                    "slippage": 0,
                    "return_pct": round(return_pct * 100, 2),
                    "sector": "",
                    "score": "",
                    "memo": f"Stop loss at {STOP_LOSS*100:.0f}%"
                }
                trades.append(trade)
                print(f"  >>> STOP LOSS: {symbol} | {return_pct*100:.2f}%")
    
    if trades:
        sheets.save_trades(trades)
        send_stop_loss(trades)
        print(f"\n{len(trades)} stop loss executed")
    else:
        print("\nNo stop loss needed")
    
    return trades


# ============================================
# Daily Update
# ============================================

def record_daily_value(sheets):
    """
    일일 포트폴리오 가치 기록
    """
    balance = kis.get_balance()
    if balance is None:
        print("Failed to get balance")
        return None, None
    
    spy_price = get_spy_price()
    
    # 이전 기록 조회
    daily_df = sheets.load_daily_values()
    prev_value = INITIAL_CAPITAL
    prev_spy = spy_price
    if len(daily_df) > 0:
        try:
            prev_value = float(daily_df.iloc[-1]["Total_Value"])
            prev_spy = float(daily_df.iloc[-1]["SPY_Price"])
        except:
            pass
    
    # 수익률 계산
    total_value = balance["total_value"]
    daily_return = (total_value - prev_value) / prev_value * 100 if prev_value > 0 else 0
    spy_return = (spy_price - prev_spy) / prev_spy * 100 if prev_spy > 0 else 0
    alpha = daily_return - spy_return
    
    daily_data = {
        "date": datetime.now().strftime("%Y-%m-%d"),
        "total_value": total_value,
        "cash": balance["cash"],
        "stocks_value": total_value - balance["cash"],
        "daily_return_pct": round(daily_return, 2),
        "spy_price": spy_price,
        "spy_return_pct": round(spy_return, 2),
        "alpha": round(alpha, 2)
    }
    
    sheets.save_daily_value(daily_data)
    print(f"Daily value recorded: ${total_value:,.2f} ({daily_return:+.2f}%)")
    
    # 포트폴리오 정보
    port_value = {
        "total": total_value,
        "cash": balance["cash"],
        "stocks": total_value - balance["cash"],
        "holdings_detail": balance["holdings"]
    }
    
    return daily_data, port_value


# ============================================
# Performance Update
# ============================================

def update_performance(sheets):
    """
    전체 성과 업데이트
    """
    balance = kis.get_balance()
    if balance is None:
        return None
    
    trades_df = sheets.load_trades()
    daily_df = sheets.load_daily_values()
    
    total_value = balance["total_value"]
    total_return = (total_value - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100
    
    # 기간 계산
    days = 0
    start_date = ""
    if len(daily_df) > 0:
        try:
            start_date = daily_df.iloc[0]["Date"]
            days = (datetime.now() - datetime.strptime(start_date, "%Y-%m-%d")).days
        except:
            pass
    
    years = days / 365 if days > 0 else 0
    cagr = ((total_value / INITIAL_CAPITAL) ** (1/years) - 1) * 100 if years > 0 else 0
    
    # SPY 수익률
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
    
    # 승률
    win_rate = 0
    total_trades = len(trades_df)
    if total_trades > 0:
        try:
            returns = trades_df["Return%"].astype(float)
            wins = (returns > 0).sum()
            win_rate = wins / total_trades * 100
        except:
            pass
    
    # Sharpe
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
        "current_value": total_value,
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
# Print Portfolio
# ============================================

def print_portfolio():
    """
    현재 포트폴리오 출력
    """
    kis.print_balance()


# ============================================
# Main Functions
# ============================================

def run_daily(sheets=None):
    """
    일일 루틴 (월, 수, 목, 금)
    - 손절 체크
    - 일일 가치 기록
    - 성과 업데이트
    """
    if sheets is None:
        sheets = SheetsManager()
    
    print("\n" + "=" * 60)
    print(f"Daily Run: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"Mode: {KIS_MODE}")
    print("=" * 60)
    
    try:
        # 손절 체크
        stop_trades = check_stop_loss(sheets)
        
        # 일일 가치 기록
        daily_data, port_value = record_daily_value(sheets)
        
        # 성과 업데이트
        update_performance(sheets)
        
        # Telegram 알림
        if daily_data and port_value:
            send_daily_summary(daily_data, port_value)
        
    except Exception as e:
        print(f"Error: {e}")
        send_error(str(e))


def run_weekly(sheets=None):
    """
    주간 루틴 (화요일)
    - 신호 생성
    - 매매 실행
    - 손절 체크
    - 일일 가치 기록
    - 성과 업데이트
    """
    if sheets is None:
        sheets = SheetsManager()
    
    print("\n" + "=" * 60)
    print(f"Weekly Run: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"Mode: {KIS_MODE}")
    print("=" * 60)
    
    try:
        # 신호 생성
        signal = get_today_signal()
        sheets.save_signal(signal)
        send_signal(signal)
        
        # 매매 실행
        if signal["signal"] == "BUY":
            trades = execute_signal(signal, sheets)
            if trades:
                send_trades(trades)
        
        # 손절 체크
        stop_trades = check_stop_loss(sheets)
        
        # 일일 가치 기록
        daily_data, port_value = record_daily_value(sheets)
        
        # 성과 업데이트
        update_performance(sheets)
        
        # Telegram 포트폴리오
        if port_value:
            send_portfolio(port_value)
        
    except Exception as e:
        print(f"Error: {e}")
        send_error(str(e))


# ============================================
# CLI
# ============================================

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python paper_trading.py [signal|portfolio|execute|stoploss|daily|weekly|balance]")
        sys.exit(1)
    
    cmd = sys.argv[1].lower()
    
    if cmd == "balance":
        # 잔고만 확인 (Sheets 필요없음)
        print_portfolio()
    
    elif cmd == "signal":
        sheets = SheetsManager()
        signal = get_today_signal()
        sheets.save_signal(signal)
        send_signal(signal)
    
    elif cmd == "portfolio":
        print_portfolio()
        sheets = SheetsManager()
        balance = kis.get_balance()
        if balance:
            port_value = {
                "total": balance["total_value"],
                "cash": balance["cash"],
                "stocks": balance["total_value"] - balance["cash"],
                "holdings_detail": balance["holdings"]
            }
            send_portfolio(port_value)
    
    elif cmd == "execute":
        sheets = SheetsManager()
        signal = get_today_signal()
        sheets.save_signal(signal)
        send_signal(signal)
        if signal["signal"] == "BUY":
            trades = execute_signal(signal, sheets)
            if trades:
                send_trades(trades)
    
    elif cmd == "stoploss":
        sheets = SheetsManager()
        check_stop_loss(sheets)
    
    elif cmd == "daily":
        sheets = SheetsManager()
        run_daily(sheets)
    
    elif cmd == "weekly":
        sheets = SheetsManager()
        run_weekly(sheets)
    
    else:
        print(f"Unknown command: {cmd}")
        print("Available: signal, portfolio, execute, stoploss, daily, weekly, balance")