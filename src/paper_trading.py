# ============================================
# src/paper_trading.py
# Paper Trading System (API 없이)
# - 신호 생성 (yfinance)
# - Google Sheets 기록
# - Telegram 알림
# - 수동 매매 기록
# ============================================

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta

from config import (
    INITIAL_CAPITAL,
    STOP_LOSS,
    TOP_N,
    LOOKBACK_DAYS,
    SP500_BACKUP
)
from data import get_sp500_list
from strategy import CustomStrategy, prepare_price_data, filter_tuesday
from sheets import SheetsManager
from telegram import (
    send_signal,
    send_portfolio,
    send_stop_loss,
    send_daily_summary,
    send_error,
    send_trade_signal
)


# ============================================
# Data Functions (yfinance 사용)
# ============================================


def download_recent_data(symbols, days=LOOKBACK_DAYS):
    """
    yfinance로 최근 데이터 다운로드
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


def get_current_prices(symbols):
    """
    yfinance로 현재가 조회
    """
    prices = {}
    for symbol in symbols:
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="1d")
            if not hist.empty:
                prices[symbol] = round(hist["Close"].iloc[-1], 2)
        except:
            continue
    return prices


def get_spy_price():
    """
    SPY 현재가 조회
    """
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
    """
    if strategy is None:
        strategy = CustomStrategy()
    
    print("=" * 60)
    print(f"Signal Generation")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 60)
    
    # S&P 500 종목 가져오기
    sp500 = get_sp500_list()
    symbols = sp500["symbol"].tolist()
    if "SPY" not in symbols:
        symbols.append("SPY")
    
    # 데이터 다운로드
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
    
    # 현재가 조회
    picks = result["picks"]
    prices = get_current_prices(picks)
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
# Portfolio Management (시트 기반)
# ============================================

def get_portfolio_from_sheets(sheets):
    """
    구글시트에서 포트폴리오 정보 가져오기
    """
    try:
        holdings_df = sheets.load_holdings()
        daily_df = sheets.load_daily_values()
        
        holdings = []
        if not holdings_df.empty:
            for _, row in holdings_df.iterrows():
                symbol = row.get("Symbol", "")
                if symbol:
                    holdings.append({
                        "symbol": symbol,
                        "shares": int(row.get("Shares", 0)),
                        "avg_price": float(row.get("Avg_Price", 0)),
                        "buy_date": row.get("Buy_Date", "")
                    })
        
        # 현재가 조회
        if holdings:
            symbols = [h["symbol"] for h in holdings]
            prices = get_current_prices(symbols)
            for h in holdings:
                h["current_price"] = prices.get(h["symbol"], h["avg_price"])
                h["value"] = h["shares"] * h["current_price"]
                h["profit_loss"] = (h["current_price"] - h["avg_price"]) * h["shares"]
                h["profit_loss_pct"] = ((h["current_price"] / h["avg_price"]) - 1) * 100 if h["avg_price"] > 0 else 0
        
        # 현금 및 총자산
        cash = INITIAL_CAPITAL
        if not daily_df.empty:
            try:
                cash = float(daily_df.iloc[-1].get("Cash", INITIAL_CAPITAL))
            except:
                pass
        
        stocks_value = sum(h["value"] for h in holdings)
        total_value = cash + stocks_value
        
        return {
            "holdings": holdings,
            "cash": cash,
            "stocks_value": stocks_value,
            "total_value": total_value
        }
    except Exception as e:
        print(f"Portfolio error: {e}")
        return {
            "holdings": [],
            "cash": INITIAL_CAPITAL,
            "stocks_value": 0,
            "total_value": INITIAL_CAPITAL
        }


def check_stop_loss(sheets):
    """
    손절 체크 (알림만, 실제 매도는 수동)
    """
    print("=" * 60)
    print("Stop Loss Check")
    print("=" * 60)
    
    portfolio = get_portfolio_from_sheets(sheets)
    stop_loss_alerts = []
    
    for holding in portfolio["holdings"]:
        symbol = holding["symbol"]
        return_pct = holding["profit_loss_pct"] / 100
        
        print(f"  {symbol}: Avg ${holding['avg_price']:.2f} | "
              f"Now ${holding['current_price']:.2f} | {return_pct*100:+.2f}%")
        
        if return_pct <= STOP_LOSS:
            stop_loss_alerts.append({
                "symbol": symbol,
                "shares": holding["shares"],
                "avg_price": holding["avg_price"],
                "current_price": holding["current_price"],
                "return_pct": round(return_pct * 100, 2)
            })
            print(f"  >>> STOP LOSS ALERT: {symbol} | {return_pct*100:.2f}%")
    
    if stop_loss_alerts:
        send_stop_loss(stop_loss_alerts)
        print(f"\n⚠️ {len(stop_loss_alerts)} stop loss alerts!")
        print("→ 한투 앱에서 수동 매도 필요!")
    else:
        print("\nNo stop loss needed")
    
    return stop_loss_alerts


# ============================================
# Daily Update
# ============================================

def record_daily_value(sheets):
    """
    일일 포트폴리오 가치 기록
    """
    portfolio = get_portfolio_from_sheets(sheets)
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
    total_value = portfolio["total_value"]
    daily_return = (total_value - prev_value) / prev_value * 100 if prev_value > 0 else 0
    spy_return = (spy_price - prev_spy) / prev_spy * 100 if prev_spy > 0 else 0
    alpha = daily_return - spy_return
    
    daily_data = {
        "date": datetime.now().strftime("%Y-%m-%d"),
        "total_value": total_value,
        "cash": portfolio["cash"],
        "stocks_value": portfolio["stocks_value"],
        "daily_return_pct": round(daily_return, 2),
        "spy_price": spy_price,
        "spy_return_pct": round(spy_return, 2),
        "alpha": round(alpha, 2)
    }
    
    sheets.save_daily_value(daily_data)
    print(f"Daily value recorded: ${total_value:,.2f} ({daily_return:+.2f}%)")
    
    return daily_data, portfolio


def update_performance(sheets):
    """
    전체 성과 업데이트
    """
    portfolio = get_portfolio_from_sheets(sheets)
    trades_df = sheets.load_trades()
    daily_df = sheets.load_daily_values()
    
    total_value = portfolio["total_value"]
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

def print_portfolio(sheets):
    """
    현재 포트폴리오 출력
    """
    portfolio = get_portfolio_from_sheets(sheets)
    
    print("=" * 60)
    print("Portfolio Balance")
    print("=" * 60)
    print(f"Cash: ${portfolio['cash']:,.2f}")
    print(f"Stocks: ${portfolio['stocks_value']:,.2f}")
    print(f"Total: ${portfolio['total_value']:,.2f}")
    print()
    
    if portfolio["holdings"]:
        print(f"Holdings ({len(portfolio['holdings'])})")
        print("-" * 60)
        for h in portfolio["holdings"]:
            print(f"  {h['symbol']:6} | {h['shares']:>4} shares | "
                  f"Avg: ${h['avg_price']:>8.2f} | "
                  f"Now: ${h['current_price']:>8.2f} | "
                  f"{h['profit_loss_pct']:>+6.2f}%")
        print("-" * 60)
    else:
        print("No holdings")
    print("=" * 60)


# ============================================
# Main Functions
# ============================================

def run_daily(sheets=None):
    """
    일일 루틴 (월, 수, 목, 금)
    - 손절 체크 (알림)
    - 일일 가치 기록
    - 성과 업데이트
    """
    if sheets is None:
        sheets = SheetsManager()
    
    print("\n" + "=" * 60)
    print(f"Daily Run: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 60)
    
    try:
        # 손절 체크 (알림만)
        stop_alerts = check_stop_loss(sheets)
        
        # 일일 가치 기록
        daily_data, portfolio = record_daily_value(sheets)
        
        # 성과 업데이트
        update_performance(sheets)
        
        # Telegram 알림
        if daily_data:
            port_value = {
                "total": portfolio["total_value"],
                "cash": portfolio["cash"],
                "stocks": portfolio["stocks_value"],
                "holdings_detail": portfolio["holdings"]
            }
            send_daily_summary(daily_data, port_value)
        
    except Exception as e:
        print(f"Error: {e}")
        send_error(str(e))


def run_weekly(sheets=None):
    """
    주간 루틴 (화요일)
    - 신호 생성
    - 텔레그램 알림 (수동 매매 안내)
    - 손절 체크
    - 일일 가치 기록
    - 성과 업데이트
    """
    if sheets is None:
        sheets = SheetsManager()
    
    print("\n" + "=" * 60)
    print(f"Weekly Run: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 60)
    
    try:
        # 신호 생성
        signal = get_today_signal()
        sheets.save_signal(signal)
        send_signal(signal)
        
        # 매매 안내 메시지
        if signal["signal"] == "BUY":
            send_trade_signal()
        
        # 손절 체크 (알림만)
        stop_alerts = check_stop_loss(sheets)
        
        # 일일 가치 기록
        daily_data, portfolio = record_daily_value(sheets)
        
        # 성과 업데이트
        update_performance(sheets)
        
        # Telegram 포트폴리오
        if portfolio:
            port_value = {
                "total": portfolio["total_value"],
                "cash": portfolio["cash"],
                "stocks": portfolio["stocks_value"],
                "holdings_detail": portfolio["holdings"]
            }
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
        print("Usage: python paper_trading.py [signal|portfolio|stoploss|daily|weekly]")
        sys.exit(1)
    
    cmd = sys.argv[1].lower()
    sheets = SheetsManager()
    
    if cmd == "signal":
        signal = get_today_signal()
        sheets.save_signal(signal)
        send_signal(signal)
    
    elif cmd == "portfolio":
        print_portfolio(sheets)
    
    elif cmd == "stoploss":
        check_stop_loss(sheets)
    
    elif cmd == "daily":
        run_daily(sheets)
    
    elif cmd == "weekly":
        run_weekly(sheets)
    
    else:
        print(f"Unknown command: {cmd}")
        print("Available: signal, portfolio, stoploss, daily, weekly")
