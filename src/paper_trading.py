# ============================================
# 파일명: src/paper_trading.py
# 설명: 모멘텀 전략 라이브 페이퍼 트레이딩 실행기
#
# 역할 요약:
#   CustomStrategy(모멘텀)를 사용하는 라이브 트레이딩 시스템.
#   매주 화요일 GitHub Actions에 의해 자동 실행.
#   실제 주문은 없고, 신호를 Telegram과 Google Sheets에만 기록.
#
# 실행 흐름:
#   1. yfinance로 최근 LOOKBACK_DAYS(약 2년) 주가 데이터 다운로드
#   2. CustomStrategy.prepare() → 상관관계 + 모멘텀 점수 계산
#   3. 오늘이 화요일이면: 종목 선정 → 신호 발송 → Sheets 기록
#   4. 보유 종목 평가 (Sheets의 Holdings 기반) → 포트폴리오 Telegram 발송
#   5. 손절 체크: 트레일링 스탑 (매수일 이후 고점 대비 -7%) → 알림
#   6. 일별/월별/연간 가치 기록 (Sheets)
#
# 모멘텀 전략 vs 하이브리드 전략 분리:
#   - paper_trading.py: 모멘텀 전략 (CustomStrategy) 전용
#   - hybrid_trading.py: 하이브리드 전략 (HybridStrategy) 전용
#   → 두 시스템은 별개의 Google Sheets 스프레드시트에 기록
#
# 주요 함수:
#   download_recent_data()   → yfinance 데이터 다운로드
#   process_data()           → 롱포맷 변환 (data.py의 download_stock_data와 유사)
#   run_paper_trading()      → 메인 실행 함수
#   main()                   → GitHub Actions 진입점
#
# 의존 관계:
#   ← strategy.py (CustomStrategy, prepare_price_data, filter_tuesday)
#   ← sheets.py (SheetsManager)
#   ← telegram.py (send_signal, send_portfolio 등)
#   ← config.py (INITIAL_CAPITAL, STOP_LOSS, LOOKBACK_DAYS 등)
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
    send_trade_signal,
    send_rebalancing
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
        strategy = CustomStrategy(ma_filter=True)
    
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
        # 참고용 top 3 종목 (시장 하락 중에도 스코어 상위 표시)
        top_row = score_df.loc[last_tuesday].dropna() if last_tuesday in score_df.index else pd.Series(dtype=float)
        top3 = top_row.nlargest(3)
        return {
            "date": last_tuesday,
            "signal": "HOLD",
            "message": "Market momentum <= 0",
            "picks": top3.index.tolist(),
            "scores": top3.values.tolist(),
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


def get_daily_ref_signal(strategy=None):
    """
    매일 참고용 시그널 (전일 종가 기준, 일별 모멘텀)

    화요일 공식 시그널과 달리 Tuesday 필터 없이
    전일 종가 기준으로 매일 스코어를 새로 계산.
    - 1개월: pct_change(21)  ← 21 거래일
    - 3개월: pct_change(63)
    - 6개월: pct_change(126)
    """
    if strategy is None:
        strategy = CustomStrategy(ma_filter=True)

    sp500 = get_sp500_list()
    symbols = sp500["symbol"].tolist()
    if "SPY" not in symbols:
        symbols.append("SPY")

    df = download_recent_data(symbols)
    if df.empty:
        return {"signal": "ERROR", "message": "Download failed"}

    # 전체 일별 종가 (Tuesday 필터 없음)
    price_df = prepare_price_data(df)
    if price_df.empty:
        return {"signal": "ERROR", "message": "No price data"}

    # 전일 종가 기준 (오늘 미완성 봉 제외)
    price_df = price_df.iloc[:-1]
    if price_df.empty:
        return {"signal": "ERROR", "message": "No previous day data"}

    latest_date = price_df.index[-1]

    # 일별 모멘텀 스코어 계산
    ret_1m = price_df.pct_change(21)
    ret_3m = price_df.pct_change(63)
    ret_6m = price_df.pct_change(126)
    score_df = (
        ret_1m * strategy.weight_1m
        + ret_3m * strategy.weight_3m
        + ret_6m * strategy.weight_6m
    )

    # 상관관계는 전체 일별 데이터로 계산
    correlation_df = strategy.calc_correlation(price_df)

    market_momentum = ret_1m.loc[latest_date].mean() if latest_date in ret_1m.index else 0

    result = strategy.select_stocks(score_df, correlation_df, latest_date, ret_1m)
    spy_price = get_spy_price()

    if result is None:
        # 참고용 top 3 종목 (시장 하락 중에도 스코어 상위 표시)
        top_row = score_df.loc[latest_date].dropna() if latest_date in score_df.index else pd.Series(dtype=float)
        top3 = top_row.nlargest(3)
        return {
            "date": latest_date,
            "signal": "HOLD",
            "picks": top3.index.tolist(),
            "scores": top3.values.tolist(),
            "allocations": [],
            "prices": {},
            "market_momentum": market_momentum,
            "spy_price": spy_price,
            "market_trend": "DOWN",
        }

    picks = result["picks"]
    prices = get_current_prices(picks)

    return {
        "date": latest_date,
        "signal": "BUY",
        "picks": picks,
        "scores": result["scores"],
        "allocations": result["allocations"],
        "prices": prices,
        "market_momentum": market_momentum,
        "spy_price": spy_price,
        "market_trend": "UP",
    }


# ============================================
# Portfolio Management (시트 기반)
# ============================================

def get_portfolio_from_sheets(sheets, sync_result=None):
    """
    구글시트에서 포트폴리오 정보 가져오기
    
    Args:
        sheets: SheetsManager 인스턴스
        sync_result: sync_holdings_from_trades() 결과 (있으면 cash 사용)
    """
    try:
        holdings_df = sheets.load_holdings()
        
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
        
        # 현금: sync_result에서 가져오거나 Daily_Value에서 가져옴
        if sync_result and "cash" in sync_result:
            cash = sync_result["cash"]
        else:
            daily_df = sheets.load_daily_values()
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


# ============================================
# Rebalancing (리밸런싱 계산)
# ============================================

def calculate_rebalancing(portfolio, signal, min_trade_amount=50):
    """
    기존 보유 vs 새 신호 비교 → 리밸런싱 계산
    
    Args:
        portfolio: get_portfolio_from_sheets() 결과
        signal: get_today_signal() 결과
        min_trade_amount: 최소 거래 금액 (이하면 무시)
    
    Returns:
        dict: {
            "actions": [
                {"action": "SELL", "symbol": "NVDA", "shares": 3, "price": 850, "amount": 2550, "reason": "신호에 없음"},
                {"action": "ADD", "symbol": "WDC", "shares": 2, "price": 274, "amount": 548, "reason": "비중 증가"},
                {"action": "BUY", "symbol": "TER", "shares": 5, "price": 180, "amount": 900, "reason": "신규 매수"},
                {"action": "HOLD", "symbol": "MU", "shares": 8, "price": 108, "amount": 0, "reason": "유지"},
            ],
            "summary": {
                "total_buy": 1448,
                "total_sell": 2550,
                "net_cash_change": 1102
            }
        }
    """
    if signal.get("signal") != "BUY":
        return {
            "actions": [],
            "summary": {"total_buy": 0, "total_sell": 0, "net_cash_change": 0},
            "message": "HOLD 신호 - 매매 없음"
        }
    
    holdings = portfolio.get("holdings", [])
    cash = portfolio.get("cash", 0)
    total_value = portfolio.get("total_value", INITIAL_CAPITAL)
    
    new_picks = signal.get("picks", [])
    allocations = signal.get("allocations", [])
    prices = signal.get("prices", {})
    
    # 현재 보유 종목 dict
    current_holdings = {h["symbol"]: h for h in holdings}
    
    actions = []
    total_buy = 0
    total_sell = 0
    
    # ----- 1. 기존 보유 중 신호에 없는 종목 → SELL -----
    for symbol, holding in current_holdings.items():
        if symbol not in new_picks:
            shares = holding["shares"]
            price = holding.get("current_price", holding["avg_price"])
            amount = shares * price
            profit_pct = holding.get("profit_loss_pct", 0)
            
            actions.append({
                "action": "SELL",
                "symbol": symbol,
                "shares": shares,
                "price": round(price, 2),
                "amount": round(amount, 2),
                "profit_pct": round(profit_pct, 2),
                "reason": "신호에서 제외"
            })
            total_sell += amount
    
    # ----- 2. 신호 종목 처리 -----
    # 매도 후 예상 Cash
    expected_cash = cash + total_sell
    
    for i, symbol in enumerate(new_picks):
        alloc = allocations[i] if i < len(allocations) else 0.33
        price = prices.get(symbol, 0)
        
        if price <= 0:
            continue
        
        # 목표 금액
        target_amount = total_value * alloc
        target_shares = int(target_amount / price)
        
        # 현재 보유
        current_shares = 0
        if symbol in current_holdings:
            current_shares = current_holdings[symbol]["shares"]
        
        # 차이 계산
        diff_shares = target_shares - current_shares
        diff_amount = diff_shares * price
        
        if diff_shares > 0 and diff_amount >= min_trade_amount:
            # 매수 (신규 또는 추가)
            action_type = "BUY" if current_shares == 0 else "ADD"
            
            # 가용 현금 체크
            if diff_amount > expected_cash * 0.95:
                diff_shares = int(expected_cash * 0.95 / price)
                diff_amount = diff_shares * price
            
            if diff_shares > 0:
                actions.append({
                    "action": action_type,
                    "symbol": symbol,
                    "shares": diff_shares,
                    "price": round(price, 2),
                    "amount": round(diff_amount, 2),
                    "current_shares": current_shares,
                    "target_shares": target_shares,
                    "reason": "신규 매수" if action_type == "BUY" else "비중 증가"
                })
                total_buy += diff_amount
                expected_cash -= diff_amount
        
        elif diff_shares < 0 and abs(diff_amount) >= min_trade_amount:
            # 비중 축소 (일부 매도)
            sell_shares = abs(diff_shares)
            sell_amount = sell_shares * price
            
            actions.append({
                "action": "REDUCE",
                "symbol": symbol,
                "shares": sell_shares,
                "price": round(price, 2),
                "amount": round(sell_amount, 2),
                "current_shares": current_shares,
                "target_shares": target_shares,
                "reason": "비중 축소"
            })
            total_sell += sell_amount
        
        else:
            # 유지
            if current_shares > 0:
                actions.append({
                    "action": "HOLD",
                    "symbol": symbol,
                    "shares": current_shares,
                    "price": round(price, 2),
                    "amount": 0,
                    "reason": "유지 (차이 미미)"
                })
    
    # 정렬: SELL → REDUCE → HOLD → ADD → BUY
    action_order = {"SELL": 0, "REDUCE": 1, "HOLD": 2, "ADD": 3, "BUY": 4}
    actions.sort(key=lambda x: action_order.get(x["action"], 5))
    
    return {
        "actions": actions,
        "summary": {
            "total_buy": round(total_buy, 2),
            "total_sell": round(total_sell, 2),
            "net_cash_change": round(total_sell - total_buy, 2)
        }
    }


def print_rebalancing(rebalancing):
    """
    리밸런싱 결과 출력
    """
    print("\n" + "=" * 60)
    print("📊 리밸런싱 계산 결과")
    print("=" * 60)
    
    if not rebalancing.get("actions"):
        print(rebalancing.get("message", "매매 없음"))
        return
    
    for act in rebalancing["actions"]:
        action = act["action"]
        symbol = act["symbol"]
        shares = act["shares"]
        price = act["price"]
        amount = act["amount"]
        reason = act.get("reason", "")
        
        if action == "SELL":
            emoji = "🔴"
            profit = act.get("profit_pct", 0)
            print(f"{emoji} {action:6} {symbol:5} | {shares}주 @ ${price} = ${amount:,.0f} ({profit:+.1f}%) - {reason}")
        elif action == "REDUCE":
            emoji = "🟠"
            print(f"{emoji} {action:6} {symbol:5} | {shares}주 @ ${price} = ${amount:,.0f} - {reason}")
        elif action == "HOLD":
            emoji = "⚪"
            print(f"{emoji} {action:6} {symbol:5} | {shares}주 @ ${price} - {reason}")
        elif action == "ADD":
            emoji = "🟢"
            print(f"{emoji} {action:6} {symbol:5} | +{shares}주 @ ${price} = ${amount:,.0f} - {reason}")
        elif action == "BUY":
            emoji = "🟢"
            print(f"{emoji} {action:6} {symbol:5} | {shares}주 @ ${price} = ${amount:,.0f} - {reason}")
    
    summary = rebalancing["summary"]
    print("-" * 60)
    print(f"총 매도: ${summary['total_sell']:,.0f}")
    print(f"총 매수: ${summary['total_buy']:,.0f}")
    print(f"현금 변화: ${summary['net_cash_change']:+,.0f}")
    print("=" * 60)


TRAILING_STOP_PCT = 0.07  # 고점 대비 -7% (트레일링 스탑)


def _get_peak_price(symbol, buy_date):
    """
    매수일부터 현재까지의 최고가 조회 (트레일링 스탑 기준)

    Args:
        symbol: 티커
        buy_date: 매수일 (문자열 'YYYY-MM-DD' 또는 날짜형)

    Returns:
        float: 기간 내 최고가, 조회 실패 시 None
    """
    try:
        ticker = yf.Ticker(symbol)
        hist = ticker.history(start=str(buy_date)[:10])
        if not hist.empty:
            return float(hist["High"].max())
    except Exception as e:
        print(f"  Peak price error ({symbol}): {e}")
    return None


def check_stop_loss(sheets):
    """
    트레일링 스탑 체크 (알림만, 실제 매도는 수동)

    로직: 매수일 이후 최고가 대비 -TRAILING_STOP_PCT% 이하이면 알림
    고점 조회 실패 시 기존 고정 손절(STOP_LOSS)로 폴백
    """
    print("=" * 60)
    print("Trailing Stop Check")
    print("=" * 60)
    print(f"  기준: 고점 대비 -{TRAILING_STOP_PCT*100:.0f}%")

    portfolio = get_portfolio_from_sheets(sheets)
    stop_loss_alerts = []

    for holding in portfolio["holdings"]:
        symbol = holding["symbol"]
        current_price = holding["current_price"]
        avg_price = holding["avg_price"]
        buy_date = holding.get("buy_date", "")

        # 매수일 이후 최고가 조회
        peak_price = _get_peak_price(symbol, buy_date) if buy_date else None

        if peak_price:
            stop_price = peak_price * (1 - TRAILING_STOP_PCT)
            triggered = current_price <= stop_price
            pct_from_peak = (current_price - peak_price) / peak_price * 100
            print(f"  {symbol}: Avg ${avg_price:.2f} | Peak ${peak_price:.2f} | "
                  f"Now ${current_price:.2f} | Peak대비 {pct_from_peak:+.2f}%")
        else:
            # 고점 조회 실패 → 고정 손절 폴백
            return_pct = (current_price - avg_price) / avg_price
            triggered = return_pct <= STOP_LOSS
            pct_from_peak = return_pct * 100
            print(f"  {symbol}: Avg ${avg_price:.2f} | Now ${current_price:.2f} | "
                  f"{pct_from_peak:+.2f}% (고점조회실패→고정손절)")

        if triggered:
            stop_loss_alerts.append({
                "symbol": symbol,
                "shares": holding["shares"],
                "avg_price": avg_price,
                "current_price": current_price,
                "peak_price": peak_price,
                "return_pct": round(pct_from_peak, 2)
            })
            print(f"  >>> TRAILING STOP ALERT: {symbol} | 고점대비 {pct_from_peak:.2f}%")

    if stop_loss_alerts:
        send_stop_loss(stop_loss_alerts)
        print(f"\n⚠️ {len(stop_loss_alerts)} trailing stop alerts!")
        print("→ 한투 앱에서 수동 매도 필요!")
    else:
        print("\nNo trailing stop triggered")

    return stop_loss_alerts


# ============================================
# Daily Update
# ============================================

def record_daily_value(sheets, sync_result=None):
    """
    일일 포트폴리오 가치 기록
    
    Args:
        sheets: SheetsManager 인스턴스
        sync_result: sync_holdings_from_trades() 결과 (있으면 cash 사용)
    """
    portfolio = get_portfolio_from_sheets(sheets, sync_result)
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


def update_performance(sheets, exchange_rate=1400):
    """
    전체 성과 업데이트 (수수료/세금 포함)
    
    Args:
        sheets: SheetsManager 인스턴스
        exchange_rate: 원/달러 환율 (세금 계산용)
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
    
    # 승률 및 수수료/실현손익
    win_rate = 0
    total_trades = len(trades_df)
    total_commission = 0
    total_realized_pnl = 0
    
    if total_trades > 0:
        try:
            returns = trades_df["Return%"].astype(float)
            wins = (returns > 0).sum()
            win_rate = wins / total_trades * 100
            
            # 수수료 합계
            if "Commission" in trades_df.columns:
                total_commission = trades_df["Commission"].astype(float).sum()
            
            # 실현손익 합계
            if "Realized_PnL" in trades_df.columns:
                total_realized_pnl = trades_df["Realized_PnL"].astype(float).sum()
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
    
    # 세금 계산 (해외주식 양도소득세: 22%, 250만원 공제)
    realized_pnl_krw = total_realized_pnl * exchange_rate
    taxable_amount = max(0, realized_pnl_krw - 2500000)
    est_tax = round(taxable_amount * 0.22)
    
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
        "days": days,
        "total_commission": round(total_commission, 2),
        "total_realized_pnl": round(total_realized_pnl, 2),
        "est_tax": est_tax
    }
    
    sheets.save_performance(metrics)
    
    # 월간/연간 리포트 자동 업데이트
    sheets.update_monthly_summary()
    sheets.update_yearly_summary(exchange_rate)
    
    print("=" * 60)
    print("Performance Updated")
    print("=" * 60)
    print(f"Total Return: {total_return:+.2f}%")
    print(f"SPY Return: {spy_return:+.2f}%")
    print(f"Alpha: {total_return - spy_return:+.2f}%")
    print(f"MDD: {mdd:.2f}%")
    print(f"Win Rate: {win_rate:.1f}%")
    print(f"Total Commission: ${total_commission:,.2f}")
    print(f"Total Realized P&L: ${total_realized_pnl:,.2f}")
    print(f"Est. Tax (KRW): ₩{est_tax:,.0f}")
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
    - Holdings 동기화 (Trades 기반)
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
        # Holdings 동기화 (Trades 기반) + Cash 계산
        sync_result = sheets.sync_holdings_from_trades()
        
        # 손절 체크 (알림만)
        stop_alerts = check_stop_loss(sheets)
        
        # 일일 가치 기록 (sync_result의 cash 사용)
        daily_data, portfolio = record_daily_value(sheets, sync_result)
        
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
            ref_signal = get_daily_ref_signal()
            sheets.save_signal(ref_signal)
            send_daily_summary(daily_data, port_value, signal=ref_signal, strategy="Momentum")
        
    except Exception as e:
        print(f"Error: {e}")
        send_error(str(e))


def run_weekly(sheets=None):
    """
    주간 루틴 (화요일)
    - Holdings 동기화 (Trades 기반)
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
        # Holdings 동기화 (Trades 기반) + Cash 계산
        sync_result = sheets.sync_holdings_from_trades()
        
        # 현재 포트폴리오 가치 확인 (sync_result의 cash 사용)
        portfolio = get_portfolio_from_sheets(sheets, sync_result)
        total_capital = portfolio["total_value"]
        
        # 신호 생성
        signal = get_today_signal()
        sheets.save_signal(signal)
        
        # 리밸런싱 계산 (기존 보유 vs 새 신호)
        rebalancing = calculate_rebalancing(portfolio, signal)
        print_rebalancing(rebalancing)
        
        # 텔레그램 발송 (리밸런싱 안내)
        send_rebalancing(rebalancing, total_capital)
        
        # 손절 체크 (알림만)
        stop_alerts = check_stop_loss(sheets)
        
        # 일일 가치 기록 (sync_result의 cash 사용)
        daily_data, portfolio = record_daily_value(sheets, sync_result)
        
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
        portfolio = get_portfolio_from_sheets(sheets)
        total_capital = portfolio["total_value"]
        signal = get_today_signal()
        sheets.save_signal(signal)
        send_signal(signal, total_capital)
    
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