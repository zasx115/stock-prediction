# ============================================
# 파일명: src/telegram.py
# 설명: Telegram 알림 모듈
#
# 역할 요약:
#   페이퍼 트레이딩 시스템의 모든 Telegram 알림을 관리.
#   Telegram Bot API를 직접 호출하여 HTML 포맷 메시지를 발송.
#
# 인증:
#   TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID 환경변수 필수
#   (GitHub Actions Secrets 또는 로컬 .env에서 설정)
#
# 알림 유형:
#   모멘텀 전략 (paper_trading.py 사용):
#     send_signal()           → 매주 BUY/HOLD 신호
#     send_portfolio()        → 포트폴리오 현황
#     send_rebalancing()      → 리밸런싱 상세
#     send_trade_signal()     → 개별 매매 알림
#
#   하이브리드 전략 (hybrid_trading.py 사용):
#     send_hybrid_signal()        → 하이브리드 BUY/HOLD 신호
#     send_hybrid_portfolio()     → 하이브리드 포트폴리오 현황
#     send_hybrid_rebalancing()   → 하이브리드 리밸런싱 상세
#
#   공통:
#     send_stop_loss()        → 손절 알림
#     send_daily_summary()    → 일별 요약
#     send_error()            → 에러 알림
#
# [주의] 데드 코드:
#   send_trades()  → 정의되어 있으나 어디서도 호출되지 않음
#                    (수동 매매 기록 시 사용 의도였으나 현재 미사용)
#
# 의존 관계:
#   ← config.py (INITIAL_CAPITAL)
#   → paper_trading.py, hybrid_trading.py 에서 호출
# ============================================

import requests
import os
from datetime import datetime
from config import INITIAL_CAPITAL

# ============================================
# Settings
# ============================================

BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")

# ============================================
# Send Message
# ============================================

def send_message(text, parse_mode="HTML"):
    """
    텔레그램 메시지 전송
    """
    if not BOT_TOKEN or not CHAT_ID:
        print("Telegram credentials not found")
        return False
    
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    
    data = {
        "chat_id": CHAT_ID,
        "text": text,
        "parse_mode": parse_mode
    }
    
    try:
        response = requests.post(url, data=data)
        if response.status_code == 200:
            print("Telegram message sent")
            return True
        else:
            print(f"Telegram error: {response.status_code}")
            return False
    except Exception as e:
        print(f"Telegram error: {e}")
        return False


# ============================================
# Message Templates
# ============================================

def send_signal(signal, total_capital=None):
    """
    매매 신호 메시지
    
    Args:
        signal: 신호 정보
        total_capital: 현재 총 자본금 (None이면 INITIAL_CAPITAL 사용)
    """
    today = datetime.now().strftime("%Y-%m-%d")
    
    # 현재 자본금 (전달받지 못하면 초기 자본금 사용)
    capital = total_capital if total_capital else INITIAL_CAPITAL
    
    if signal["signal"] == "BUY":
        # 종목별 정보
        picks_text = ""
        for i, (symbol, score, alloc) in enumerate(zip(signal["picks"], signal["scores"], signal["allocations"])):
            price = signal.get("prices", {}).get(symbol, 0)
            # 투자금액 계산 (현재 자본금 기준)
            invest_amount = capital * alloc
            shares = int(invest_amount / price) if price > 0 else 0
            picks_text += f"{i+1}. {symbol} ({score:.4f}) ({alloc*100:.0f}%) - ${price:.2f} ({shares}주)\n"
        
        text = f"""<b>BUY Signal ({today})</b>
Capital: ${capital:,.2f}
Market: UP (Momentum: {signal.get("market_momentum", 0):.4f})
SPY: ${signal.get("spy_price", 0):.2f}

<b>Picks:</b>
{picks_text}"""
    
    else:
        text = f"""<b>HOLD Signal ({today})</b>
Market: DOWN (Momentum: {signal.get("market_momentum", 0):.4f})
SPY: ${signal.get("spy_price", 0):.2f}

매수 신호 없음"""
    
    return send_message(text)


def send_portfolio(portfolio_value):
    """
    포트폴리오 상태 메시지
    """
    today = datetime.now().strftime("%Y-%m-%d")
    
    total = portfolio_value.get("total", 0)
    cash = portfolio_value.get("cash", 0)
    stocks = portfolio_value.get("stocks", 0)
    
    # Holdings 정보
    holdings_text = ""
    for h in portfolio_value.get("holdings_detail", []):
        symbol = h.get("symbol", "")
        shares = h.get("shares", 0)
        return_pct = h.get("profit_loss_pct", 0)
        holdings_text += f"- {symbol}: {shares}주 ({return_pct:+.2f}%)\n"
    
    if not holdings_text:
        holdings_text = "- 보유 종목 없음\n"
    
    text = f"""<b>Portfolio Status ({today})</b>
Total: ${total:,.2f}
Cash: ${cash:,.2f}
Stocks: ${stocks:,.2f}

<b>Holdings:</b>
{holdings_text}"""
    
    return send_message(text)


def send_daily_summary(daily_data, portfolio_value, signal=None, strategy=None, period="Daily"):
    """
    일일/주간 요약 메시지

    Args:
        signal: 시그널 dict (없으면 생략)
        period: "Daily" 또는 "Weekly"
    """
    date = daily_data.get("date", datetime.now().strftime("%Y-%m-%d"))

    total = portfolio_value.get("total", 0)
    daily_return = daily_data.get("daily_return_pct", 0)
    spy_return = daily_data.get("spy_return_pct", 0)
    alpha = daily_data.get("alpha", 0)

    # Holdings 정보
    holdings_text = ""
    for h in portfolio_value.get("holdings_detail", []):
        symbol = h.get("symbol", "")
        shares = h.get("shares", 0)
        return_pct = h.get("profit_loss_pct", 0)
        holdings_text += f"- {symbol}: {shares}주 ({return_pct:+.2f}%)\n"

    if not holdings_text:
        holdings_text = "- 보유 종목 없음\n"

    # 시그널 섹션
    signal_label = "오늘의 시그널 (참고용)" if period == "Daily" else "이번 주 신호"
    signal_text = ""
    if signal:
        sig_type = signal.get("signal", "")
        sig_date = signal.get("date", "")
        if hasattr(sig_date, "strftime"):
            sig_date = sig_date.strftime("%Y-%m-%d")
        market_trend = signal.get("market_trend", "")
        market_momentum = signal.get("market_momentum", 0)
        picks = signal.get("picks", [])
        allocations = signal.get("allocations", [])
        scores = signal.get("scores", [])

        if sig_type == "BUY":
            picks_text = ""
            for i, sym in enumerate(picks):
                alloc = allocations[i] if i < len(allocations) else 0
                score = scores[i] if i < len(scores) else 0
                picks_text += f"  {i+1}. {sym} ({alloc*100:.0f}%) score:{score:.4f}\n"
            signal_text = (
                f"\n<b>📊 {signal_label}</b>\n"
                f"기준일: {sig_date} | {market_trend} ({market_momentum:.4f})\n"
                f"신호: BUY\n{picks_text}"
            )
        else:
            rank_text = ""
            for i, sym in enumerate(picks):
                score = scores[i] if i < len(scores) else 0
                rank_text += f"  {i+1}위 : {sym} ({score:.4f})\n"
            rank_block = f"\n종목 분석\n{rank_text}" if rank_text else ""
            signal_text = (
                f"\n<b>📊 {signal_label}</b>\n"
                f"기준일: {sig_date} | {market_trend} ({market_momentum:.4f})\n"
                f"신호: {sig_type}\n{rank_block}"
            )

    strategy_label = f" [{strategy}]" if strategy else ""
    text = f"""<b>{period} Summary{strategy_label} ({date})</b>
Portfolio: ${total:,.2f}
{period}: {daily_return:+.2f}%
SPY: {spy_return:+.2f}%
Alpha: {alpha:+.2f}%
{signal_text}
<b>Holdings:</b>
{holdings_text}"""

    return send_message(text)


def send_stop_loss(alerts):
    """
    손절 알림 메시지
    """
    if not alerts:
        return True
    
    today = datetime.now().strftime("%Y-%m-%d")
    
    # 손절 종목 정보
    alerts_text = ""
    for a in alerts:
        symbol = a.get("symbol", "")
        return_pct = a.get("return_pct", 0)
        profit_loss = a.get("profit_loss", 0)
        alerts_text += f"- {symbol}: {return_pct:.2f}% (${profit_loss:+,.2f})\n"
    
    text = f"""<b>Stop Loss Alert! ({today})</b>
{alerts_text}
손절 필요!"""
    
    return send_message(text)


def send_trade_signal():
    """
    매매 신호 발생 안내
    """
    today = datetime.now().strftime("%Y-%m-%d")
    
    text = f"""<b>매매 시그널 발생! ({today})</b>
매매 후 Trades에 기록해주세요."""
    
    return send_message(text)


def send_rebalancing(rebalancing, total_capital=None):
    """
    리밸런싱 안내 메시지
    
    Args:
        rebalancing: calculate_rebalancing() 결과
        total_capital: 현재 총 자본금
    """
    today = datetime.now().strftime("%Y-%m-%d")
    
    actions = rebalancing.get("actions", [])
    
    if not actions:
        text = f"""<b>📊 리밸런싱 ({today})</b>
{rebalancing.get("message", "매매 없음")}"""
        return send_message(text)
    
    # 액션별 분류
    sells = [a for a in actions if a["action"] == "SELL"]
    reduces = [a for a in actions if a["action"] == "REDUCE"]
    holds = [a for a in actions if a["action"] == "HOLD"]
    adds = [a for a in actions if a["action"] == "ADD"]
    buys = [a for a in actions if a["action"] == "BUY"]
    
    # 메시지 구성
    capital_str = f"${total_capital:,.0f}" if total_capital else ""
    
    text = f"""<b>📊 리밸런싱 ({today})</b>
Capital: {capital_str}

"""
    
    # 매도
    if sells:
        text += "<b>🔴 매도 (전량)</b>\n"
        for a in sells:
            profit = a.get("profit_pct", 0)
            text += f"• {a['symbol']} {a['shares']}주 @ ${a['price']} ({profit:+.1f}%)\n"
        text += "\n"
    
    # 비중 축소
    if reduces:
        text += "<b>🟠 비중 축소</b>\n"
        for a in reduces:
            text += f"• {a['symbol']} -{a['shares']}주 @ ${a['price']}\n"
        text += "\n"
    
    # 유지
    if holds:
        text += "<b>⚪ 유지</b>\n"
        for a in holds:
            text += f"• {a['symbol']} {a['shares']}주\n"
        text += "\n"
    
    # 추가 매수
    if adds:
        text += "<b>🟢 추가 매수</b>\n"
        for a in adds:
            text += f"• {a['symbol']} +{a['shares']}주 @ ${a['price']}\n"
        text += "\n"
    
    # 신규 매수
    if buys:
        text += "<b>🟢 신규 매수</b>\n"
        for a in buys:
            text += f"• {a['symbol']} {a['shares']}주 @ ${a['price']}\n"
        text += "\n"
    
    # 요약
    summary = rebalancing.get("summary", {})
    text += f"""<b>💰 요약</b>
매도 금액: ${summary.get('total_sell', 0):,.0f}
매수 금액: ${summary.get('total_buy', 0):,.0f}
현금 변화: ${summary.get('net_cash_change', 0):+,.0f}"""
    
    return send_message(text)


def send_hybrid_signal(signal, total_capital, weight_momentum=None, weight_ai=None, label="Hybrid"):
    """
    Hybrid 매매 신호 메시지 (모멘텀 포맷 기반)

    Args:
        label: 전략 이름 ("Hybrid" 또는 "Hybrid_New")
    """
    today = datetime.now().strftime("%Y-%m-%d")

    weights_str = ""
    if weight_momentum is not None and weight_ai is not None:
        weights_str = f"\n가중치: M{weight_momentum*100:.0f}% + AI{weight_ai*100:.0f}%"

    market_momentum = signal.get("market_momentum", 0)
    spy_price = signal.get("spy_price", signal.get("prices", {}).get("SPY", 0))

    if signal.get("market_filter", False):
        weights_line = f"가중치: M{weight_momentum*100:.0f}% + AI{weight_ai*100:.0f}%\n" if weight_momentum is not None and weight_ai is not None else ""
        text = f"""<b>📊 {label} 리밸런싱 ({today})</b>
{weights_line}HOLD 신호 - 매매 없음"""
    else:
        picks_text = ""
        for i, (symbol, score) in enumerate(zip(signal["picks"], signal["scores"])):
            price = signal["prices"].get(symbol, 0)
            alloc = signal["allocations"][i]
            shares = int(total_capital * alloc / price) if price > 0 else 0
            picks_text += f"{i+1}. {symbol} ({score:.4f}) ({alloc*100:.0f}%) - ${price:.2f} ({shares}주)\n"

        text = f"""<b>{label} BUY Signal ({today})</b>
Capital: ${total_capital:,.2f}
Market: UP (Momentum: {market_momentum:.4f})
SPY: ${spy_price:.2f}{weights_str}

<b>Picks:</b>
{picks_text}"""

    return send_message(text)


def send_hybrid_portfolio(portfolio_value, label="Hybrid"):
    """
    Hybrid 포트폴리오 상태 메시지
    """
    today = datetime.now().strftime("%Y-%m-%d")

    total = portfolio_value.get("total", 0)
    cash = portfolio_value.get("cash", 0)
    stocks = portfolio_value.get("stocks", 0)

    holdings_text = ""
    for h in portfolio_value.get("holdings_detail", []):
        symbol = h.get("symbol", "")
        shares = h.get("shares", 0)
        return_pct = h.get("profit_loss_pct", 0)
        holdings_text += f"- {symbol}: {shares}주 ({return_pct:+.2f}%)\n"

    if not holdings_text:
        holdings_text = "- 보유 종목 없음\n"

    text = f"""<b>{label} Portfolio Status ({today})</b>
Total: ${total:,.2f}
Cash: ${cash:,.2f}
Stocks: ${stocks:,.2f}

<b>Holdings:</b>
{holdings_text}"""

    return send_message(text)


def send_hybrid_rebalancing(rebalancing, total_capital, signal=None, weight_momentum=None, weight_ai=None, label="Hybrid"):
    """
    Hybrid 리밸런싱 메시지
    """
    today = datetime.now().strftime("%Y-%m-%d")

    weights_str = ""
    if weight_momentum is not None and weight_ai is not None:
        weights_str = f"\n가중치: M{weight_momentum*100:.0f}% + AI{weight_ai*100:.0f}%"

    actions = rebalancing.get("actions", [])

    if not actions:
        text = f"""<b>📊 {label} 리밸런싱 ({today})</b>
Capital: ${total_capital:,.0f}{weights_str}
{rebalancing.get("message", "매매 없음")}"""
        return send_message(text)

    # 선정 종목 (점수 포함)
    picks_text = ""
    if signal:
        for i, (symbol, score) in enumerate(zip(signal["picks"], signal["scores"])):
            price = signal["prices"].get(symbol, 0)
            picks_text += f"{i+1}. {symbol}: 점수 {score:.4f}, 가격 ${price:.2f}\n"

    # 액션별 분류
    sells = [a for a in actions if a["action"] == "SELL"]
    reduces = [a for a in actions if a["action"] == "REDUCE"]
    holds = [a for a in actions if a["action"] == "HOLD"]
    adds = [a for a in actions if a["action"] == "ADD"]
    buys = [a for a in actions if a["action"] == "BUY"]

    capital_str = f"${total_capital:,.0f}"

    text = f"<b>📊 {label} 리밸런싱 ({today})</b>\n"
    text += f"Capital: {capital_str}{weights_str}\n"

    if picks_text:
        text += f"\n<b>선정 종목:</b>\n{picks_text}"

    if sells:
        text += "\n<b>🔴 매도 (전량)</b>\n"
        for a in sells:
            profit = a.get("return_pct", 0)
            text += f"• {a['symbol']} {a['shares']}주 @ ${a['price']:.2f} ({profit:+.1f}%)\n"

    if reduces:
        text += "\n<b>🟠 비중 축소</b>\n"
        for a in reduces:
            text += f"• {a['symbol']} -{a['shares']}주 @ ${a['price']:.2f}\n"

    if holds:
        text += "\n<b>⚪ 유지</b>\n"
        for a in holds:
            text += f"• {a['symbol']} {a['shares']}주\n"

    if adds:
        text += "\n<b>🟢 추가 매수</b>\n"
        for a in adds:
            text += f"• {a['symbol']} +{a['shares']}주 @ ${a['price']:.2f}\n"

    if buys:
        text += "\n<b>🟢 신규 매수</b>\n"
        for a in buys:
            text += f"• {a['symbol']} {a['shares']}주 @ ${a['price']:.2f}\n"

    summary = rebalancing.get("summary", {})
    text += f"""\n<b>💰 요약</b>
매도 금액: ${summary.get('total_sell', 0):,.0f}
매수 금액: ${summary.get('total_buy', 0):,.0f}
현금 변화: ${summary.get('net_cash_change', 0):+,.0f}"""

    return send_message(text)


def send_trades(trades):
    """
    거래 실행 메시지 (수동 매매라 현재 미사용)

    [데드 코드] 이 함수는 현재 paper_trading.py, hybrid_trading.py 어디서도 호출되지 않음.
    수동 매매 기록 시 사용하려고 정의했으나, 자동화 시스템에서는 send_rebalancing()으로 대체됨.
    """
    if not trades:
        return True
    
    today = datetime.now().strftime("%Y-%m-%d")
    
    trades_text = ""
    for t in trades:
        action = t.get("action", "")
        symbol = t.get("symbol", "")
        shares = t.get("shares", 0)
        price = t.get("price", 0)
        trades_text += f"- {action}: {symbol} {shares}주 @ ${price:.2f}\n"
    
    text = f"""<b>Trades ({today})</b>
{trades_text}"""
    
    return send_message(text)


def send_error(error_msg):
    """
    에러 메시지
    """
    today = datetime.now().strftime("%Y-%m-%d")
    
    text = f"""<b>Error ({today})</b>
{error_msg}"""
    
    return send_message(text)


# ============================================
# Test
# ============================================

if __name__ == "__main__":
    print("Telegram Test")
    result = send_message("Test message from Stock Trading Bot")
    print(f"Result: {result}")