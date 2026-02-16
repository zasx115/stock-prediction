# ============================================
# src/telegram.py
# Telegram Notification Module
# ============================================

import requests
import os
from datetime import datetime

# ============================================
# Settings
# ============================================

BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")
INITIAL_CAPITAL = 2000

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


def send_daily_summary(daily_data, portfolio_value):
    """
    일일 요약 메시지
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
    
    text = f"""<b>Daily Summary ({date})</b>
Portfolio: ${total:,.2f}
Daily: {daily_return:+.2f}%
SPY: {spy_return:+.2f}%
Alpha: {alpha:+.2f}%

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
매매 후 Holdings에 기록해주세요."""
    
    return send_message(text)


def send_trades(trades):
    """
    거래 실행 메시지 (수동 매매라 현재 미사용)
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