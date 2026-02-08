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

# ============================================
# Send Message
# ============================================

def send_message(text, parse_mode="HTML"):
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

def send_signal(signal):
    if signal["signal"] == "BUY":
        picks_text = ""
        for i, (symbol, score, alloc) in enumerate(zip(signal["picks"], signal["scores"], signal["allocations"])):
            price = signal.get("prices", {}).get(symbol, 0)
            picks_text += f"  {i+1}. <b>{symbol}</b> | {score:.4f} | {alloc*100:.0f}% | ${price}\n"
        
        text = f"""
<b>BUY Signal</b>
Date: {signal.get("date", "")}

<b>Picks:</b>
{picks_text}
Market Momentum: {signal.get("market_momentum", 0):.4f}
SPY: ${signal.get("spy_price", 0)}
"""
    else:
        text = f"""
<b>HOLD Signal</b>
Date: {signal.get("date", "")}

Market downtrend - No buy
Market Momentum: {signal.get("market_momentum", 0):.4f}
"""
    
    return send_message(text)


def send_trades(trades):
    if not trades:
        return True
    
    trades_text = ""
    for t in trades:
        action = t.get("action", "")
        symbol = t.get("symbol", "")
        shares = t.get("shares", 0)
        price = t.get("price", 0)
        return_pct = t.get("return_pct", 0)
        
        if action in ["SELL", "STOP_LOSS"]:
            trades_text += f"  {action}: {symbol} | {shares} shares | ${price} | {return_pct:+.2f}%\n"
        else:
            trades_text += f"  {action}: {symbol} | {shares} shares | ${price}\n"
    
    text = f"""
<b>Trades Executed</b>
{datetime.now().strftime("%Y-%m-%d %H:%M")}

{trades_text}
Total: {len(trades)} trades
"""
    
    return send_message(text)


def send_portfolio(portfolio_value):
    total = portfolio_value.get("total", 0)
    cash = portfolio_value.get("cash", 0)
    stocks = portfolio_value.get("stocks", 0)
    
    holdings_text = ""
    for h in portfolio_value.get("holdings_detail", []):
        holdings_text += f"  {h['symbol']}: {h['shares']} shares | {h['return_pct']:+.2f}%\n"
    
    if not holdings_text:
        holdings_text = "  No holdings\n"
    
    initial = 2000
    total_return = (total - initial) / initial * 100
    
    text = f"""
<b>Portfolio Status</b>
{datetime.now().strftime("%Y-%m-%d %H:%M")}

Total: <b>${total:,.2f}</b>
Cash: ${cash:,.2f}
Stocks: ${stocks:,.2f}

Return: <b>{total_return:+.2f}%</b>

<b>Holdings:</b>
{holdings_text}"""
    
    return send_message(text)


def send_stop_loss(trades):
    if not trades:
        return True
    
    trades_text = ""
    for t in trades:
        symbol = t.get("symbol", "")
        shares = t.get("shares", 0)
        price = t.get("price", 0)
        return_pct = t.get("return_pct", 0)
        trades_text += f"  {symbol} | {shares} shares | ${price} | {return_pct:.2f}%\n"
    
    text = f"""
<b>STOP LOSS Executed</b>
{datetime.now().strftime("%Y-%m-%d %H:%M")}

{trades_text}
"""
    
    return send_message(text)


def send_daily_summary(daily_data, portfolio_value):
    total = portfolio_value.get("total", 0)
    daily_return = daily_data.get("daily_return_pct", 0)
    spy_return = daily_data.get("spy_return_pct", 0)
    alpha = daily_data.get("alpha", 0)
    
    initial = 2000
    total_return = (total - initial) / initial * 100
    
    text = f"""
<b>Daily Summary</b>
{daily_data.get("date", "")}

Total: <b>${total:,.2f}</b>
Total Return: <b>{total_return:+.2f}%</b>

Today: {daily_return:+.2f}%
SPY: {spy_return:+.2f}%
Alpha: {alpha:+.2f}%
"""
    
    return send_message(text)


def send_error(error_msg):
    text = f"""
<b>Error</b>
{datetime.now().strftime("%Y-%m-%d %H:%M")}

{error_msg}
"""
    
    return send_message(text)


# ============================================
# Test
# ============================================

if __name__ == "__main__":
    print("Telegram Test")
    
    # Test message
    result = send_message("Test message from Stock Trading Bot")
    print(f"Result: {result}")