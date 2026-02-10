# ============================================
# src/kis.py
# Korea Investment Securities API Wrapper
# 해외주식 매매용 (모의투자/실전투자)
# ============================================

import requests
import json
from datetime import datetime, timedelta

from src.config import (
    get_kis_url,
    get_kis_credentials,
    get_exchange,
    KIS_HTS_ID,
    KIS_ACCOUNT_PROD,
    KIS_MODE
)

# ============================================
# Token Management
# ============================================

_token = None
_token_expires = None


def get_token():
    global _token, _token_expires
    
    if _token and _token_expires:
        if datetime.now() < _token_expires:
            return _token
    
    creds = get_kis_credentials()
    url = f"{get_kis_url()}/oauth2/tokenP"
    
    headers = {"content-type": "application/json"}
    body = {
        "grant_type": "client_credentials",
        "appkey": creds["app_key"],
        "appsecret": creds["app_secret"]
    }
    
    try:
        res = requests.post(url, headers=headers, data=json.dumps(body))
        res.raise_for_status()
        data = res.json()
        
        _token = data.get("access_token")
        expires_in = int(data.get("expires_in", 86400))
        _token_expires = datetime.now() + timedelta(seconds=expires_in - 60)
        
        print(f"Token acquired (expires in {expires_in}s)")
        return _token
    except Exception as e:
        print(f"Token error: {e}")
        return None


def _get_headers(tr_id):
    creds = get_kis_credentials()
    token = get_token()
    
    return {
        "Content-Type": "application/json; charset=utf-8",
        "authorization": f"Bearer {token}",
        "appkey": creds["app_key"],
        "appsecret": creds["app_secret"],
        "tr_id": tr_id,
        "custtype": "P"
    }


# ============================================
# Get Current Price
# ============================================

def get_price(symbol, exchange=None):
    if exchange is None:
        exchange = get_exchange(symbol)
    
    tr_id = "HHDFS76200200"
    url = f"{get_kis_url()}/uapi/overseas-price/v1/quotations/price"
    
    headers = _get_headers(tr_id)
    params = {
        "AUTH": "",
        "EXCD": exchange,
        "SYMB": symbol
    }
    
    try:
        res = requests.get(url, headers=headers, params=params)
        res.raise_for_status()
        data = res.json()
        
        if data.get("rt_cd") == "0":
            output = data.get("output", {})
            return {
                "symbol": symbol,
                "price": float(output.get("last", 0)),
                "change": float(output.get("diff", 0)),
                "change_pct": float(output.get("rate", 0)),
                "volume": int(float(output.get("tvol", 0))),
                "open": float(output.get("open", 0)),
                "high": float(output.get("high", 0)),
                "low": float(output.get("low", 0)),
                "time": output.get("ordy", "")
            }
        else:
            print(f"Price error [{symbol}]: {data.get('msg1')}")
            return None
    except Exception as e:
        print(f"Price error [{symbol}]: {e}")
        return None


def get_prices(symbols):
    result = {}
    for symbol in symbols:
        price_data = get_price(symbol)
        if price_data:
            result[symbol] = price_data["price"]
    return result


# ============================================
# Get Balance
# ============================================

def get_balance():
    creds = get_kis_credentials()
    
    # 모의투자: VTTS3012R, 실전: TTTS3012R
    tr_id = "VTTS3012R" if KIS_MODE == "paper" else "TTTS3012R"
    url = f"{get_kis_url()}/uapi/overseas-stock/v1/trading/inquire-balance"
    
    headers = _get_headers(tr_id)
    params = {
        "CANO": creds["account"],
        "ACNT_PRDT_CD": KIS_ACCOUNT_PROD,
        "OVRS_EXCG_CD": "NASD",
        "TR_CRCY_CD": "USD",
        "CTX_AREA_FK200": "",
        "CTX_AREA_NK200": ""
    }
    
    try:
        res = requests.get(url, headers=headers, params=params)
        res.raise_for_status()
        data = res.json()
        
        if data.get("rt_cd") == "0":
            output1 = data.get("output1", [])
            output2 = data.get("output2", {})
            
            holdings = []
            for item in output1:
                qty = int(float(item.get("ovrs_cblc_qty", 0)))
                if qty > 0:
                    holdings.append({
                        "symbol": item.get("ovrs_pdno", ""),
                        "name": item.get("ovrs_item_name", ""),
                        "shares": qty,
                        "avg_price": float(item.get("pchs_avg_pric", 0)),
                        "current_price": float(item.get("now_pric2", 0)),
                        "value": float(item.get("ovrs_stck_evlu_amt", 0)),
                        "profit_loss": float(item.get("frcr_evlu_pfls_amt", 0)),
                        "profit_loss_pct": float(item.get("evlu_pfls_rt", 0))
                    })
            
            cash_usd = 0
            total_eval = 0
            if output2:
                cash_usd = float(output2.get("frcr_dncl_amt_2", 0))
                total_eval = float(output2.get("tot_evlu_pfls_amt", 0))
            
            return {
                "holdings": holdings,
                "cash": cash_usd,
                "total_value": cash_usd + sum(h["value"] for h in holdings),
                "total_profit_loss": total_eval
            }
        else:
            print(f"Balance error: {data.get('msg1')}")
            return None
    except Exception as e:
        print(f"Balance error: {e}")
        return None


# ============================================
# Buy Order
# ============================================

def buy(symbol, qty, price=0, exchange=None):
    if exchange is None:
        exchange = get_exchange(symbol)
    
    creds = get_kis_credentials()
    
    # 모의투자: VTTT1002U, 실전: TTTT1002U
    tr_id = "VTTT1002U" if KIS_MODE == "paper" else "TTTT1002U"
    url = f"{get_kis_url()}/uapi/overseas-stock/v1/trading/order"
    
    headers = _get_headers(tr_id)
    
    body = {
        "CANO": creds["account"],
        "ACNT_PRDT_CD": KIS_ACCOUNT_PROD,
        "OVRS_EXCG_CD": exchange,
        "PDNO": symbol,
        "ORD_QTY": str(int(qty)),
        "OVRS_ORD_UNPR": str(price) if price > 0 else "0",
        "ORD_SVR_DVSN_CD": "0",
        "ORD_DVSN": "00" if price > 0 else "01"  # 00: 지정가, 01: 시장가
    }
    
    try:
        res = requests.post(url, headers=headers, data=json.dumps(body))
        res.raise_for_status()
        data = res.json()
        
        if data.get("rt_cd") == "0":
            output = data.get("output", {})
            print(f"BUY order success: {symbol} x {qty}")
            return {
                "success": True,
                "order_no": output.get("ODNO", ""),
                "order_time": output.get("ORD_TMD", ""),
                "symbol": symbol,
                "qty": qty,
                "price": price,
                "action": "BUY"
            }
        else:
            msg = data.get("msg1", "Unknown error")
            print(f"BUY order failed [{symbol}]: {msg}")
            return {
                "success": False,
                "message": msg
            }
    except Exception as e:
        print(f"BUY order error [{symbol}]: {e}")
        return {
            "success": False,
            "message": str(e)
        }


# ============================================
# Sell Order
# ============================================

def sell(symbol, qty, price=0, exchange=None):
    if exchange is None:
        exchange = get_exchange(symbol)
    
    creds = get_kis_credentials()
    
    # 모의투자: VTTT1001U, 실전: TTTT1006U
    tr_id = "VTTT1001U" if KIS_MODE == "paper" else "TTTT1006U"
    url = f"{get_kis_url()}/uapi/overseas-stock/v1/trading/order"
    
    headers = _get_headers(tr_id)
    
    body = {
        "CANO": creds["account"],
        "ACNT_PRDT_CD": KIS_ACCOUNT_PROD,
        "OVRS_EXCG_CD": exchange,
        "PDNO": symbol,
        "ORD_QTY": str(int(qty)),
        "OVRS_ORD_UNPR": str(price) if price > 0 else "0",
        "ORD_SVR_DVSN_CD": "0",
        "ORD_DVSN": "00" if price > 0 else "01"
    }
    
    try:
        res = requests.post(url, headers=headers, data=json.dumps(body))
        res.raise_for_status()
        data = res.json()
        
        if data.get("rt_cd") == "0":
            output = data.get("output", {})
            print(f"SELL order success: {symbol} x {qty}")
            return {
                "success": True,
                "order_no": output.get("ODNO", ""),
                "order_time": output.get("ORD_TMD", ""),
                "symbol": symbol,
                "qty": qty,
                "price": price,
                "action": "SELL"
            }
        else:
            msg = data.get("msg1", "Unknown error")
            print(f"SELL order failed [{symbol}]: {msg}")
            return {
                "success": False,
                "message": msg
            }
    except Exception as e:
        print(f"SELL order error [{symbol}]: {e}")
        return {
            "success": False,
            "message": str(e)
        }


# ============================================
# Get Order History
# ============================================

def get_orders(start_date=None, end_date=None):
    creds = get_kis_credentials()
    
    if start_date is None:
        start_date = datetime.now().strftime("%Y%m%d")
    if end_date is None:
        end_date = datetime.now().strftime("%Y%m%d")
    
    # 모의투자: VTTS3035R, 실전: TTTS3035R
    tr_id = "VTTS3035R" if KIS_MODE == "paper" else "TTTS3035R"
    url = f"{get_kis_url()}/uapi/overseas-stock/v1/trading/inquire-ccnl"
    
    headers = _get_headers(tr_id)
    params = {
        "CANO": creds["account"],
        "ACNT_PRDT_CD": KIS_ACCOUNT_PROD,
        "PDNO": "%",
        "ORD_STRT_DT": start_date,
        "ORD_END_DT": end_date,
        "SLL_BUY_DVSN": "00",
        "CCLD_NCCS_DVSN": "00",
        "OVRS_EXCG_CD": "%",
        "SORT_SQN": "DS",
        "ORD_DT": "",
        "ORD_GNO_BRNO": "",
        "ODNO": "",
        "CTX_AREA_FK200": "",
        "CTX_AREA_NK200": ""
    }
    
    try:
        res = requests.get(url, headers=headers, params=params)
        res.raise_for_status()
        data = res.json()
        
        if data.get("rt_cd") == "0":
            orders = []
            for item in data.get("output", []):
                orders.append({
                    "order_no": item.get("odno", ""),
                    "symbol": item.get("pdno", ""),
                    "action": "BUY" if item.get("sll_buy_dvsn_cd") == "02" else "SELL",
                    "qty": int(float(item.get("ft_ord_qty", 0))),
                    "price": float(item.get("ft_ord_unpr3", 0)),
                    "filled_qty": int(float(item.get("ft_ccld_qty", 0))),
                    "filled_price": float(item.get("ft_ccld_unpr3", 0)),
                    "status": item.get("ord_dvsn_name", ""),
                    "time": item.get("ord_tmd", "")
                })
            return orders
        else:
            print(f"Orders error: {data.get('msg1')}")
            return []
    except Exception as e:
        print(f"Orders error: {e}")
        return []


# ============================================
# Check if Market is Open
# ============================================

def is_market_open():
    now = datetime.now()
    
    # 미국 동부시간 기준 (한국시간 -14시간, 서머타임 -13시간)
    # 장 시간: 9:30 ~ 16:00 EST
    # 한국시간: 23:30 ~ 06:00 (겨울), 22:30 ~ 05:00 (여름)
    
    # 주말 체크
    if now.weekday() >= 5:
        return False
    
    hour = now.hour
    
    # 대략적인 체크 (한국시간 기준)
    # 겨울: 23:30 ~ 06:00
    # 여름: 22:30 ~ 05:00
    if 22 <= hour or hour < 7:
        return True
    
    return False


# ============================================
# Print Summary
# ============================================

def print_balance():
    balance = get_balance()
    
    if balance is None:
        print("Failed to get balance")
        return
    
    print("=" * 60)
    print("Portfolio Balance")
    print("=" * 60)
    print(f"Cash: ${balance['cash']:,.2f}")
    print(f"Total Value: ${balance['total_value']:,.2f}")
    print(f"Profit/Loss: ${balance['total_profit_loss']:,.2f}")
    print()
    
    if balance["holdings"]:
        print(f"Holdings ({len(balance['holdings'])})")
        print("-" * 60)
        for h in balance["holdings"]:
            print(f"  {h['symbol']:6} | {h['shares']:>4} shares | "
                  f"Avg: ${h['avg_price']:>8.2f} | "
                  f"Now: ${h['current_price']:>8.2f} | "
                  f"{h['profit_loss_pct']:>+6.2f}%")
        print("-" * 60)
    else:
        print("No holdings")
    print("=" * 60)


# ============================================
# Test
# ============================================

if __name__ == "__main__":
    print("KIS API Test")
    print("=" * 60)
    print(f"Mode: {KIS_MODE}")
    print(f"URL: {get_kis_url()}")
    print()
    
    # Token test
    token = get_token()
    if token:
        print(f"Token: {token[:30]}...")
    else:
        print("Token failed!")
        exit(1)
    
    # Balance test
    print("\n=== Balance ===")
    print_balance()
    
    # Price test
    print("\n=== Price ===")
    for symbol in ["AAPL", "MSFT", "NVDA"]:
        price = get_price(symbol)
        if price:
            print(f"{symbol}: ${price['price']:.2f} ({price['change_pct']:+.2f}%)")