# ============================================
# src/kis.py
# Korea Investment Securities API Wrapper
# 해외주식 매매용 (모의투자/실전투자)
# ============================================

import requests
import json
from datetime import datetime, timedelta

from config import (
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
    """
    접근 토큰 발급
    """
    global _token, _token_expires
    
    # 기존 토큰 유효하면 재사용
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
    """
    API 호출용 헤더 생성
    """
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


def _get_tr_id(base_id, is_buy=True):
    """
    모의투자/실전투자에 따른 tr_id 반환
    
    해외주식 tr_id:
    - 실전 매수: TTTT1002U
    - 실전 매도: TTTT1006U
    - 모의 매수: VTTT1002U
    - 모의 매도: VTTT1001U
    - 실전 잔고: TTTS3012R
    - 모의 잔고: VTTS3012R
    - 실전 체결: TTTS3035R
    - 모의 체결: VTTS3035R
    """
    if KIS_MODE == "paper":
        # 모의투자
        if base_id == "balance":
            return "VTTS3012R"
        elif base_id == "buy":
            return "VTTT1002U"
        elif base_id == "sell":
            return "VTTT1001U"
        elif base_id == "orders":
            return "VTTS3035R"
        elif base_id == "price":
            return "HHDFS76200200"
    else:
        # 실전투자
        if base_id == "balance":
            return "TTTS3012R"
        elif base_id == "buy":
            return "TTTT1002U"
        elif base_id == "sell":
            return "TTTT1006U"
        elif base_id == "orders":
            return "TTTS3035R"
        elif base_id == "price":
            return "HHDFS76200200"
    
    return base_id


# ============================================
# Get Current Price
# ============================================

def get_price(symbol, exchange=None):
    """
    해외주식 현재가 조회
    
    Args:
        symbol: 종목코드 (예: "AAPL")
        exchange: 거래소 (None이면 자동판단)
    
    Returns:
        dict: 시세 정보 또는 None
    """
    if exchange is None:
        exchange = get_exchange(symbol)
    
    tr_id = _get_tr_id("price")
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
                "price": float(output.get("last", 0) or 0),
                "change": float(output.get("diff", 0) or 0),
                "change_pct": float(output.get("rate", 0) or 0),
                "volume": int(float(output.get("tvol", 0) or 0)),
                "open": float(output.get("open", 0) or 0),
                "high": float(output.get("high", 0) or 0),
                "low": float(output.get("low", 0) or 0),
                "time": output.get("ordy", "")
            }
        else:
            print(f"Price error [{symbol}]: {data.get('msg1')}")
            return None
    except Exception as e:
        print(f"Price error [{symbol}]: {e}")
        return None


def get_prices(symbols):
    """
    여러 종목 현재가 조회
    
    Args:
        symbols: 종목코드 리스트
    
    Returns:
        dict: {symbol: price, ...}
    """
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
    """
    해외주식 잔고 조회
    
    Returns:
        dict: 잔고 정보 또는 None
    """
    creds = get_kis_credentials()
    tr_id = _get_tr_id("balance")
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
            
            # 보유 종목
            holdings = []
            for item in output1:
                qty = int(float(item.get("ovrs_cblc_qty", 0) or 0))
                if qty > 0:
                    avg_price = float(item.get("pchs_avg_pric", 0) or 0)
                    current_price = float(item.get("now_pric2", 0) or 0)
                    value = float(item.get("ovrs_stck_evlu_amt", 0) or 0)
                    profit_loss = float(item.get("frcr_evlu_pfls_amt", 0) or 0)
                    profit_loss_pct = float(item.get("evlu_pfls_rt", 0) or 0)
                    
                    holdings.append({
                        "symbol": item.get("ovrs_pdno", ""),
                        "name": item.get("ovrs_item_name", ""),
                        "shares": qty,
                        "avg_price": avg_price,
                        "current_price": current_price,
                        "value": value,
                        "profit_loss": profit_loss,
                        "profit_loss_pct": profit_loss_pct
                    })
            
            # 현금 및 총 평가
            cash_usd = 0
            total_eval = 0
            if output2:
                cash_usd = float(output2.get("frcr_dncl_amt_2", 0) or 0)
                total_eval = float(output2.get("tot_evlu_pfls_amt", 0) or 0)
            
            # 총 자산 계산
            stocks_value = sum(h["value"] for h in holdings)
            total_value = cash_usd + stocks_value
            
            return {
                "holdings": holdings,
                "cash": cash_usd,
                "stocks_value": stocks_value,
                "total_value": total_value,
                "total_profit_loss": total_eval
            }
        else:
            print(f"Balance error: {data.get('msg1')} ({data.get('msg_cd')})")
            return None
    except Exception as e:
        print(f"Balance error: {e}")
        return None


# ============================================
# Buy Order
# ============================================

def buy(symbol, qty, price=0, exchange=None):
    """
    해외주식 매수 주문
    
    Args:
        symbol: 종목코드
        qty: 수량
        price: 가격 (0이면 시장가)
        exchange: 거래소
    
    Returns:
        dict: 주문 결과
    """
    if exchange is None:
        exchange = get_exchange(symbol)
    
    creds = get_kis_credentials()
    tr_id = _get_tr_id("buy")
    url = f"{get_kis_url()}/uapi/overseas-stock/v1/trading/order"
    
    headers = _get_headers(tr_id)
    
    # 주문 구분: 00=지정가, 01=시장가
    ord_dvsn = "00" if price > 0 else "01"
    
    body = {
        "CANO": creds["account"],
        "ACNT_PRDT_CD": KIS_ACCOUNT_PROD,
        "OVRS_EXCG_CD": exchange,
        "PDNO": symbol,
        "ORD_QTY": str(int(qty)),
        "OVRS_ORD_UNPR": str(price) if price > 0 else "0",
        "ORD_SVR_DVSN_CD": "0",
        "ORD_DVSN": ord_dvsn
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
    """
    해외주식 매도 주문
    
    Args:
        symbol: 종목코드
        qty: 수량
        price: 가격 (0이면 시장가)
        exchange: 거래소
    
    Returns:
        dict: 주문 결과
    """
    if exchange is None:
        exchange = get_exchange(symbol)
    
    creds = get_kis_credentials()
    tr_id = _get_tr_id("sell")
    url = f"{get_kis_url()}/uapi/overseas-stock/v1/trading/order"
    
    headers = _get_headers(tr_id)
    
    ord_dvsn = "00" if price > 0 else "01"
    
    body = {
        "CANO": creds["account"],
        "ACNT_PRDT_CD": KIS_ACCOUNT_PROD,
        "OVRS_EXCG_CD": exchange,
        "PDNO": symbol,
        "ORD_QTY": str(int(qty)),
        "OVRS_ORD_UNPR": str(price) if price > 0 else "0",
        "ORD_SVR_DVSN_CD": "0",
        "ORD_DVSN": ord_dvsn
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
    """
    주문 내역 조회
    
    Args:
        start_date: 시작일 (YYYYMMDD)
        end_date: 종료일 (YYYYMMDD)
    
    Returns:
        list: 주문 내역
    """
    creds = get_kis_credentials()
    
    if start_date is None:
        start_date = datetime.now().strftime("%Y%m%d")
    if end_date is None:
        end_date = datetime.now().strftime("%Y%m%d")
    
    tr_id = _get_tr_id("orders")
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
                    "qty": int(float(item.get("ft_ord_qty", 0) or 0)),
                    "price": float(item.get("ft_ord_unpr3", 0) or 0),
                    "filled_qty": int(float(item.get("ft_ccld_qty", 0) or 0)),
                    "filled_price": float(item.get("ft_ccld_unpr3", 0) or 0),
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
    """
    미국 장 오픈 여부 확인 (대략적)
    """
    now = datetime.now()
    
    # 주말 체크
    if now.weekday() >= 5:
        return False
    
    # 한국시간 기준 대략적 체크
    # 겨울: 23:30 ~ 06:00
    # 여름: 22:30 ~ 05:00
    hour = now.hour
    if 22 <= hour or hour < 7:
        return True
    
    return False


# ============================================
# Print Summary
# ============================================

def print_balance():
    """
    잔고 출력
    """
    balance = get_balance()
    
    if balance is None:
        print("Failed to get balance")
        return
    
    print("=" * 60)
    print("Portfolio Balance")
    print("=" * 60)
    print(f"Cash: ${balance['cash']:,.2f}")
    print(f"Stocks: ${balance['stocks_value']:,.2f}")
    print(f"Total: ${balance['total_value']:,.2f}")
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
