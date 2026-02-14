# ============================================
# src/kis.py
# Korea Investment Securities API Wrapper
# 한투 공식 라이브러리 함수 래핑
# ============================================

import os
import sys

# 한투 공식 라이브러리 경로 추가
KIS_API_PATH = os.environ.get("KIS_API_PATH", "/content/kis-api/examples_user")
sys.path.insert(0, KIS_API_PATH)
sys.path.insert(0, os.path.join(KIS_API_PATH, "overseas_stock"))

# 설정
from config import (
    KIS_MODE,
    KIS_ACCOUNT_PROD,
    get_kis_credentials,
    get_exchange
)

# ============================================
# Initialize (한투 공식 인증)
# ============================================

_initialized = False

def init():
    """
    한투 공식 라이브러리 인증 초기화
    """
    global _initialized
    
    if _initialized:
        return True
    
    try:
        import kis_auth as ka
        
        # 모의투자: vps, 실전투자: prod
        svr = "vps" if KIS_MODE == "paper" else "prod"
        ka.auth(svr=svr, product=KIS_ACCOUNT_PROD)
        
        _initialized = True
        print(f"✅ KIS Auth initialized (mode={KIS_MODE})")
        return True
    except Exception as e:
        print(f"❌ KIS Auth failed: {e}")
        return False


# ============================================
# Get Balance
# ============================================

def get_balance():
    """
    해외주식 잔고 조회
    """
    if not init():
        return None
    
    try:
        from overseas_stock_functions import inquire_balance
        
        creds = get_kis_credentials()
        env_dv = "demo" if KIS_MODE == "paper" else "real"
        
        result = inquire_balance(
            env_dv=env_dv,
            cano=creds["account"],
            acnt_prdt_cd=KIS_ACCOUNT_PROD,
            ovrs_excg_cd="NASD",
            tr_crcy_cd="USD"
        )
        
        # DataFrame을 dict로 변환
        holdings_df, summary_df = result
        
        holdings = []
        if not holdings_df.empty:
            for _, row in holdings_df.iterrows():
                holdings.append({
                    "symbol": row.get("ovrs_pdno", ""),
                    "name": row.get("ovrs_item_name", ""),
                    "shares": int(float(row.get("ovrs_cblc_qty", 0) or 0)),
                    "avg_price": float(row.get("pchs_avg_pric", 0) or 0),
                    "current_price": float(row.get("now_pric2", 0) or 0),
                    "value": float(row.get("ovrs_stck_evlu_amt", 0) or 0),
                    "profit_loss": float(row.get("frcr_evlu_pfls_amt", 0) or 0),
                    "profit_loss_pct": float(row.get("evlu_pfls_rt", 0) or 0)
                })
        
        # 요약 정보
        cash = 0
        total_value = 0
        if not summary_df.empty:
            row = summary_df.iloc[0]
            cash = float(row.get("frcr_dncl_amt_2", 0) or row.get("ovrs_ord_psbl_amt", 0) or 0)
        
        stocks_value = sum(h["value"] for h in holdings)
        total_value = cash + stocks_value
        
        return {
            "holdings": holdings,
            "cash": cash,
            "stocks_value": stocks_value,
            "total_value": total_value
        }
        
    except Exception as e:
        print(f"Balance error: {e}")
        return None


# ============================================
# Get Price
# ============================================

def get_price(symbol, exchange=None):
    """
    해외주식 현재가 조회
    """
    if not init():
        return None
    
    if exchange is None:
        exchange = get_exchange(symbol)
    
    try:
        from overseas_stock_functions import inquire_price
        
        env_dv = "demo" if KIS_MODE == "paper" else "real"
        
        result = inquire_price(
            env_dv=env_dv,
            excd=exchange,
            symb=symbol
        )
        
        if result is not None and not result.empty:
            row = result.iloc[0]
            return {
                "symbol": symbol,
                "price": float(row.get("last", 0) or 0),
                "change": float(row.get("diff", 0) or 0),
                "change_pct": float(row.get("rate", 0) or 0),
                "volume": int(float(row.get("tvol", 0) or 0)),
                "open": float(row.get("open", 0) or 0),
                "high": float(row.get("high", 0) or 0),
                "low": float(row.get("low", 0) or 0)
            }
        return None
        
    except Exception as e:
        print(f"Price error [{symbol}]: {e}")
        return None


def get_prices(symbols):
    """
    여러 종목 현재가 조회
    """
    result = {}
    for symbol in symbols:
        price_data = get_price(symbol)
        if price_data and price_data["price"] > 0:
            result[symbol] = price_data["price"]
    return result


# ============================================
# Buy Order
# ============================================

def buy(symbol, qty, price=0, exchange=None):
    """
    해외주식 매수 주문
    """
    if not init():
        return {"success": False, "message": "Auth failed"}
    
    if exchange is None:
        exchange = get_exchange(symbol)
    
    try:
        from overseas_stock_functions import order_buy
        
        creds = get_kis_credentials()
        env_dv = "demo" if KIS_MODE == "paper" else "real"
        
        # 주문 구분: 00=지정가, 01=시장가
        ord_dvsn = "00" if price > 0 else "01"
        
        result = order_buy(
            env_dv=env_dv,
            cano=creds["account"],
            acnt_prdt_cd=KIS_ACCOUNT_PROD,
            ovrs_excg_cd=exchange,
            pdno=symbol,
            ord_qty=str(int(qty)),
            ovrs_ord_unpr=str(price) if price > 0 else "0",
            ord_dvsn=ord_dvsn
        )
        
        if result is not None:
            print(f"BUY order: {symbol} x {qty}")
            return {
                "success": True,
                "symbol": symbol,
                "qty": qty,
                "price": price,
                "action": "BUY",
                "result": result
            }
        else:
            return {"success": False, "message": "Order failed"}
        
    except Exception as e:
        print(f"BUY error [{symbol}]: {e}")
        return {"success": False, "message": str(e)}


# ============================================
# Sell Order
# ============================================

def sell(symbol, qty, price=0, exchange=None):
    """
    해외주식 매도 주문
    """
    if not init():
        return {"success": False, "message": "Auth failed"}
    
    if exchange is None:
        exchange = get_exchange(symbol)
    
    try:
        from overseas_stock_functions import order_sell
        
        creds = get_kis_credentials()
        env_dv = "demo" if KIS_MODE == "paper" else "real"
        
        ord_dvsn = "00" if price > 0 else "01"
        
        result = order_sell(
            env_dv=env_dv,
            cano=creds["account"],
            acnt_prdt_cd=KIS_ACCOUNT_PROD,
            ovrs_excg_cd=exchange,
            pdno=symbol,
            ord_qty=str(int(qty)),
            ovrs_ord_unpr=str(price) if price > 0 else "0",
            ord_dvsn=ord_dvsn
        )
        
        if result is not None:
            print(f"SELL order: {symbol} x {qty}")
            return {
                "success": True,
                "symbol": symbol,
                "qty": qty,
                "price": price,
                "action": "SELL",
                "result": result
            }
        else:
            return {"success": False, "message": "Order failed"}
        
    except Exception as e:
        print(f"SELL error [{symbol}]: {e}")
        return {"success": False, "message": str(e)}


# ============================================
# Print Balance
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
    
    # 잔고 조회
    print("\n=== Balance ===")
    print_balance()
    
    # 현재가 조회
    print("\n=== Price ===")
    for symbol in ["AAPL", "MSFT", "NVDA"]:
        price = get_price(symbol)
        if price:
            print(f"{symbol}: ${price['price']:.2f} ({price['change_pct']:+.2f}%)")
