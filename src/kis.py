# ============================================
# 파일명: src/kis.py
# 설명: 한국투자증권(KIS) Open API 연동 모듈
#
# 기능:
# - OAuth 인증 및 토큰 관리
# - 현재가 조회 (국내주식)
# - 주식 매수/매도 주문
# - 주문 내역 조회
# - 잔고 조회
# - 일별 시세 조회
#
# 참고:
# - 실전 도메인: https://openapi.koreainvestment.com:9443
# - 모의 도메인: https://openapivts.koreainvestment.com:29443
# - API 문서: https://apiportal.koreainvestment.com
# ============================================

import os
import json
import time
import requests
import pandas as pd
from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv()


# ============================================
# 설정
# ============================================

# 환경변수에서 API 키 불러오기
KIS_APP_KEY = os.environ.get("KIS_APP_KEY", "")
KIS_APP_SECRET = os.environ.get("KIS_APP_SECRET", "")
KIS_ACCOUNT_NO = os.environ.get("KIS_ACCOUNT_NO", "")          # 계좌번호 (8자리)
KIS_ACCOUNT_PRODUCT = os.environ.get("KIS_ACCOUNT_PRODUCT", "01")  # 계좌상품코드

# 모의투자 / 실전투자 전환
# True: 모의투자 (기본값, 안전), False: 실전투자
KIS_PAPER_TRADE = os.environ.get("KIS_PAPER_TRADE", "true").lower() == "true"

# 도메인 설정
DOMAIN_REAL = "https://openapi.koreainvestment.com:9443"
DOMAIN_PAPER = "https://openapivts.koreainvestment.com:29443"

BASE_URL = DOMAIN_PAPER if KIS_PAPER_TRADE else DOMAIN_REAL


# ============================================
# 토큰 관리
# ============================================

_token_cache = {
    "access_token": "",
    "token_type": "Bearer",
    "expires_at": 0,
}


def get_access_token():
    """
    OAuth 액세스 토큰 발급

    토큰은 발급 후 약 24시간 유효합니다.
    캐시를 사용하여 불필요한 재발급을 방지합니다.

    Returns:
        str: 액세스 토큰
    """
    global _token_cache

    # 캐시된 토큰이 유효하면 재사용
    if _token_cache["access_token"] and time.time() < _token_cache["expires_at"]:
        return _token_cache["access_token"]

    if not KIS_APP_KEY or not KIS_APP_SECRET:
        print("KIS API 키가 설정되지 않았습니다")
        print("환경변수 KIS_APP_KEY, KIS_APP_SECRET을 확인하세요")
        return ""

    url = f"{BASE_URL}/oauth2/tokenP"
    headers = {"content-type": "application/json"}
    body = {
        "grant_type": "client_credentials",
        "appkey": KIS_APP_KEY,
        "appsecret": KIS_APP_SECRET,
    }

    try:
        response = requests.post(url, headers=headers, json=body)
        response.raise_for_status()
        data = response.json()

        _token_cache["access_token"] = data["access_token"]
        _token_cache["token_type"] = data.get("token_type", "Bearer")
        # 만료 시간: 발급 후 23시간으로 설정 (여유분 1시간)
        _token_cache["expires_at"] = time.time() + 23 * 3600

        print("KIS 토큰 발급 완료")
        return _token_cache["access_token"]

    except requests.exceptions.RequestException as e:
        print(f"KIS 토큰 발급 실패: {e}")
        return ""


def get_hashkey(data):
    """
    해시키 발급 (POST 요청 시 필요)

    Args:
        data: 요청 body (dict)

    Returns:
        str: 해시키
    """
    url = f"{BASE_URL}/uapi/hashkey"
    headers = {
        "content-type": "application/json",
        "appkey": KIS_APP_KEY,
        "appsecret": KIS_APP_SECRET,
    }

    try:
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()
        return response.json().get("HASH", "")
    except requests.exceptions.RequestException as e:
        print(f"해시키 발급 실패: {e}")
        return ""


# ============================================
# 공통 헤더 생성
# ============================================

def _make_headers(tr_id, is_post=False):
    """
    API 요청 공통 헤더 생성

    Args:
        tr_id: 거래ID (예: "FHKST01010100")
        is_post: POST 요청 여부

    Returns:
        dict: 요청 헤더
    """
    token = get_access_token()
    if not token:
        return {}

    headers = {
        "content-type": "application/json; charset=utf-8",
        "authorization": f"Bearer {token}",
        "appkey": KIS_APP_KEY,
        "appsecret": KIS_APP_SECRET,
        "tr_id": tr_id,
    }

    return headers


# ============================================
# 현재가 조회
# ============================================

def get_current_price(stock_code, market="J"):
    """
    국내주식 현재가 조회

    Args:
        stock_code: 종목코드 (예: "005930" = 삼성전자)
        market: 시장구분 ("J"=주식/ETF/ETN, "W"=ELW)

    Returns:
        dict: {
            'stock_code': 종목코드,
            'stock_name': 종목명,
            'price': 현재가,
            'change': 전일대비,
            'change_pct': 등락률,
            'volume': 거래량,
            'high': 고가,
            'low': 저가,
            'open': 시가,
            'market_cap': 시가총액,
        } 또는 None
    """
    tr_id = "FHKST01010100"
    headers = _make_headers(tr_id)
    if not headers:
        return None

    url = f"{BASE_URL}/uapi/domestic-stock/v1/quotations/inquire-price"
    params = {
        "FID_COND_MRKT_DIV_CODE": market,
        "FID_INPUT_ISCD": stock_code,
    }

    try:
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()
        data = response.json()

        if data.get("rt_cd") != "0":
            print(f"현재가 조회 실패 [{stock_code}]: {data.get('msg1', '')}")
            return None

        output = data.get("output", {})
        return {
            "stock_code": stock_code,
            "stock_name": output.get("hts_kor_isnm", ""),
            "price": int(output.get("stck_prpr", 0)),
            "change": int(output.get("prdy_vrss", 0)),
            "change_pct": float(output.get("prdy_ctrt", 0)),
            "volume": int(output.get("acml_vol", 0)),
            "high": int(output.get("stck_hgpr", 0)),
            "low": int(output.get("stck_lwpr", 0)),
            "open": int(output.get("stck_oprc", 0)),
            "market_cap": int(output.get("hts_avls", 0)),
        }

    except requests.exceptions.RequestException as e:
        print(f"현재가 조회 오류 [{stock_code}]: {e}")
        return None


def get_multiple_prices(stock_codes, market="J"):
    """
    여러 종목의 현재가 일괄 조회

    Args:
        stock_codes: 종목코드 리스트
        market: 시장구분

    Returns:
        dict: {종목코드: 가격정보 dict, ...}
    """
    prices = {}
    for code in stock_codes:
        result = get_current_price(code, market)
        if result:
            prices[code] = result
        time.sleep(0.1)  # API 호출 간격 (초당 10회 제한)
    return prices


# ============================================
# 일별 시세 조회
# ============================================

def get_daily_price(stock_code, period="D", adj_price=True):
    """
    국내주식 일별 시세 조회 (최근 100일)

    Args:
        stock_code: 종목코드
        period: 기간구분 ("D"=일, "W"=주, "M"=월)
        adj_price: 수정주가 적용 여부

    Returns:
        DataFrame: date, open, high, low, close, volume
    """
    tr_id = "FHKST01010400"
    headers = _make_headers(tr_id)
    if not headers:
        return pd.DataFrame()

    url = f"{BASE_URL}/uapi/domestic-stock/v1/quotations/inquire-daily-price"
    params = {
        "FID_COND_MRKT_DIV_CODE": "J",
        "FID_INPUT_ISCD": stock_code,
        "FID_PERIOD_DIV_CODE": period,
        "FID_ORG_ADJ_PRC": "0" if adj_price else "1",
    }

    try:
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()
        data = response.json()

        if data.get("rt_cd") != "0":
            print(f"일별시세 조회 실패 [{stock_code}]: {data.get('msg1', '')}")
            return pd.DataFrame()

        records = []
        for item in data.get("output", []):
            records.append({
                "date": item.get("stck_bsop_date", ""),
                "open": int(item.get("stck_oprc", 0)),
                "high": int(item.get("stck_hgpr", 0)),
                "low": int(item.get("stck_lwpr", 0)),
                "close": int(item.get("stck_clpr", 0)),
                "volume": int(item.get("acml_vol", 0)),
            })

        if not records:
            return pd.DataFrame()

        df = pd.DataFrame(records)
        df["date"] = pd.to_datetime(df["date"], format="%Y%m%d")
        df["symbol"] = stock_code
        df = df.sort_values("date").reset_index(drop=True)

        return df

    except requests.exceptions.RequestException as e:
        print(f"일별시세 조회 오류 [{stock_code}]: {e}")
        return pd.DataFrame()


def get_daily_price_range(stock_code, start_date, end_date=None):
    """
    기간 지정 일별 시세 조회 (일별주가 기간 조회)

    Args:
        stock_code: 종목코드
        start_date: 시작일 ("YYYYMMDD")
        end_date: 종료일 ("YYYYMMDD", None이면 오늘)

    Returns:
        DataFrame: date, open, high, low, close, volume
    """
    if end_date is None:
        end_date = datetime.now().strftime("%Y%m%d")

    tr_id = "FHKST03010100"
    headers = _make_headers(tr_id)
    if not headers:
        return pd.DataFrame()

    url = f"{BASE_URL}/uapi/domestic-stock/v1/quotations/inquire-daily-itemchartprice"
    params = {
        "FID_COND_MRKT_DIV_CODE": "J",
        "FID_INPUT_ISCD": stock_code,
        "FID_INPUT_DATE_1": start_date,
        "FID_INPUT_DATE_2": end_date,
        "FID_PERIOD_DIV_CODE": "D",
        "FID_ORG_ADJ_PRC": "0",
    }

    try:
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()
        data = response.json()

        if data.get("rt_cd") != "0":
            print(f"기간시세 조회 실패 [{stock_code}]: {data.get('msg1', '')}")
            return pd.DataFrame()

        records = []
        for item in data.get("output2", []):
            date_str = item.get("stck_bsop_date", "")
            if not date_str:
                continue
            records.append({
                "date": date_str,
                "open": int(item.get("stck_oprc", 0)),
                "high": int(item.get("stck_hgpr", 0)),
                "low": int(item.get("stck_lwpr", 0)),
                "close": int(item.get("stck_clpr", 0)),
                "volume": int(item.get("acml_vol", 0)),
            })

        if not records:
            return pd.DataFrame()

        df = pd.DataFrame(records)
        df["date"] = pd.to_datetime(df["date"], format="%Y%m%d")
        df["symbol"] = stock_code
        df = df.sort_values("date").reset_index(drop=True)

        return df

    except requests.exceptions.RequestException as e:
        print(f"기간시세 조회 오류 [{stock_code}]: {e}")
        return pd.DataFrame()


# ============================================
# 주문 (매수 / 매도)
# ============================================

def buy_stock(stock_code, quantity, price=0, order_type="00"):
    """
    국내주식 매수 주문

    Args:
        stock_code: 종목코드 (예: "005930")
        quantity: 주문수량
        price: 주문가격 (지정가일 때). 0이면 시장가
        order_type: 주문유형
            "00" = 지정가
            "01" = 시장가
            "02" = 조건부지정가
            "03" = 최유리지정가
            "04" = 최우선지정가

    Returns:
        dict: {
            'success': 성공여부,
            'order_no': 주문번호,
            'message': 메시지,
        }
    """
    # 모의투자 / 실전투자 TR_ID 분기
    if KIS_PAPER_TRADE:
        tr_id = "VTTC0802U"  # 모의투자 매수
    else:
        tr_id = "TTTC0802U"  # 실전투자 매수

    return _place_order(tr_id, stock_code, quantity, price, order_type, "매수")


def sell_stock(stock_code, quantity, price=0, order_type="00"):
    """
    국내주식 매도 주문

    Args:
        stock_code: 종목코드
        quantity: 주문수량
        price: 주문가격 (지정가일 때). 0이면 시장가
        order_type: 주문유형 ("00"=지정가, "01"=시장가, ...)

    Returns:
        dict: {
            'success': 성공여부,
            'order_no': 주문번호,
            'message': 메시지,
        }
    """
    if KIS_PAPER_TRADE:
        tr_id = "VTTC0801U"  # 모의투자 매도
    else:
        tr_id = "TTTC0801U"  # 실전투자 매도

    return _place_order(tr_id, stock_code, quantity, price, order_type, "매도")


def _place_order(tr_id, stock_code, quantity, price, order_type, action_name):
    """
    주문 실행 (내부 함수)

    Args:
        tr_id: 거래ID
        stock_code: 종목코드
        quantity: 수량
        price: 가격
        order_type: 주문유형
        action_name: "매수" 또는 "매도" (로그용)

    Returns:
        dict: 주문 결과
    """
    headers = _make_headers(tr_id, is_post=True)
    if not headers:
        return {"success": False, "order_no": "", "message": "인증 실패"}

    if not KIS_ACCOUNT_NO:
        return {"success": False, "order_no": "", "message": "계좌번호 미설정"}

    # 시장가 주문이면 가격은 0
    if order_type == "01":
        price = 0

    body = {
        "CANO": KIS_ACCOUNT_NO,
        "ACNT_PRDT_CD": KIS_ACCOUNT_PRODUCT,
        "PDNO": stock_code,
        "ORD_DVSN": order_type,
        "ORD_QTY": str(int(quantity)),
        "ORD_UNPR": str(int(price)),
    }

    # 해시키 추가
    hashkey = get_hashkey(body)
    if hashkey:
        headers["hashkey"] = hashkey

    url = f"{BASE_URL}/uapi/domestic-stock/v1/trading/order-cash"

    try:
        response = requests.post(url, headers=headers, json=body)
        response.raise_for_status()
        data = response.json()

        if data.get("rt_cd") == "0":
            order_no = data.get("output", {}).get("ODNO", "")
            print(f"{action_name} 주문 성공: {stock_code} / {quantity}주 / 주문번호: {order_no}")
            return {
                "success": True,
                "order_no": order_no,
                "message": data.get("msg1", "주문 성공"),
            }
        else:
            msg = data.get("msg1", "알 수 없는 오류")
            print(f"{action_name} 주문 실패 [{stock_code}]: {msg}")
            return {"success": False, "order_no": "", "message": msg}

    except requests.exceptions.RequestException as e:
        print(f"{action_name} 주문 오류 [{stock_code}]: {e}")
        return {"success": False, "order_no": "", "message": str(e)}


# ============================================
# 주문 취소 / 정정
# ============================================

def cancel_order(order_no, stock_code, quantity, order_type="00"):
    """
    주문 취소

    Args:
        order_no: 원주문번호
        stock_code: 종목코드
        quantity: 취소수량 (전량이면 원래 수량)
        order_type: 원주문유형

    Returns:
        dict: 취소 결과
    """
    if KIS_PAPER_TRADE:
        tr_id = "VTTC0803U"  # 모의투자 취소
    else:
        tr_id = "TTTC0803U"  # 실전투자 취소

    headers = _make_headers(tr_id, is_post=True)
    if not headers:
        return {"success": False, "message": "인증 실패"}

    body = {
        "CANO": KIS_ACCOUNT_NO,
        "ACNT_PRDT_CD": KIS_ACCOUNT_PRODUCT,
        "KRX_FWDG_ORD_ORGNO": "",
        "ORGN_ODNO": order_no,
        "ORD_DVSN": order_type,
        "RVSE_CNCL_DVSN_CD": "02",  # 02=취소
        "ORD_QTY": str(int(quantity)),
        "ORD_UNPR": "0",
        "QTY_ALL_ORD_YN": "Y",  # 전량취소
    }

    hashkey = get_hashkey(body)
    if hashkey:
        headers["hashkey"] = hashkey

    url = f"{BASE_URL}/uapi/domestic-stock/v1/trading/order-rvsecncl"

    try:
        response = requests.post(url, headers=headers, json=body)
        response.raise_for_status()
        data = response.json()

        if data.get("rt_cd") == "0":
            print(f"주문 취소 성공: {order_no}")
            return {"success": True, "message": data.get("msg1", "취소 성공")}
        else:
            msg = data.get("msg1", "알 수 없는 오류")
            print(f"주문 취소 실패: {msg}")
            return {"success": False, "message": msg}

    except requests.exceptions.RequestException as e:
        print(f"주문 취소 오류: {e}")
        return {"success": False, "message": str(e)}


# ============================================
# 잔고 조회
# ============================================

def get_balance():
    """
    계좌 잔고 조회

    보유 종목, 수량, 평균단가, 현재가, 수익률 등을 조회합니다.

    Returns:
        dict: {
            'holdings': [
                {
                    'stock_code': 종목코드,
                    'stock_name': 종목명,
                    'quantity': 보유수량,
                    'avg_price': 평균매입가,
                    'current_price': 현재가,
                    'eval_amount': 평가금액,
                    'profit_loss': 평가손익,
                    'profit_pct': 수익률,
                }, ...
            ],
            'total_eval': 총평가금액,
            'total_profit': 총평가손익,
            'total_profit_pct': 총수익률,
            'cash': 예수금,
            'total_asset': 총자산,
        } 또는 None
    """
    if KIS_PAPER_TRADE:
        tr_id = "VTTC8434R"  # 모의투자 잔고
    else:
        tr_id = "TTTC8434R"  # 실전투자 잔고

    headers = _make_headers(tr_id)
    if not headers:
        return None

    url = f"{BASE_URL}/uapi/domestic-stock/v1/trading/inquire-balance"
    params = {
        "CANO": KIS_ACCOUNT_NO,
        "ACNT_PRDT_CD": KIS_ACCOUNT_PRODUCT,
        "AFHR_FLPR_YN": "N",
        "OFL_YN": "",
        "INQR_DVSN": "02",
        "UNPR_DVSN": "01",
        "FUND_STTL_ICLD_YN": "N",
        "FNCG_AMT_AUTO_RDPT_YN": "N",
        "PRCS_DVSN": "01",
        "CTX_AREA_FK100": "",
        "CTX_AREA_NK100": "",
    }

    try:
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()
        data = response.json()

        if data.get("rt_cd") != "0":
            print(f"잔고 조회 실패: {data.get('msg1', '')}")
            return None

        # 보유 종목 목록
        holdings = []
        for item in data.get("output1", []):
            quantity = int(item.get("hldg_qty", 0))
            if quantity <= 0:
                continue
            avg_price = float(item.get("pchs_avg_pric", 0))
            current_price = int(item.get("prpr", 0))
            eval_amount = int(item.get("evlu_amt", 0))
            profit_loss = int(item.get("evlu_pfls_amt", 0))
            profit_pct = float(item.get("evlu_pfls_rt", 0))

            holdings.append({
                "stock_code": item.get("pdno", ""),
                "stock_name": item.get("prdt_name", ""),
                "quantity": quantity,
                "avg_price": avg_price,
                "current_price": current_price,
                "eval_amount": eval_amount,
                "profit_loss": profit_loss,
                "profit_pct": profit_pct,
            })

        # 계좌 요약 (output2의 첫 번째 항목)
        summary = {}
        output2 = data.get("output2", [])
        if output2:
            s = output2[0]
            summary = {
                "total_eval": int(s.get("scts_evlu_amt", 0)),
                "total_profit": int(s.get("evlu_pfls_smtl_amt", 0)),
                "cash": int(s.get("dnca_tot_amt", 0)),
                "total_asset": int(s.get("tot_evlu_amt", 0)),
            }
            total_purchase = int(s.get("pchs_amt_smtl_amt", 0))
            if total_purchase > 0:
                summary["total_profit_pct"] = round(
                    summary["total_profit"] / total_purchase * 100, 2
                )
            else:
                summary["total_profit_pct"] = 0.0

        return {
            "holdings": holdings,
            **summary,
        }

    except requests.exceptions.RequestException as e:
        print(f"잔고 조회 오류: {e}")
        return None


# ============================================
# 주문 내역 조회
# ============================================

def get_orders(start_date=None, end_date=None, status="all"):
    """
    주문 내역 조회

    Args:
        start_date: 조회 시작일 ("YYYYMMDD", None이면 오늘)
        end_date: 조회 종료일 ("YYYYMMDD", None이면 오늘)
        status: "all"=전체, "filled"=체결, "pending"=미체결

    Returns:
        list[dict]: 주문 내역 리스트
    """
    if start_date is None:
        start_date = datetime.now().strftime("%Y%m%d")
    if end_date is None:
        end_date = datetime.now().strftime("%Y%m%d")

    if KIS_PAPER_TRADE:
        tr_id = "VTTC8001R"  # 모의투자 주문내역
    else:
        tr_id = "TTTC8001R"  # 실전투자 주문내역

    headers = _make_headers(tr_id)
    if not headers:
        return []

    # 주문상태 구분
    inqr_dvsn = "00"  # 전체
    if status == "filled":
        inqr_dvsn = "01"
    elif status == "pending":
        inqr_dvsn = "02"

    url = f"{BASE_URL}/uapi/domestic-stock/v1/trading/inquire-daily-ccld"
    params = {
        "CANO": KIS_ACCOUNT_NO,
        "ACNT_PRDT_CD": KIS_ACCOUNT_PRODUCT,
        "INQR_STRT_DT": start_date,
        "INQR_END_DT": end_date,
        "SLL_BUY_DVSN_CD": "00",  # 매도매수 전체
        "INQR_DVSN": "00",
        "PDNO": "",
        "CCLD_DVSN": inqr_dvsn,
        "ORD_GNO_BRNO": "",
        "ODNO": "",
        "INQR_DVSN_3": "00",
        "INQR_DVSN_1": "",
        "CTX_AREA_FK100": "",
        "CTX_AREA_NK100": "",
    }

    try:
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()
        data = response.json()

        if data.get("rt_cd") != "0":
            print(f"주문내역 조회 실패: {data.get('msg1', '')}")
            return []

        orders = []
        for item in data.get("output1", []):
            order_no = item.get("odno", "")
            if not order_no:
                continue
            orders.append({
                "order_no": order_no,
                "order_date": item.get("ord_dt", ""),
                "order_time": item.get("ord_tmd", ""),
                "stock_code": item.get("pdno", ""),
                "stock_name": item.get("prdt_name", ""),
                "action": "매수" if item.get("sll_buy_dvsn_cd") == "02" else "매도",
                "order_qty": int(item.get("ord_qty", 0)),
                "order_price": int(item.get("ord_unpr", 0)),
                "filled_qty": int(item.get("tot_ccld_qty", 0)),
                "filled_price": int(item.get("avg_prvs", 0)),
                "status": "체결" if int(item.get("tot_ccld_qty", 0)) > 0 else "미체결",
            })

        return orders

    except requests.exceptions.RequestException as e:
        print(f"주문내역 조회 오류: {e}")
        return []


# ============================================
# 예수금 조회
# ============================================

def get_cash_balance():
    """
    예수금(주문가능현금) 조회

    Returns:
        dict: {
            'deposit': 예수금총액,
            'available': 주문가능금액,
            'd2_deposit': D+2 예수금,
        } 또는 None
    """
    if KIS_PAPER_TRADE:
        tr_id = "VTTC8908R"  # 모의투자
    else:
        tr_id = "TTTC8908R"  # 실전투자

    headers = _make_headers(tr_id)
    if not headers:
        return None

    url = f"{BASE_URL}/uapi/domestic-stock/v1/trading/inquire-psbl-order"
    params = {
        "CANO": KIS_ACCOUNT_NO,
        "ACNT_PRDT_CD": KIS_ACCOUNT_PRODUCT,
        "PDNO": "005930",  # 아무 종목 (필수 파라미터)
        "ORD_UNPR": "0",
        "ORD_DVSN": "01",
        "CMA_EVLU_AMT_ICLD_YN": "Y",
        "OVRS_ICLD_YN": "Y",
    }

    try:
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()
        data = response.json()

        if data.get("rt_cd") != "0":
            print(f"예수금 조회 실패: {data.get('msg1', '')}")
            return None

        output = data.get("output", {})
        return {
            "deposit": int(output.get("dnca_tot_amt", 0)),
            "available": int(output.get("nrcvb_buy_amt", 0)),
            "d2_deposit": int(output.get("prvs_rcdl_excc_amt", 0)),
        }

    except requests.exceptions.RequestException as e:
        print(f"예수금 조회 오류: {e}")
        return None


# ============================================
# 편의 함수
# ============================================

def print_balance():
    """
    잔고를 보기 좋게 출력합니다.
    """
    balance = get_balance()
    if not balance:
        print("잔고 조회에 실패했습니다")
        return

    print("=" * 60)
    print("KIS 계좌 잔고")
    print("=" * 60)
    mode = "모의투자" if KIS_PAPER_TRADE else "실전투자"
    print(f"모드: {mode}")
    print(f"계좌: {KIS_ACCOUNT_NO}-{KIS_ACCOUNT_PRODUCT}")
    print("-" * 60)
    print(f"총자산: {balance.get('total_asset', 0):>15,}원")
    print(f"예수금: {balance.get('cash', 0):>15,}원")
    print(f"주식평가: {balance.get('total_eval', 0):>15,}원")
    print(f"평가손익: {balance.get('total_profit', 0):>+15,}원 ({balance.get('total_profit_pct', 0):+.2f}%)")
    print("-" * 60)

    holdings = balance.get("holdings", [])
    if holdings:
        print(f"\n보유종목 ({len(holdings)}개)")
        print("-" * 60)
        for h in holdings:
            name = h["stock_name"][:8]
            print(
                f"  {h['stock_code']} {name:8} | "
                f"{h['quantity']:>5}주 | "
                f"평균 {h['avg_price']:>10,.0f} | "
                f"현재 {h['current_price']:>10,} | "
                f"{h['profit_pct']:>+7.2f}%"
            )
        print("-" * 60)
    else:
        print("\n보유종목 없음")

    print("=" * 60)


def print_orders(days=0):
    """
    주문 내역을 출력합니다.

    Args:
        days: 조회 기간 (0=오늘, 7=최근 7일 등)
    """
    if days > 0:
        start = (datetime.now() - timedelta(days=days)).strftime("%Y%m%d")
    else:
        start = None

    orders = get_orders(start_date=start)
    if not orders:
        print("주문 내역이 없습니다")
        return

    print("=" * 60)
    print(f"주문 내역 ({len(orders)}건)")
    print("=" * 60)
    for o in orders:
        print(
            f"  [{o['status']}] {o['order_date']} {o['order_time'][:4]} | "
            f"{o['action']} {o['stock_name'][:8]:8} | "
            f"{o['order_qty']}주 @ {o['order_price']:,}원"
        )
    print("=" * 60)


# ============================================
# 시장 상태 확인
# ============================================

def is_market_open():
    """
    현재 국내 주식시장 개장 여부 확인

    Returns:
        bool: 개장 중이면 True
    """
    now = datetime.now()

    # 주말 체크
    if now.weekday() >= 5:
        return False

    # 장 시간: 09:00 ~ 15:30
    market_open = now.replace(hour=9, minute=0, second=0, microsecond=0)
    market_close = now.replace(hour=15, minute=30, second=0, microsecond=0)

    return market_open <= now <= market_close


# ============================================
# 테스트
# ============================================

if __name__ == "__main__":
    print("=" * 60)
    print("KIS API 연동 테스트")
    print("=" * 60)

    mode = "모의투자" if KIS_PAPER_TRADE else "실전투자"
    print(f"모드: {mode}")
    print(f"도메인: {BASE_URL}")
    print(f"계좌: {KIS_ACCOUNT_NO or '(미설정)'}")
    print()

    # 인증 테스트
    token = get_access_token()
    if token:
        print(f"토큰: {token[:20]}...")
    else:
        print("토큰 발급 실패 - 환경변수를 확인하세요")
        print("  KIS_APP_KEY: {'설정됨' if KIS_APP_KEY else '미설정'}")
        print("  KIS_APP_SECRET: {'설정됨' if KIS_APP_SECRET else '미설정'}")
        print("  KIS_ACCOUNT_NO: {'설정됨' if KIS_ACCOUNT_NO else '미설정'}")
        exit(1)

    # 현재가 조회 테스트 (삼성전자)
    print("\n[삼성전자 현재가 조회]")
    price = get_current_price("005930")
    if price:
        print(f"  종목명: {price['stock_name']}")
        print(f"  현재가: {price['price']:,}원")
        print(f"  등락률: {price['change_pct']:+.2f}%")
        print(f"  거래량: {price['volume']:,}")

    # 잔고 조회 테스트
    print("\n[잔고 조회]")
    print_balance()
