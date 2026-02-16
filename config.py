# ============================================
# 파일명: config.py
# 설명: 프로젝트 전체 설정 파일
# 작성일: 2025-01-29
# 수정일: 2026-02-08 (한투 API 추가)
# ============================================
# 
# 이 파일에서 모든 설정을 관리합니다.
# API 키 같은 민감한 정보는 GitHub Secrets 또는
# 환경변수로 관리하고, 여기서는 불러와서 사용합니다.
# ============================================

import os
from dotenv import load_dotenv

# .env 파일에서 환경변수 불러오기 (로컬 테스트용)
load_dotenv()


# ============================================
# 1. 기본 설정
# ============================================

PROJECT_NAME = "stock-prediction"
DEBUG_MODE = True


# ============================================
# 2. 주식 관련 설정
# ============================================

# ----- 시장 지표 (벤치마크 + AI 학습용) -----
MARKET_SYMBOLS = {
    "SPY": "S&P 500 지수",
    "^VIX": "변동성 지수",
}

# ----- 섹터 ETF 11개 -----
SECTOR_ETFS = {
    "XLK": "기술",
    "XLF": "금융", 
    "XLE": "에너지",
    "XLV": "헬스케어",
    "XLY": "경기소비재",
    "XLP": "필수소비재",
    "XLI": "산업재",
    "XLB": "소재",
    "XLRE": "부동산",
    "XLC": "통신",
    "XLU": "유틸리티",
}

# ----- S&P 500 백업 리스트 (Wikipedia 실패 시) -----
SP500_BACKUP = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "BRK-B", "UNH", "JNJ",
    "JPM", "V", "PG", "XOM", "MA", "HD", "CVX", "MRK", "ABBV", "LLY",
    "PEP", "KO", "COST", "AVGO", "WMT", "MCD", "CSCO", "ACN", "TMO", "ABT",
    "DHR", "NEE", "VZ", "ADBE", "CRM", "NKE", "PM", "TXN", "CMCSA", "ORCL",
    "AMD", "INTC", "QCOM", "HON", "UPS", "LOW", "IBM", "CAT", "BA", "GE",
    "RTX", "SPGI", "AMGN", "MS", "GS", "BLK", "AXP", "ISRG", "MDLZ", "GILD",
    "ADI", "BKNG", "SYK", "VRTX", "MMC", "LRCX", "REGN", "PLD", "SCHW", "CB",
    "CI", "ZTS", "NOW", "MO", "TMUS", "SO", "DUK", "BDX", "AON", "APD",
    "CME", "CL", "EQIX", "ITW", "NOC", "SHW", "PNC", "USB", "ICE", "MCO",
    "NSC", "EMR", "KLAC", "TGT", "SNPS", "CDNS", "WM", "FDX", "ADP", "MPC"
]

# ----- 거래소 매핑 (NYSE vs NASDAQ) -----
NYSE_SYMBOLS = [
    "BRK-B", "JNJ", "JPM", "V", "PG", "XOM", "MA", "HD", "CVX", "MRK",
    "ABBV", "LLY", "PEP", "KO", "WMT", "MCD", "ACN", "TMO", "ABT",
    "DHR", "NEE", "VZ", "NKE", "PM", "CMCSA", "HON", "UPS", "LOW",
    "IBM", "CAT", "BA", "GE", "RTX", "SPGI", "MS", "GS", "BLK", "AXP"
]

def get_exchange(symbol):
    if symbol in NYSE_SYMBOLS:
        return "NYSE"
    return "NASD"


# ============================================
# 3. 기간 설정
# ============================================

# AI 학습 기간
AI_TRAIN_START = "2020-01-01"
AI_TRAIN_END = "2023-12-31"

# 백테스트 기간
BACKTEST_START = "2020-01-01"
BACKTEST_END = None  # None = 오늘까지

# 예측용
PREDICTION_DAYS = 60
LOOKBACK_DAYS = 200


# ============================================
# 4. 매매 전략 설정
# ============================================

# 전략 타입: "custom", "ai", "hybrid"
STRATEGY_TYPE = "custom"

# Hybrid 전략 가중치
CUSTOM_WEIGHT = 0.4
AI_WEIGHT = 0.6

# 모멘텀 가중치
WEIGHT_SHORT = 0.3
WEIGHT_MID = 0.3
WEIGHT_LONG = 0.4

# 상관관계 필터
MIN_CORRELATION = 0.5

# 종목 수
TOP_N = 3

# 손절
STOP_LOSS = -0.07


# ============================================
# 5. 페이퍼트레이딩 설정
# ============================================

# 초기 투자금 (USD)
INITIAL_CAPITAL = 2000

# 수수료율
BUY_COMMISSION = 0.0025
SELL_COMMISSION = 0.0025
SLIPPAGE = 0.001


# ============================================
# 6. 한국투자증권 API 설정
# ============================================

# 모의투자 (Paper Trading)
KIS_PAPER_APP_KEY = os.getenv("KIS_PAPER_APP_KEY", "")
KIS_PAPER_APP_SECRET = os.getenv("KIS_PAPER_APP_SECRET", "")
KIS_PAPER_ACCOUNT = os.getenv("KIS_PAPER_ACCOUNT", "")

# 실전투자 (Real Trading) - 나중에 사용
KIS_REAL_APP_KEY = os.getenv("KIS_REAL_APP_KEY", "")
KIS_REAL_APP_SECRET = os.getenv("KIS_REAL_APP_SECRET", "")
KIS_REAL_ACCOUNT = os.getenv("KIS_REAL_ACCOUNT", "")

# 공통
KIS_HTS_ID = os.getenv("KIS_HTS_ID", "")
KIS_ACCOUNT_PROD = os.getenv("KIS_ACCOUNT_PROD", "01")

# 모드: "paper" or "real"
KIS_MODE = os.getenv("KIS_MODE", "paper")

# API URL
KIS_URL_PAPER = "https://openapivts.koreainvestment.com:29443"
KIS_URL_REAL = "https://openapi.koreainvestment.com:9443"

def get_kis_url():
    if KIS_MODE == "real":
        return KIS_URL_REAL
    return KIS_URL_PAPER

def get_kis_credentials():
    if KIS_MODE == "real":
        return {
            "app_key": KIS_REAL_APP_KEY,
            "app_secret": KIS_REAL_APP_SECRET,
            "account": KIS_REAL_ACCOUNT
        }
    return {
        "app_key": KIS_PAPER_APP_KEY,
        "app_secret": KIS_PAPER_APP_SECRET,
        "account": KIS_PAPER_ACCOUNT
    }


# ============================================
# 7. 텔레그램 설정
# ============================================

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")


# ============================================
# 8. 구글 시트 설정
# ============================================

SPREADSHEET_NAME = "Stock_Paper_Trading"
GOOGLE_SHEET_ID = os.getenv("GOOGLE_SHEET_ID", "")
GOOGLE_CREDENTIALS = os.getenv("GOOGLE_CREDENTIALS", "")
GOOGLE_CREDENTIALS_FILE = "service_account.json"


# ============================================
# 9. AI 모델 설정
# ============================================

MODEL_SAVE_PATH = "models/"
LSTM_UNITS = 50
EPOCHS = 100
BATCH_SIZE = 32


# ============================================
# 설정 확인 함수
# ============================================

def print_config():
    print("=" * 60)
    print("Configuration")
    print("=" * 60)
    print(f"Project: {PROJECT_NAME}")
    print(f"Strategy: {STRATEGY_TYPE}")
    print(f"Mode: {KIS_MODE}")
    print()
    print(f"Initial Capital: ${INITIAL_CAPITAL:,}")
    print(f"Stop Loss: {STOP_LOSS*100}%")
    print(f"Top N: {TOP_N}")
    print()
    creds = get_kis_credentials()
    print(f"KIS App Key: {'SET' if creds['app_key'] else 'NOT SET'}")
    print(f"KIS Account: {'SET' if creds['account'] else 'NOT SET'}")
    print(f"HTS ID: {'SET' if KIS_HTS_ID else 'NOT SET'}")
    print()
    print(f"Telegram: {'SET' if TELEGRAM_BOT_TOKEN else 'NOT SET'}")
    print(f"Google Sheets: {SPREADSHEET_NAME}")
    print("=" * 60)


if __name__ == "__main__":
    print_config()