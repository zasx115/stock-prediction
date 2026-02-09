# ============================================
# 파일명: config.py
# 설명: 프로젝트 전체 설정 파일
# 작성일: 2025-01-29
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

# 프로젝트 이름
PROJECT_NAME = "stock-prediction"

# 디버그 모드 (True: 상세 로그 출력, False: 간단 로그)
DEBUG_MODE = True


# ============================================
# 2. 주식 관련 설정
# ============================================

# ----- S&P 500 전체를 대상으로 함 -----
# 개별 종목은 자동으로 가져오므로 여기서 설정 안 함

# ----- 시장 지표 (벤치마크 + AI 학습용) -----
MARKET_SYMBOLS = {
    "SPY": "S&P 500 지수",      # 벤치마크
    "^VIX": "변동성 지수",       # 공포 지수
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

# ----- 기간 설정 -----
# AI 학습 기간 (과거 데이터로 모델 훈련)
AI_TRAIN_START = "2020-01-01"    # 학습 시작일
AI_TRAIN_END = "2023-12-31"      # 학습 종료일

# 백테스트 기간 (학습된 모델 검증)
BACKTEST_START = "2020-01-01"    # 백테스트 시작일
BACKTEST_END = None              # 백테스트 종료일 (None = 오늘까지)

# 예측용 (매일 실행 시 최근 데이터)
PREDICTION_DAYS = 60             # 예측에 필요한 과거 일수


# ============================================
# 3. 매매 전략 설정
# ============================================

# 사용할 전략 선택
# - "custom": 내가 만든 규칙 기반 전략
# - "ai": AI 모델 기반 전략  
# - "hybrid": 커스텀 + AI 결합 전략
STRATEGY_TYPE = "hybrid"

# 전략별 가중치 (hybrid 전략에서 사용)
# 커스텀 전략과 AI 전략의 비율 (합이 1.0이 되어야 함)
CUSTOM_WEIGHT = 0.4   # 커스텀 전략 40%
AI_WEIGHT = 0.6       # AI 전략 60%


# ============================================
# 4. 페이퍼트레이딩 (가상매매) 설정
# ============================================

# 초기 투자금 (가상)
INITIAL_CAPITAL = 10000000  # 1,000만원

# 1회 매매 금액
TRADE_AMOUNT = 1000000  # 100만원

# 수수료율 (0.015% = 키움증권 기준)
COMMISSION_RATE = 0.00015


# ============================================
# 5. 텔레그램 설정
# ============================================

# 텔레그램 봇 토큰 (환경변수에서 가져옴)
# - 봇 생성: 텔레그램에서 @BotFather 검색 후 /newbot 명령
# - 토큰은 GitHub Secrets에 저장
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")

# 텔레그램 채팅 ID (결과를 받을 채팅방)
# - 확인 방법: 봇에게 메시지 보낸 후 
#   https://api.telegram.org/bot<토큰>/getUpdates 접속
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")


# ============================================
# 6. 구글 시트 설정
# ============================================

# 구글 시트 문서 ID
# - 구글 시트 URL에서 확인: 
#   https://docs.google.com/spreadsheets/d/여기가_ID/edit
GOOGLE_SHEET_ID = os.getenv("GOOGLE_SHEET_ID", "")

# 구글 서비스 계정 JSON 파일 경로
# - Google Cloud Console에서 서비스 계정 생성 후 JSON 키 다운로드
GOOGLE_CREDENTIALS_FILE = "credentials.json"


# ============================================
# 7. 한국투자증권(KIS) API 설정
# ============================================

# KIS Open API 키 (환경변수에서 가져옴)
# - 발급: https://apiportal.koreainvestment.com 에서 앱 등록
# - APP_KEY, APP_SECRET은 GitHub Secrets에 저장
KIS_APP_KEY = os.getenv("KIS_APP_KEY", "")
KIS_APP_SECRET = os.getenv("KIS_APP_SECRET", "")

# 계좌 정보
KIS_ACCOUNT_NO = os.getenv("KIS_ACCOUNT_NO", "")           # 계좌번호 (8자리)
KIS_ACCOUNT_PRODUCT = os.getenv("KIS_ACCOUNT_PRODUCT", "01")  # 계좌상품코드

# 모의투자 / 실전투자 전환
# True: 모의투자 (기본값, 안전), False: 실전투자
KIS_PAPER_TRADE = True


# ============================================
# 8. AI 모델 설정
# ============================================

# 모델 저장 경로
MODEL_SAVE_PATH = "models/"

# 학습 파라미터
LSTM_UNITS = 50           # LSTM 레이어 유닛 수
EPOCHS = 100              # 학습 반복 횟수
BATCH_SIZE = 32           # 배치 크기
LOOKBACK_DAYS = 60        # 과거 며칠 데이터를 볼지


# ============================================
# 9. 백테스트 설정
# ============================================

# 백테스트 기간
BACKTEST_START = "2020-01-01"    # 백테스트 시작일
BACKTEST_END = None              # 백테스트 종료일 (None = 오늘까지)

# 수익률 계산 기준
BENCHMARK = "KOSPI"  # 비교 지수


# ============================================
# 설정 확인 함수 (디버그용)
# ============================================

def print_config():
    """
    현재 설정을 출력하는 함수
    디버깅할 때 사용하세요
    """
    print("=" * 50)
    print("현재 설정")
    print("=" * 50)
    print(f"프로젝트: {PROJECT_NAME}")
    print(f"종목 수: {len(STOCK_LIST)}개")
    print(f"전략: {STRATEGY_TYPE}")
    print(f"초기자금: {INITIAL_CAPITAL:,}원")
    print(f"텔레그램 설정: {'완료' if TELEGRAM_BOT_TOKEN else '미설정'}")
    print(f"구글시트 설정: {'완료' if GOOGLE_SHEET_ID else '미설정'}")
    print("=" * 50)


# 이 파일을 직접 실행하면 설정 확인
if __name__ == "__main__":
    print_config()
