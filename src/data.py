# ============================================
# 파일명: src/data.py
# 설명: 데이터 다운로드 및 전처리
#
# 역할 요약:
#   모든 전략(backtest, ai_backtest, hybrid)의 데이터 소스 계층.
#   - Wikipedia에서 S&P 500 종목 목록 파싱 (실패 시 config.SP500_BACKUP 폴백)
#   - yfinance를 사용해 수정 주가(auto_adjust=True) 멀티 종목 병렬 다운로드
#   - 다중 종목 멀티인덱스 → (date, symbol, OHLCV) 롱포맷으로 변환
#   - 백테스트용 데이터에 SPY(벤치마크) 및 섹터 정보 자동 추가
#   - 실시간 최신가 조회 및 미국 장 시간 확인 유틸리티 제공
#
# 주요 함수:
#   get_sp500_list()        → Wikipedia 파싱 → symbol/company/sector DataFrame
#   download_stock_data()   → yfinance 다운로드 → 롱포맷 DataFrame
#   get_backtest_data()     → 위 두 함수 조합 + 섹터 합치기
#   get_latest_prices()     → 실시간 단가 dict 반환 (paper trading용)
#   get_market_status()     → 미국 동부시간 기준 장 개장 여부 판단
#
# 의존 관계:
#   ← config.py (BACKTEST_START, BACKTEST_END, SP500_BACKUP)
#   → backtest.py, ai_data.py, hybrid_trading.py, paper_trading.py 에서 호출
# ============================================

import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import sys
import os

# 상위 폴더의 config.py 임포트
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import BACKTEST_START, BACKTEST_END


# ============================================
# [1] S&P 500 종목 리스트 가져오기
# ============================================

def get_sp500_list():
    """
    위키피디아에서 S&P 500 종목 리스트를 가져옵니다.

    알고리즘:
      1. Wikipedia S&P 500 페이지 HTTP 요청 (User-Agent 헤더 포함 → 403 방지)
      2. pd.read_html()로 HTML 테이블 파싱 → 첫 번째 테이블이 종목 목록
      3. Symbol / Security / GICS Sector 컬럼 추출 및 이름 정규화
      4. BRK.B → BRK-B 형식으로 변환 (yfinance는 '.' 대신 '-' 사용)
      5. 실패 시 config.SP500_BACKUP 하드코딩 리스트로 폴백

    Returns:
        DataFrame: symbol(티커), company(회사명), sector(섹터) 컬럼 포함

    사용 예시:
        sp500 = get_sp500_list()
        symbols = sp500['symbol'].tolist()  # ['AAPL', 'MSFT', ...]
    """
    import requests
    from io import StringIO

    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'

    # User-Agent 추가 (403 에러 방지)
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }

    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        tables = pd.read_html(StringIO(response.text))

        # 첫 번째 테이블이 종목 리스트
        df = tables[0]

        # 필요한 컬럼만 선택 및 이름 변경
        df = df[['Symbol', 'Security', 'GICS Sector']].copy()
        df.columns = ['symbol', 'company', 'sector']

        # 티커 정리 (BRK.B → BRK-B 형식으로 변환, yfinance 호환)
        df['symbol'] = df['symbol'].str.replace('.', '-', regex=False)

        print(f"✅ S&P 500 종목 로드: {len(df)}개")
        return df

    except Exception as e:
        print(f"⚠️ 위키피디아 실패: {e}")
        print("→ 백업 리스트 사용")

        # 백업 리스트 사용
        from config import SP500_BACKUP
        return pd.DataFrame({
            'symbol': SP500_BACKUP,
            'company': '',
            'sector': ''
        })


# ============================================
# [2] 주가 데이터 다운로드
# ============================================

def download_stock_data(symbols, start_date=None, end_date=None):
    """
    yfinance를 사용하여 주가 데이터를 다운로드합니다.

    알고리즘:
      1. start_date/end_date가 None이면 config 기본값 사용
      2. yf.download()로 전체 종목 일괄 요청 (threads=True → 병렬, auto_adjust=True → 수정 주가)
      3. 반환 데이터 구조 분기:
         - 단일 종목: 컬럼이 1차원 → 직접 rename
         - 다중 종목: (Price, Symbol) 멀티인덱스 → 종목별로 분리하여 롱포맷으로 변환
      4. close NaN 행 제거 후 pd.concat()으로 합치기

    Args:
        symbols: 티커 리스트 ['AAPL', 'MSFT', ...]
        start_date: 시작일 (문자열 'YYYY-MM-DD' 또는 None)
        end_date: 종료일 (문자열 'YYYY-MM-DD' 또는 None)

    Returns:
        DataFrame: date, symbol, open, high, low, close, volume 컬럼 (롱포맷)

    사용 예시:
        df = download_stock_data(['AAPL', 'MSFT'], '2020-01-01', '2024-12-31')
    """
    # 기본값 설정
    if start_date is None:
        start_date = BACKTEST_START
    if end_date is None:
        end_date = BACKTEST_END

    print(f"데이터 다운로드 시작...")
    print(f"  종목 수: {len(symbols)}개")
    print(f"  기간: {start_date} ~ {end_date}")

    # yfinance로 데이터 다운로드 (병렬 처리)
    data = yf.download(
        symbols,
        start=start_date,
        end=end_date,
        auto_adjust=True,  # 수정 주가 사용 (배당/분할 반영)
        threads=True       # 병렬 다운로드로 속도 향상
    )

    # 데이터가 없으면 빈 DataFrame 반환
    if data.empty:
        print("❌ 데이터 다운로드 실패")
        return pd.DataFrame()

    # 멀티인덱스 → 일반 DataFrame으로 변환
    result = []

    # 단일 종목인 경우: yfinance가 1차원 컬럼 반환
    if len(symbols) == 1:
        df = data.copy()
        df['symbol'] = symbols[0]
        df = df.reset_index()
        df.columns = ['date', 'close', 'high', 'low', 'open', 'volume', 'symbol']
        result.append(df)

    # 다중 종목인 경우: (가격종류, 티커) 2단계 멀티인덱스 반환
    else:
        for symbol in symbols:
            try:
                # 해당 종목 데이터 추출
                if symbol not in data['Close'].columns:
                    continue

                df = pd.DataFrame({
                    'date': data.index,
                    'open': data['Open'][symbol].values,
                    'high': data['High'][symbol].values,
                    'low': data['Low'][symbol].values,
                    'close': data['Close'][symbol].values,
                    'volume': data['Volume'][symbol].values,
                    'symbol': symbol
                })

                # 상장폐지/데이터 없는 날짜 제거
                df = df.dropna(subset=['close'])

                if not df.empty:
                    result.append(df)

            except Exception as e:
                print(f"  ⚠️ {symbol} 데이터 처리 실패: {e}")
                continue

    # 모든 종목 데이터 합치기
    if result:
        final_df = pd.concat(result, ignore_index=True)
        final_df['date'] = pd.to_datetime(final_df['date'])

        print(f"✅ 다운로드 완료!")
        print(f"  유효 종목: {final_df['symbol'].nunique()}개")
        print(f"  총 행 수: {len(final_df):,}개")

        return final_df
    else:
        print("❌ 유효한 데이터 없음")
        return pd.DataFrame()


# ============================================
# [3] 백테스트용 데이터 준비
# ============================================

def get_backtest_data(symbols=None, start_date=None, end_date=None):
    """
    백테스트에 필요한 데이터를 준비합니다.

    알고리즘:
      1. symbols가 None이면 get_sp500_list()로 전체 S&P 500 목록 가져오기
      2. SPY가 없으면 자동 추가 (모든 전략에서 벤치마크/상관관계 계산에 사용)
      3. download_stock_data()로 가격 데이터 다운로드
      4. get_sp500_list()로 섹터 정보 조회 후 symbol 기준으로 매핑
      5. (date, symbol) 기준 정렬 후 반환

    Args:
        symbols: 티커 리스트 (None이면 S&P 500 전체)
        start_date: 시작일 (None이면 config 사용)
        end_date: 종료일 (None이면 config 사용)

    Returns:
        DataFrame: 백테스트용 데이터 (date, symbol, close 등)

    사용 예시:
        # S&P 500 전체 데이터
        df = get_backtest_data()

        # 특정 종목만
        df = get_backtest_data(symbols=['AAPL', 'MSFT', 'SPY'])
    """
    # 종목 리스트 준비
    if symbols is None:
        sp500 = get_sp500_list()
        symbols = sp500['symbol'].tolist()

    # SPY 추가 (벤치마크 + 전략의 상관관계 계산에 필수)
    if 'SPY' not in symbols:
        symbols.append('SPY')

    # 데이터 다운로드
    df = download_stock_data(symbols, start_date, end_date)

    if df.empty:
        return df

    # 섹터 정보 추가 (symbol → sector 매핑)
    sp500 = get_sp500_list()
    sector_map = dict(zip(sp500['symbol'], sp500['sector']))
    df['sector'] = df['symbol'].map(sector_map)

    # 날짜순 정렬
    df = df.sort_values(['date', 'symbol']).reset_index(drop=True)

    return df


# ============================================
# [4] 유틸리티 함수
# ============================================

def get_latest_prices(symbols):
    """
    최신 주가를 가져옵니다 (실시간 매매용).
    
    Args:
        symbols: 티커 리스트
    
    Returns:
        dict: {symbol: price}
    
    사용 예시:
        prices = get_latest_prices(['AAPL', 'MSFT'])
        # {'AAPL': 185.50, 'MSFT': 375.20}
    """
    prices = {}
    
    for symbol in symbols:
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period='1d')
            
            if not hist.empty:
                prices[symbol] = hist['Close'].iloc[-1]
        except:
            continue
    
    return prices


def get_market_status():
    """
    미국 시장 상태를 확인합니다.

    알고리즘:
      1. pytz로 미국 동부 시간(ET) 기준 현재 시각 계산
      2. 주말 여부 확인 (weekday >= 5이면 토/일)
      3. 장 시간 범위 비교: 9:30 AM ~ 4:00 PM ET
      4. 주말 아님 AND 장 시간 내 → is_open = True
      주의: 공휴일은 별도 체크하지 않음 (yfinance 데이터로 보완)

    Returns:
        dict: {
            'is_open': bool,           # True이면 현재 장 개장 중
            'current_time': datetime,  # ET 기준 현재 시각
            'market_time': str         # 포맷된 문자열
        }
    """
    from datetime import datetime
    import pytz

    # 미국 동부 시간 (ET)
    et = pytz.timezone('US/Eastern')
    now = datetime.now(et)

    # 주말 체크 (weekday: 0=월 ~ 6=일, 5 이상이면 주말)
    is_weekend = now.weekday() >= 5

    # 장 시간 체크 (9:30 AM ~ 4:00 PM ET)
    market_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
    market_close = now.replace(hour=16, minute=0, second=0, microsecond=0)
    is_market_hours = market_open <= now <= market_close

    is_open = not is_weekend and is_market_hours

    return {
        'is_open': is_open,
        'current_time': now,
        'market_time': now.strftime('%Y-%m-%d %H:%M:%S ET')
    }


# ============================================
# [5] 테스트 실행
# ============================================

if __name__ == "__main__":
    # S&P 500 리스트 테스트
    print("=" * 50)
    print("S&P 500 종목 리스트 테스트")
    print("=" * 50)
    
    sp500 = get_sp500_list()
    print(f"총 종목 수: {len(sp500)}개")
    print(f"\n상위 5개 종목:")
    print(sp500.head())
    
    print(f"\n섹터별 종목 수:")
    print(sp500['sector'].value_counts())
    
    # 백테스트 데이터 테스트 (소규모)
    print("\n" + "=" * 50)
    print("데이터 다운로드 테스트 (5개 종목)")
    print("=" * 50)
    
    test_symbols = ['AAPL', 'MSFT', 'GOOGL', 'NVDA', 'SPY']
    df = get_backtest_data(symbols=test_symbols)
    
    print(f"\n데이터 샘플:")
    print(df.head(10))