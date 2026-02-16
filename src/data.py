# ============================================
# 파일명: src/data.py
# 설명: 데이터 다운로드 및 전처리
# 
# 기능:
# - S&P 500 종목 리스트 가져오기 (위키피디아)
# - 주가 데이터 다운로드 (yfinance)
# - 백테스트용 데이터 준비
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
    
    Returns:
        DataFrame: symbol(티커), company(회사명), sector(섹터) 컬럼 포함
    
    사용 예시:
        sp500 = get_sp500_list()
        symbols = sp500['symbol'].tolist()  # ['AAPL', 'MSFT', ...]
    """
    # 위키피디아 S&P 500 페이지에서 테이블 읽기
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    tables = pd.read_html(url)
    
    # 첫 번째 테이블이 종목 리스트
    df = tables[0]
    
    # 필요한 컬럼만 선택 및 이름 변경
    df = df[['Symbol', 'Security', 'GICS Sector']].copy()
    df.columns = ['symbol', 'company', 'sector']
    
    # 티커 정리 (BRK.B → BRK-B 형식으로 변환, yfinance 호환)
    df['symbol'] = df['symbol'].str.replace('.', '-', regex=False)
    
    return df


# ============================================
# [2] 주가 데이터 다운로드
# ============================================

def download_stock_data(symbols, start_date=None, end_date=None):
    """
    yfinance를 사용하여 주가 데이터를 다운로드합니다.
    
    Args:
        symbols: 티커 리스트 ['AAPL', 'MSFT', ...]
        start_date: 시작일 (문자열 'YYYY-MM-DD' 또는 None)
        end_date: 종료일 (문자열 'YYYY-MM-DD' 또는 None)
    
    Returns:
        DataFrame: date, symbol, open, high, low, close, volume 컬럼
    
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
        auto_adjust=True,  # 수정 주가 사용
        threads=True       # 병렬 다운로드
    )
    
    # 데이터가 없으면 빈 DataFrame 반환
    if data.empty:
        print("❌ 데이터 다운로드 실패")
        return pd.DataFrame()
    
    # 멀티인덱스 → 일반 DataFrame으로 변환
    result = []
    
    # 단일 종목인 경우
    if len(symbols) == 1:
        df = data.copy()
        df['symbol'] = symbols[0]
        df = df.reset_index()
        df.columns = ['date', 'close', 'high', 'low', 'open', 'volume', 'symbol']
        result.append(df)
    
    # 다중 종목인 경우
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
                
                # NaN 제거
                df = df.dropna(subset=['close'])
                
                if not df.empty:
                    result.append(df)
            
            except Exception as e:
                print(f"  ⚠️ {symbol} 데이터 처리 실패: {e}")
                continue
    
    # 모든 데이터 합치기
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
    
    # SPY 추가 (벤치마크)
    if 'SPY' not in symbols:
        symbols.append('SPY')
    
    # 데이터 다운로드
    df = download_stock_data(symbols, start_date, end_date)
    
    if df.empty:
        return df
    
    # 섹터 정보 추가
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
    
    Returns:
        dict: {
            'is_open': bool,
            'current_time': datetime,
            'market_time': str
        }
    """
    from datetime import datetime
    import pytz
    
    # 미국 동부 시간
    et = pytz.timezone('US/Eastern')
    now = datetime.now(et)
    
    # 주말 체크
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