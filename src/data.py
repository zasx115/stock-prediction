# ============================================
# 파일명: src/data.py
# 설명: 주식 데이터 수집 및 전처리 모듈
# 작성일: 2025-01-29
# ============================================
#
# 이 파일의 역할:
# 1. S&P 500 종목 리스트 가져오기
# 2. 주가 데이터 수집 (Yahoo Finance)
# 3. 시장 지표 수집 (SPY, VIX)
# 4. 섹터 ETF 데이터 수집
# 5. 기술적 지표 계산 (이동평균, RSI, MACD 등)
# 6. AI 학습용 / 백테스트용 / 예측용 데이터 제공
# ============================================


# ----- 필요한 라이브러리 불러오기 -----
import pandas as pd                    # 데이터 처리
import numpy as np                     # 수치 계산
import yfinance as yf                  # 야후 파이낸스 데이터
from datetime import datetime, timedelta  # 날짜 처리
import warnings                        # 경고 메시지 관리

# 경고 메시지 숨기기 (출력 깔끔하게)
warnings.filterwarnings('ignore')

# ----- 설정 파일 불러오기 -----
import sys
import os

# src 폴더에서 실행해도 config를 찾을 수 있도록 경로 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


# ============================================
# 1. S&P 500 종목 리스트 가져오기
# ============================================

def get_sp500_list():
    """
    S&P 500에 포함된 종목 리스트를 가져옵니다.
    
    Returns:
        DataFrame: 종목코드(Symbol), 회사명(Name), 섹터(Sector) 포함
    
    사용 예시:
        sp500 = get_sp500_list()
        print(sp500.head())
    """
    
    print("S&P 500 종목 리스트 가져오는 중...")
    
    try:
        # 위키피디아에서 S&P 500 리스트 가져오기
        # (가장 최신 정보를 유지함)
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        
        # HTML 테이블 읽기 (첫 번째 테이블이 S&P 500 리스트)
        tables = pd.read_html(url)
        sp500_table = tables[0]
        
        # 필요한 컬럼만 선택하고 이름 변경
        sp500 = sp500_table[['Symbol', 'Security', 'GICS Sector']].copy()
        sp500.columns = ['symbol', 'name', 'sector']
        
        # 종목코드 정리 (일부 종목은 '.'이 들어있어서 '-'로 변경)
        # 예: BRK.B → BRK-B (Yahoo Finance 형식)
        sp500['symbol'] = sp500['symbol'].str.replace('.', '-', regex=False)
        
        print(f"✅ S&P 500 종목 {len(sp500)}개 로드 완료!")
        
        return sp500
    
    except Exception as e:
        print(f"❌ S&P 500 리스트 가져오기 실패: {e}")
        return pd.DataFrame()


# ============================================
# 2. 개별 종목 주가 데이터 수집
# ============================================

def get_stock_data(symbol, start_date, end_date=None):
    """
    특정 종목의 주가 데이터를 가져옵니다.
    
    Args:
        symbol (str): 종목 코드 (예: "AAPL", "MSFT")
        start_date (str): 시작일 (예: "2020-01-01")
        end_date (str): 종료일 (예: "2023-12-31"), None이면 오늘까지
    
    Returns:
        DataFrame: 날짜, 시가, 고가, 저가, 종가, 거래량 포함
    
    사용 예시:
        data = get_stock_data("AAPL", "2020-01-01", "2023-12-31")
        print(data.head())
    """
    
    # 종료일이 없으면 오늘 날짜 사용
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')
    
    try:
        # Yahoo Finance에서 데이터 다운로드
        ticker = yf.Ticker(symbol)
        data = ticker.history(start=start_date, end=end_date)
        
        # 데이터가 비어있으면 None 반환
        if data.empty:
            print(f"⚠️ {symbol}: 데이터 없음")
            return None
        
        # 컬럼명을 소문자로 변경 (일관성)
        data.columns = data.columns.str.lower()
        
        # 필요한 컬럼만 선택
        columns_to_keep = ['open', 'high', 'low', 'close', 'volume']
        data = data[[col for col in columns_to_keep if col in data.columns]]
        
        # 인덱스(날짜)를 컬럼으로 변환
        data = data.reset_index()
        data.rename(columns={'Date': 'date', 'index': 'date'}, inplace=True)
        
        # 날짜를 datetime 형식으로 (시간대 정보 제거)
        data['date'] = pd.to_datetime(data['date']).dt.tz_localize(None)
        
        # 종목 코드 컬럼 추가
        data['symbol'] = symbol
        
        return data
    
    except Exception as e:
        print(f"❌ {symbol} 데이터 수집 실패: {e}")
        return None


# ============================================
# 3. 여러 종목 데이터 한번에 수집
# ============================================

def get_multiple_stocks_data(symbols, start_date, end_date=None, show_progress=True):
    """
    여러 종목의 주가 데이터를 한번에 가져옵니다.
    
    Args:
        symbols (list): 종목 코드 리스트 (예: ["AAPL", "MSFT", "GOOGL"])
        start_date (str): 시작일
        end_date (str): 종료일 (None이면 오늘까지)
        show_progress (bool): 진행 상황 출력 여부
    
    Returns:
        dict: {종목코드: DataFrame} 형태의 딕셔너리
    
    사용 예시:
        stocks = get_multiple_stocks_data(["AAPL", "MSFT"], "2020-01-01")
        print(stocks["AAPL"].head())
    """
    
    all_data = {}           # 결과 저장할 딕셔너리
    success_count = 0       # 성공 개수
    fail_count = 0          # 실패 개수
    
    total = len(symbols)
    
    for i, symbol in enumerate(symbols):
        # 진행 상황 출력
        if show_progress and (i + 1) % 50 == 0:
            print(f"  진행중... {i + 1}/{total} ({(i + 1) / total * 100:.1f}%)")
        
        # 데이터 수집
        data = get_stock_data(symbol, start_date, end_date)
        
        if data is not None and not data.empty:
            all_data[symbol] = data
            success_count += 1
        else:
            fail_count += 1
    
    print(f"✅ 데이터 수집 완료: 성공 {success_count}개, 실패 {fail_count}개")
    
    return all_data


# ============================================
# 4. 시장 지표 데이터 수집 (SPY, VIX)
# ============================================

def get_market_indicators(start_date, end_date=None):
    """
    시장 지표 데이터를 가져옵니다. (SPY, VIX)
    
    Args:
        start_date (str): 시작일
        end_date (str): 종료일 (None이면 오늘까지)
    
    Returns:
        dict: {지표명: DataFrame} 형태의 딕셔너리
    
    사용 예시:
        market = get_market_indicators("2020-01-01")
        print(market["SPY"].head())
    """
    
    print("시장 지표 데이터 수집 중...")
    
    market_data = {}
    
    # config에서 시장 지표 목록 가져오기
    for symbol, description in config.MARKET_SYMBOLS.items():
        print(f"  - {description} ({symbol}) 수집 중...")
        
        data = get_stock_data(symbol, start_date, end_date)
        
        if data is not None:
            market_data[symbol] = data
            print(f"    ✅ {len(data)}일 데이터 수집 완료")
        else:
            print(f"    ❌ 수집 실패")
    
    return market_data


# ============================================
# 5. 섹터 ETF 데이터 수집
# ============================================

def get_sector_data(start_date, end_date=None):
    """
    11개 섹터 ETF 데이터를 가져옵니다.
    
    Args:
        start_date (str): 시작일
        end_date (str): 종료일 (None이면 오늘까지)
    
    Returns:
        dict: {섹터ETF: DataFrame} 형태의 딕셔너리
    
    사용 예시:
        sectors = get_sector_data("2020-01-01")
        print(sectors["XLK"].head())  # 기술 섹터
    """
    
    print("섹터 ETF 데이터 수집 중...")
    
    sector_data = {}
    
    # config에서 섹터 ETF 목록 가져오기
    for symbol, sector_name in config.SECTOR_ETFS.items():
        print(f"  - {sector_name} ({symbol}) 수집 중...")
        
        data = get_stock_data(symbol, start_date, end_date)
        
        if data is not None:
            sector_data[symbol] = data
            print(f"    ✅ {len(data)}일 데이터 수집 완료")
        else:
            print(f"    ❌ 수집 실패")
    
    return sector_data


# ============================================
# 6. 기술적 지표 계산
# ============================================

def add_technical_indicators(df):
    """
    주가 데이터에 기술적 지표를 추가합니다.
    
    추가되는 지표:
    - 이동평균선 (MA): 5일, 10일, 20일, 60일, 120일
    - RSI: 상대강도지수 (14일)
    - MACD: 이동평균수렴확산
    - 볼린저 밴드: 상단, 중간, 하단
    - 거래량 이동평균
    
    Args:
        df (DataFrame): 주가 데이터 (close, volume 컬럼 필요)
    
    Returns:
        DataFrame: 기술적 지표가 추가된 데이터
    
    사용 예시:
        data = get_stock_data("AAPL", "2020-01-01")
        data_with_indicators = add_technical_indicators(data)
    """
    
    # 원본 데이터 복사 (원본 보존)
    df = df.copy()
    
    # close 컬럼이 없으면 에러
    if 'close' not in df.columns:
        print("❌ 'close' 컬럼이 필요합니다.")
        return df
    
    # ----- 이동평균선 (Moving Average) -----
    # 단기, 중기, 장기 추세를 파악하는 데 사용
    df['ma_5'] = df['close'].rolling(window=5).mean()      # 5일 (1주)
    df['ma_10'] = df['close'].rolling(window=10).mean()    # 10일 (2주)
    df['ma_20'] = df['close'].rolling(window=20).mean()    # 20일 (1개월)
    df['ma_60'] = df['close'].rolling(window=60).mean()    # 60일 (3개월)
    df['ma_120'] = df['close'].rolling(window=120).mean()  # 120일 (6개월)
    
    # ----- RSI (Relative Strength Index) -----
    # 과매수(70 이상) / 과매도(30 이하) 판단
    # 14일 기준이 일반적
    delta = df['close'].diff()                             # 전일 대비 변화량
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()   # 상승분 평균
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()  # 하락분 평균
    rs = gain / loss                                       # 상대강도
    df['rsi'] = 100 - (100 / (1 + rs))                    # RSI 계산
    
    # ----- MACD (Moving Average Convergence Divergence) -----
    # 추세의 방향과 강도를 파악
    # MACD = 12일 EMA - 26일 EMA
    # Signal = MACD의 9일 EMA
    ema_12 = df['close'].ewm(span=12, adjust=False).mean()  # 12일 지수이동평균
    ema_26 = df['close'].ewm(span=26, adjust=False).mean()  # 26일 지수이동평균
    df['macd'] = ema_12 - ema_26                            # MACD 라인
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()  # 시그널 라인
    df['macd_histogram'] = df['macd'] - df['macd_signal']   # MACD 히스토그램
    
    # ----- 볼린저 밴드 (Bollinger Bands) -----
    # 주가의 변동성 범위를 파악
    # 중심선 = 20일 이동평균
    # 상단 = 중심선 + (2 * 표준편차)
    # 하단 = 중심선 - (2 * 표준편차)
    df['bb_middle'] = df['close'].rolling(window=20).mean()           # 중심선
    bb_std = df['close'].rolling(window=20).std()                     # 표준편차
    df['bb_upper'] = df['bb_middle'] + (2 * bb_std)                   # 상단 밴드
    df['bb_lower'] = df['bb_middle'] - (2 * bb_std)                   # 하단 밴드
    df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']  # 밴드 폭
    
    # ----- 거래량 관련 지표 -----
    if 'volume' in df.columns:
        df['volume_ma_20'] = df['volume'].rolling(window=20).mean()   # 거래량 20일 평균
        df['volume_ratio'] = df['volume'] / df['volume_ma_20']        # 거래량 비율
    
    # ----- 가격 변화율 -----
    df['price_change'] = df['close'].pct_change()          # 일일 수익률
    df['price_change_5d'] = df['close'].pct_change(5)      # 5일 수익률
    df['price_change_20d'] = df['close'].pct_change(20)    # 20일 수익률
    
    # ----- 고가/저가 대비 현재가 위치 -----
    # 최근 20일 중 현재 가격의 위치 (0~1 사이)
    high_20 = df['high'].rolling(window=20).max() if 'high' in df.columns else df['close'].rolling(window=20).max()
    low_20 = df['low'].rolling(window=20).min() if 'low' in df.columns else df['close'].rolling(window=20).min()
    df['price_position'] = (df['close'] - low_20) / (high_20 - low_20 + 0.0001)  # 0.0001은 0으로 나누기 방지
    
    return df


# ============================================
# 7. AI 학습용 데이터 가져오기
# ============================================

def get_data_for_training(symbols=None, include_indicators=True):
    """
    AI 모델 학습용 데이터를 가져옵니다.
    기간: config.AI_TRAIN_START ~ config.AI_TRAIN_END
    
    Args:
        symbols (list): 종목 리스트 (None이면 S&P 500 전체)
        include_indicators (bool): 기술적 지표 포함 여부
    
    Returns:
        dict: {
            'stocks': {종목: DataFrame},
            'market': {SPY, VIX: DataFrame},
            'sectors': {섹터ETF: DataFrame}
        }
    
    사용 예시 (Colab에서):
        from src.data import get_data_for_training
        data = get_data_for_training()
        
        # 애플 데이터 확인
        print(data['stocks']['AAPL'].head())
    """
    
    print("=" * 50)
    print("AI 학습용 데이터 수집 시작")
    print(f"기간: {config.AI_TRAIN_START} ~ {config.AI_TRAIN_END}")
    print("=" * 50)
    
    # 종목 리스트가 없으면 S&P 500 전체
    if symbols is None:
        sp500 = get_sp500_list()
        symbols = sp500['symbol'].tolist()
    
    # 결과 저장할 딕셔너리
    result = {
        'stocks': {},
        'market': {},
        'sectors': {},
        'info': {
            'period': f"{config.AI_TRAIN_START} ~ {config.AI_TRAIN_END}",
            'purpose': 'training'
        }
    }
    
    # 1. 개별 종목 데이터
    print(f"\n[1/3] S&P 500 종목 데이터 수집 ({len(symbols)}개)...")
    result['stocks'] = get_multiple_stocks_data(
        symbols, 
        config.AI_TRAIN_START, 
        config.AI_TRAIN_END
    )
    
    # 2. 시장 지표 데이터
    print(f"\n[2/3] 시장 지표 데이터 수집...")
    result['market'] = get_market_indicators(
        config.AI_TRAIN_START, 
        config.AI_TRAIN_END
    )
    
    # 3. 섹터 데이터
    print(f"\n[3/3] 섹터 ETF 데이터 수집...")
    result['sectors'] = get_sector_data(
        config.AI_TRAIN_START, 
        config.AI_TRAIN_END
    )
    
    # 기술적 지표 추가
    if include_indicators:
        print(f"\n기술적 지표 계산 중...")
        
        # 개별 종목에 지표 추가
        for symbol in result['stocks']:
            result['stocks'][symbol] = add_technical_indicators(result['stocks'][symbol])
        
        # 시장 지표에도 추가
        for symbol in result['market']:
            result['market'][symbol] = add_technical_indicators(result['market'][symbol])
        
        # 섹터에도 추가
        for symbol in result['sectors']:
            result['sectors'][symbol] = add_technical_indicators(result['sectors'][symbol])
        
        print("✅ 기술적 지표 추가 완료")
    
    print("\n" + "=" * 50)
    print("AI 학습용 데이터 수집 완료!")
    print(f"- 종목: {len(result['stocks'])}개")
    print(f"- 시장지표: {len(result['market'])}개")
    print(f"- 섹터: {len(result['sectors'])}개")
    print("=" * 50)
    
    return result


# ============================================
# 8. 백테스트용 데이터 가져오기
# ============================================

def get_data_for_backtest(symbols=None, include_indicators=True):
    """
    백테스트용 데이터를 가져옵니다.
    기간: config.BACKTEST_START ~ config.BACKTEST_END (또는 오늘)
    
    Args:
        symbols (list): 종목 리스트 (None이면 S&P 500 전체)
        include_indicators (bool): 기술적 지표 포함 여부
    
    Returns:
        dict: get_data_for_training()과 동일한 구조
    
    사용 예시 (Colab에서):
        from src.data import get_data_for_backtest
        data = get_data_for_backtest()
    """
    
    # 종료일 설정 (None이면 오늘)
    end_date = config.BACKTEST_END
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')
    
    print("=" * 50)
    print("백테스트용 데이터 수집 시작")
    print(f"기간: {config.BACKTEST_START} ~ {end_date}")
    print("=" * 50)
    
    # 종목 리스트가 없으면 S&P 500 전체
    if symbols is None:
        sp500 = get_sp500_list()
        symbols = sp500['symbol'].tolist()
    
    # 결과 저장
    result = {
        'stocks': {},
        'market': {},
        'sectors': {},
        'info': {
            'period': f"{config.BACKTEST_START} ~ {end_date}",
            'purpose': 'backtest'
        }
    }
    
    # 1. 개별 종목 데이터
    print(f"\n[1/3] S&P 500 종목 데이터 수집 ({len(symbols)}개)...")
    result['stocks'] = get_multiple_stocks_data(
        symbols, 
        config.BACKTEST_START, 
        end_date
    )
    
    # 2. 시장 지표 데이터
    print(f"\n[2/3] 시장 지표 데이터 수집...")
    result['market'] = get_market_indicators(
        config.BACKTEST_START, 
        end_date
    )
    
    # 3. 섹터 데이터
    print(f"\n[3/3] 섹터 ETF 데이터 수집...")
    result['sectors'] = get_sector_data(
        config.BACKTEST_START, 
        end_date
    )
    
    # 기술적 지표 추가
    if include_indicators:
        print(f"\n기술적 지표 계산 중...")
        
        for symbol in result['stocks']:
            result['stocks'][symbol] = add_technical_indicators(result['stocks'][symbol])
        
        for symbol in result['market']:
            result['market'][symbol] = add_technical_indicators(result['market'][symbol])
        
        for symbol in result['sectors']:
            result['sectors'][symbol] = add_technical_indicators(result['sectors'][symbol])
        
        print("✅ 기술적 지표 추가 완료")
    
    print("\n" + "=" * 50)
    print("백테스트용 데이터 수집 완료!")
    print(f"- 종목: {len(result['stocks'])}개")
    print(f"- 시장지표: {len(result['market'])}개")
    print(f"- 섹터: {len(result['sectors'])}개")
    print("=" * 50)
    
    return result


# ============================================
# 9. 예측용 데이터 가져오기 (매일 실행용)
# ============================================

def get_data_for_prediction(symbols=None, days=None):
    """
    예측용 최근 데이터를 가져옵니다.
    GitHub Actions에서 매일 실행할 때 사용합니다.
    
    Args:
        symbols (list): 종목 리스트 (None이면 S&P 500 전체)
        days (int): 가져올 일수 (None이면 config.PREDICTION_DAYS)
    
    Returns:
        dict: get_data_for_training()과 동일한 구조
    
    사용 예시:
        from src.data import get_data_for_prediction
        data = get_data_for_prediction(symbols=["AAPL", "MSFT"])
    """
    
    # 일수 설정
    if days is None:
        days = config.PREDICTION_DAYS
    
    # 시작일 계산 (오늘 - days일)
    # 주말/휴일 고려해서 여유있게 계산
    start_date = (datetime.now() - timedelta(days=days * 2)).strftime('%Y-%m-%d')
    end_date = datetime.now().strftime('%Y-%m-%d')
    
    print("=" * 50)
    print("예측용 데이터 수집 시작")
    print(f"최근 {days}일 데이터")
    print("=" * 50)
    
    # 종목 리스트가 없으면 S&P 500 전체
    if symbols is None:
        sp500 = get_sp500_list()
        symbols = sp500['symbol'].tolist()
    
    # 결과 저장
    result = {
        'stocks': {},
        'market': {},
        'sectors': {},
        'info': {
            'period': f"최근 {days}일",
            'purpose': 'prediction'
        }
    }
    
    # 1. 개별 종목 데이터
    print(f"\n[1/3] 종목 데이터 수집 ({len(symbols)}개)...")
    result['stocks'] = get_multiple_stocks_data(symbols, start_date, end_date)
    
    # 최근 N일만 유지
    for symbol in result['stocks']:
        result['stocks'][symbol] = result['stocks'][symbol].tail(days)
    
    # 2. 시장 지표 데이터
    print(f"\n[2/3] 시장 지표 데이터 수집...")
    result['market'] = get_market_indicators(start_date, end_date)
    for symbol in result['market']:
        result['market'][symbol] = result['market'][symbol].tail(days)
    
    # 3. 섹터 데이터
    print(f"\n[3/3] 섹터 ETF 데이터 수집...")
    result['sectors'] = get_sector_data(start_date, end_date)
    for symbol in result['sectors']:
        result['sectors'][symbol] = result['sectors'][symbol].tail(days)
    
    # 기술적 지표 추가 (항상)
    print(f"\n기술적 지표 계산 중...")
    for symbol in result['stocks']:
        result['stocks'][symbol] = add_technical_indicators(result['stocks'][symbol])
    for symbol in result['market']:
        result['market'][symbol] = add_technical_indicators(result['market'][symbol])
    for symbol in result['sectors']:
        result['sectors'][symbol] = add_technical_indicators(result['sectors'][symbol])
    print("✅ 기술적 지표 추가 완료")
    
    print("\n" + "=" * 50)
    print("예측용 데이터 수집 완료!")
    print("=" * 50)
    
    return result


# ============================================
# 10. 특정 종목 섹터 찾기
# ============================================

def get_stock_sector(symbol):
    """
    특정 종목이 속한 섹터를 찾습니다.
    
    Args:
        symbol (str): 종목 코드
    
    Returns:
        str: 섹터명 (없으면 None)
    
    사용 예시:
        sector = get_stock_sector("AAPL")
        print(sector)  # "Information Technology"
    """
    
    sp500 = get_sp500_list()
    stock_info = sp500[sp500['symbol'] == symbol]
    
    if not stock_info.empty:
        return stock_info['sector'].values[0]
    
    return None


# ============================================
# 11. 섹터별 종목 필터링
# ============================================

def get_stocks_by_sector(sector_name):
    """
    특정 섹터에 속한 종목 리스트를 가져옵니다.
    
    Args:
        sector_name (str): 섹터명 (예: "Information Technology", "Health Care")
    
    Returns:
        list: 종목 코드 리스트
    
    사용 예시:
        tech_stocks = get_stocks_by_sector("Information Technology")
        print(tech_stocks)  # ["AAPL", "MSFT", "NVDA", ...]
    """
    
    sp500 = get_sp500_list()
    filtered = sp500[sp500['sector'] == sector_name]
    
    return filtered['symbol'].tolist()


# ============================================
# 테스트 코드 (이 파일을 직접 실행할 때)
# ============================================

if __name__ == "__main__":
    """
    이 파일을 직접 실행하면 간단한 테스트를 수행합니다.
    
    실행 방법:
        python src/data.py
    """
    
    print("\n" + "=" * 60)
    print("data.py 테스트 시작")
    print("=" * 60)
    
    # 테스트 1: S&P 500 리스트
    print("\n[테스트 1] S&P 500 리스트 가져오기")
    sp500 = get_sp500_list()
    print(sp500.head())
    
    # 테스트 2: 단일 종목 데이터
    print("\n[테스트 2] 애플(AAPL) 데이터 가져오기")
    aapl = get_stock_data("AAPL", "2024-01-01", "2024-01-31")
    if aapl is not None:
        print(aapl.head())
    
    # 테스트 3: 기술적 지표 추가
    print("\n[테스트 3] 기술적 지표 추가")
    if aapl is not None:
        aapl_with_indicators = add_technical_indicators(aapl)
        print(f"컬럼 목록: {aapl_with_indicators.columns.tolist()}")
    
    # 테스트 4: 종목 섹터 찾기
    print("\n[테스트 4] AAPL 섹터 찾기")
    sector = get_stock_sector("AAPL")
    print(f"AAPL 섹터: {sector}")
    
    print("\n" + "=" * 60)
    print("테스트 완료!")
    print("=" * 60)
