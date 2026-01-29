# ============================================
# 파일명: src/data.py
# 설명: 주식 데이터 수집 및 전처리 모듈
# 작성일: 2025-01-29
# 수정일: 2025-01-29 (S&P 500 리스트 버그 수정)
# ============================================
#
# 이 파일의 역할:
# 1. S&P 500 종목 리스트 가져오기
# 2. 주가 데이터 수집 (Yahoo Finance)
# 3. 시장 지표 수집 (SPY, VIX)
# 4. 섹터 ETF 데이터 수집
# 5. 기술적 지표 계산 (이동평균, RSI, MACD 등)
# 6. AI 학습용 / 백테스트용 / 예측용 데이터 제공
# 7. 개별 데이터를 하나의 테이블로 병합
# ============================================


# ----- 필요한 라이브러리 불러오기 -----
import pandas as pd                    # 데이터 처리
import numpy as np                     # 수치 계산
import yfinance as yf                  # 야후 파이낸스 데이터
from datetime import datetime, timedelta  # 날짜 처리
import warnings                        # 경고 메시지 관리
import requests                        # HTTP 요청

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
    여러 방법을 시도해서 하나라도 성공하면 반환합니다.
    
    Returns:
        DataFrame: 종목코드(Symbol), 회사명(Name), 섹터(Sector) 포함
    
    사용 예시:
        sp500 = get_sp500_list()
        print(sp500.head())
    """
    
    print("S&P 500 종목 리스트 가져오는 중...")
    
    # ----- 방법 1: GitHub 저장된 리스트 사용 (가장 안정적) -----
    try:
        print("  → 방법 1: GitHub 데이터 시도...")
        url = "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/main/data/constituents.csv"
        
        sp500 = pd.read_csv(url)
        
        # 컬럼명 통일
        sp500 = sp500[['Symbol', 'Security', 'GICS Sector']].copy()
        sp500.columns = ['symbol', 'name', 'sector']
        sp500['symbol'] = sp500['symbol'].str.replace('.', '-', regex=False)
        
        print(f"✅ S&P 500 종목 {len(sp500)}개 로드 완료! (GitHub)")
        return sp500
        
    except Exception as e:
        print(f"  ⚠️ GitHub 리스트 실패: {e}")
    
    # ----- 방법 2: 위키피디아 (requests 사용) -----
    try:
        print("  → 방법 2: 위키피디아 시도...")
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Connection': 'keep-alive',
        }
        
        response = redef get_sp500_list():
    """
    S&P 500에 포함된 종목 리스트를 가져옵니다.
    
    Returns:
        DataFrame: 종목코드(Symbol), 회사명(Name), 섹터(Sector) 포함
    """
    
    print("S&P 500 종목 리스트 가져오는 중...")
    
    try:
        url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers)
        table = pd.read_html(response.text)
        
        # 첫 번째 테이블에서 필요한 컬럼 추출
        sp500_table = table[0]
        sp500 = sp500_table[['Symbol', 'Security', 'GICS Sector']].copy()
        sp500.columns = ['symbol', 'name', 'sector']
        
        # 종목코드 정리 (BRK.B → BRK-B)
        sp500['symbol'] = sp500['symbol'].replace('\.', '-', regex=True)
        
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
    df['ma_5'] = df['close'].rolling(window=5).mean()
    df['ma_10'] = df['close'].rolling(window=10).mean()
    df['ma_20'] = df['close'].rolling(window=20).mean()
    df['ma_60'] = df['close'].rolling(window=60).mean()
    df['ma_120'] = df['close'].rolling(window=120).mean()
    
    # ----- RSI (Relative Strength Index) -----
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # ----- MACD -----
    ema_12 = df['close'].ewm(span=12, adjust=False).mean()
    ema_26 = df['close'].ewm(span=26, adjust=False).mean()
    df['macd'] = ema_12 - ema_26
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_histogram'] = df['macd'] - df['macd_signal']
    
    # ----- 볼린저 밴드 -----
    df['bb_middle'] = df['close'].rolling(window=20).mean()
    bb_std = df['close'].rolling(window=20).std()
    df['bb_upper'] = df['bb_middle'] + (2 * bb_std)
    df['bb_lower'] = df['bb_middle'] - (2 * bb_std)
    df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
    
    # ----- 거래량 관련 지표 -----
    if 'volume' in df.columns:
        df['volume_ma_20'] = df['volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_ma_20']
    
    # ----- 가격 변화율 -----
    df['price_change'] = df['close'].pct_change()
    df['price_change_5d'] = df['close'].pct_change(5)
    df['price_change_20d'] = df['close'].pct_change(20)
    
    # ----- 고가/저가 대비 현재가 위치 -----
    high_20 = df['high'].rolling(window=20).max() if 'high' in df.columns else df['close'].rolling(window=20).max()
    low_20 = df['low'].rolling(window=20).min() if 'low' in df.columns else df['close'].rolling(window=20).min()
    df['price_position'] = (df['close'] - low_20) / (high_20 - low_20 + 0.0001)
    
    return df


# ============================================
# 7. 개별 종목 데이터를 하나의 테이블로 병합
# ============================================

def merge_all_stocks(stocks_dict, sp500_info=None):
    """
    개별 종목 데이터(딕셔너리)를 하나의 큰 DataFrame으로 병합합니다.
    
    Args:
        stocks_dict (dict): {종목코드: DataFrame} 형태의 딕셔너리
        sp500_info (DataFrame): S&P 500 정보 (섹터 정보 추가용)
    
    Returns:
        DataFrame: 모든 종목이 합쳐진 하나의 테이블
    """
    
    print("개별 종목 데이터 병합 중...")
    
    if not stocks_dict:
        print("❌ 병합할 데이터가 없습니다.")
        return pd.DataFrame()
    
    if sp500_info is None:
        sp500_info = get_sp500_list()
    
    sector_dict = dict(zip(sp500_info['symbol'], sp500_info['sector']))
    
    all_dataframes = []
    
    for symbol, df in stocks_dict.items():
        df_copy = df.copy()
        df_copy['sector'] = sector_dict.get(symbol, 'Unknown')
        all_dataframes.append(df_copy)
    
    merged_df = pd.concat(all_dataframes, ignore_index=True)
    merged_df = merged_df.sort_values(['date', 'symbol']).reset_index(drop=True)
    
    print(f"✅ 병합 완료: {len(stocks_dict)}개 종목, {len(merged_df):,}행")
    
    return merged_df


# ============================================
# 8. 시장 데이터를 종목 데이터에 결합
# ============================================

def merge_with_market_data(stocks_df, market_dict):
    """
    종목 데이터에 시장 지표(SPY, VIX)를 결합합니다.
    """
    
    print("시장 지표 데이터 결합 중...")
    
    result_df = stocks_df.copy()
    
    if 'SPY' in market_dict:
        spy_df = market_dict['SPY'][['date', 'close']].copy()
        spy_df.columns = ['date', 'spy_close']
        spy_df['spy_change'] = spy_df['spy_close'].pct_change()
        result_df = result_df.merge(spy_df, on='date', how='left')
        print("  ✅ SPY 데이터 결합 완료")
    
    if '^VIX' in market_dict:
        vix_df = market_dict['^VIX'][['date', 'close']].copy()
        vix_df.columns = ['date', 'vix_close']
        vix_df['vix_change'] = vix_df['vix_close'].pct_change()
        result_df = result_df.merge(vix_df, on='date', how='left')
        print("  ✅ VIX 데이터 결합 완료")
    
    print(f"✅ 시장 지표 결합 완료: 총 {len(result_df.columns)}개 컬럼")
    
    return result_df


# ============================================
# 9. 섹터 ETF 데이터를 종목 데이터에 결합
# ============================================

def merge_with_sector_data(stocks_df, sector_dict):
    """
    종목 데이터에 해당 종목의 섹터 ETF 데이터를 결합합니다.
    """
    
    print("섹터 ETF 데이터 결합 중...")
    
    sector_to_etf = {
        'Information Technology': 'XLK',
        'Financials': 'XLF',
        'Energy': 'XLE',
        'Health Care': 'XLV',
        'Consumer Discretionary': 'XLY',
        'Consumer Staples': 'XLP',
        'Industrials': 'XLI',
        'Materials': 'XLB',
        'Real Estate': 'XLRE',
        'Communication Services': 'XLC',
        'Utilities': 'XLU',
    }
    
    sector_etf_data = {}
    
    for etf_symbol, etf_df in sector_dict.items():
        temp_df = etf_df[['date', 'close']].copy()
        temp_df.columns = ['date', f'{etf_symbol}_close']
        temp_df[f'{etf_symbol}_change'] = temp_df[f'{etf_symbol}_close'].pct_change()
        sector_etf_data[etf_symbol] = temp_df
    
    result_df = stocks_df.copy()
    result_df['sector_etf_close'] = np.nan
    result_df['sector_etf_change'] = np.nan
    
    for sector_name, etf_symbol in sector_to_etf.items():
        if etf_symbol in sector_etf_data:
            etf_df = sector_etf_data[etf_symbol]
            sector_mask = result_df['sector'] == sector_name
            
            for idx in result_df[sector_mask].index:
                date = result_df.loc[idx, 'date']
                etf_row = etf_df[etf_df['date'] == date]
                
                if not etf_row.empty:
                    result_df.loc[idx, 'sector_etf_close'] = etf_row[f'{etf_symbol}_close'].values[0]
                    result_df.loc[idx, 'sector_etf_change'] = etf_row[f'{etf_symbol}_change'].values[0]
    
    print(f"✅ 섹터 ETF 결합 완료")
    
    return result_df


# ============================================
# 10. AI 학습용 완성 데이터셋
# ============================================

def get_full_dataset_for_training(symbols=None):
    """
    AI 모델 학습에 바로 사용할 수 있는 완성된 데이터셋을 반환합니다.
    기간: config.AI_TRAIN_START ~ config.AI_TRAIN_END
    """
    
    print("=" * 60)
    print("AI 학습용 완성 데이터셋 생성 시작")
    print(f"기간: {config.AI_TRAIN_START} ~ {config.AI_TRAIN_END}")
    print("=" * 60)
    
    sp500_info = get_sp500_list()
    
    if symbols is None:
        symbols = sp500_info['symbol'].tolist()
    
    result = {
        'stocks': {},
        'market': {},
        'sectors': {},
    }
    
    print(f"\n[1/6] 종목 데이터 수집 ({len(symbols)}개)...")
    result['stocks'] = get_multiple_stocks_data(
        symbols,
        config.AI_TRAIN_START,
        config.AI_TRAIN_END
    )
    
    print(f"\n[2/6] 기술적 지표 계산 중...")
    for symbol in result['stocks']:
        result['stocks'][symbol] = add_technical_indicators(result['stocks'][symbol])
    print("  ✅ 완료")
    
    print(f"\n[3/6] 종목 데이터 병합 중...")
    merged_df = merge_all_stocks(result['stocks'], sp500_info)
    
    print(f"\n[4/6] 시장 지표 수집 및 결합...")
    result['market'] = get_market_indicators(
        config.AI_TRAIN_START,
        config.AI_TRAIN_END
    )
    merged_df = merge_with_market_data(merged_df, result['market'])
    
    print(f"\n[5/6] 섹터 ETF 수집 및 결합...")
    result['sectors'] = get_sector_data(
        config.AI_TRAIN_START,
        config.AI_TRAIN_END
    )
    merged_df = merge_with_sector_data(merged_df, result['sectors'])
    
    print(f"\n[6/6] 데이터 정리 중...")
    min_date = merged_df['date'].min() + pd.Timedelta(days=150)
    merged_df = merged_df[merged_df['date'] >= min_date].reset_index(drop=True)
    merged_df = merged_df.fillna(method='ffill')
    merged_df = merged_df.dropna().reset_index(drop=True)
    print("  ✅ 완료")
    
    print("\n" + "=" * 60)
    print("AI 학습용 데이터셋 생성 완료!")
    print("=" * 60)
    print(f"  - 총 행 수: {len(merged_df):,}행")
    print(f"  - 총 컬럼 수: {len(merged_df.columns)}개")
    print(f"  - 종목 수: {merged_df['symbol'].nunique()}개")
    print(f"  - 기간: {merged_df['date'].min().strftime('%Y-%m-%d')} ~ {merged_df['date'].max().strftime('%Y-%m-%d')}")
    print("=" * 60)
    
    return merged_df


# ============================================
# 11. 백테스트용 완성 데이터셋
# ============================================

def get_full_dataset_for_backtest(symbols=None):
    """
    백테스트에 사용할 완성된 데이터셋을 반환합니다.
    기간: config.BACKTEST_START ~ config.BACKTEST_END (또는 오늘)
    """
    
    end_date = config.BACKTEST_END
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')
    
    print("=" * 60)
    print("백테스트용 완성 데이터셋 생성 시작")
    print(f"기간: {config.BACKTEST_START} ~ {end_date}")
    print("=" * 60)
    
    sp500_info = get_sp500_list()
    
    if symbols is None:
        symbols = sp500_info['symbol'].tolist()
    
    print(f"\n[1/6] 종목 데이터 수집 ({len(symbols)}개)...")
    stocks_data = get_multiple_stocks_data(
        symbols,
        config.BACKTEST_START,
        end_date
    )
    
    print(f"\n[2/6] 기술적 지표 계산 중...")
    for symbol in stocks_data:
        stocks_data[symbol] = add_technical_indicators(stocks_data[symbol])
    print("  ✅ 완료")
    
    print(f"\n[3/6] 종목 데이터 병합 중...")
    merged_df = merge_all_stocks(stocks_data, sp500_info)
    
    print(f"\n[4/6] 시장 지표 수집 및 결합...")
    market_data = get_market_indicators(config.BACKTEST_START, end_date)
    merged_df = merge_with_market_data(merged_df, market_data)
    
    print(f"\n[5/6] 섹터 ETF 수집 및 결합...")
    sector_data = get_sector_data(config.BACKTEST_START, end_date)
    merged_df = merge_with_sector_data(merged_df, sector_data)
    
    print(f"\n[6/6] 데이터 정리 중...")
    min_date = merged_df['date'].min() + pd.Timedelta(days=150)
    merged_df = merged_df[merged_df['date'] >= min_date].reset_index(drop=True)
    merged_df = merged_df.fillna(method='ffill')
    merged_df = merged_df.dropna().reset_index(drop=True)
    print("  ✅ 완료")
    
    print("\n" + "=" * 60)
    print("백테스트용 데이터셋 생성 완료!")
    print("=" * 60)
    print(f"  - 총 행 수: {len(merged_df):,}행")
    print(f"  - 종목 수: {merged_df['symbol'].nunique()}개")
    print("=" * 60)
    
    return merged_df


# ============================================
# 12. 예측용 완성 데이터셋 (매일 실행용)
# ============================================

def get_full_dataset_for_prediction(symbols=None, days=None):
    """
    예측에 사용할 최근 데이터셋을 반환합니다.
    GitHub Actions에서 매일 실행할 때 사용합니다.
    """
    
    if days is None:
        days = config.PREDICTION_DAYS
    
    start_date = (datetime.now() - timedelta(days=days + 200)).strftime('%Y-%m-%d')
    end_date = datetime.now().strftime('%Y-%m-%d')
    
    print("=" * 60)
    print("예측용 데이터셋 생성 시작")
    print(f"최근 {days}일 데이터")
    print("=" * 60)
    
    sp500_info = get_sp500_list()
    
    if symbols is None:
        symbols = sp500_info['symbol'].tolist()
    
    print(f"\n[1/6] 종목 데이터 수집 ({len(symbols)}개)...")
    stocks_data = get_multiple_stocks_data(symbols, start_date, end_date)
    
    print(f"\n[2/6] 기술적 지표 계산 중...")
    for symbol in stocks_data:
        stocks_data[symbol] = add_technical_indicators(stocks_data[symbol])
    print("  ✅ 완료")
    
    print(f"\n[3/6] 종목 데이터 병합 중...")
    merged_df = merge_all_stocks(stocks_data, sp500_info)
    
    print(f"\n[4/6] 시장 지표 수집 및 결합...")
    market_data = get_market_indicators(start_date, end_date)
    merged_df = merge_with_market_data(merged_df, market_data)
    
    print(f"\n[5/6] 섹터 ETF 수집 및 결합...")
    sector_data = get_sector_data(start_date, end_date)
    merged_df = merge_with_sector_data(merged_df, sector_data)
    
    print(f"\n[6/6] 최근 {days}일 데이터만 유지...")
    merged_df = merged_df.fillna(method='ffill')
    merged_df = merged_df.dropna().reset_index(drop=True)
    
    latest_date = merged_df['date'].max()
    cutoff_date = latest_date - pd.Timedelta(days=days)
    merged_df = merged_df[merged_df['date'] >= cutoff_date].reset_index(drop=True)
    print("  ✅ 완료")
    
    print("\n" + "=" * 60)
    print("예측용 데이터셋 생성 완료!")
    print("=" * 60)
    
    return merged_df


# ============================================
# 13. 유틸리티 함수들
# ============================================

def get_stock_sector(symbol):
    """특정 종목이 속한 섹터를 찾습니다."""
    sp500 = get_sp500_list()
    stock_info = sp500[sp500['symbol'] == symbol]
    
    if not stock_info.empty:
        return stock_info['sector'].values[0]
    return None


def get_stocks_by_sector(sector_name):
    """특정 섹터에 속한 종목 리스트를 가져옵니다."""
    sp500 = get_sp500_list()
    filtered = sp500[sp500['sector'] == sector_name]
    return filtered['symbol'].tolist()


# ============================================
# 테스트 코드
# ============================================

if __name__ == "__main__":
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
    
    print("\n" + "=" * 60)
    print("테스트 완료!")
    print("=" * 60)
