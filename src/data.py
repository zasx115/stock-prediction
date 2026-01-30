# ============================================
# 파일명: src/data.py
# 설명: 주식 데이터 수집 및 전처리
# ============================================

import pandas as pd
import numpy as np
import yfinance as yf
import requests
from io import StringIO
from datetime import datetime, timedelta

# 설정 불러오기
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


# ============================================
# 1. S&P 500 종목 리스트 가져오기
# ============================================

def get_sp500_list():
    """
    S&P 500 종목 리스트를 위키피디아에서 가져옵니다.
    
    Returns:
        DataFrame: symbol(종목코드), name(회사명), sector(섹터)
    """
    print("S&P 500 종목 리스트 가져오는 중...")
    
    try:
        url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers)
        table = pd.read_html(StringIO(response.text))
        
        sp500 = table[0][['Symbol', 'Security', 'GICS Sector']].copy()
        sp500.columns = ['symbol', 'name', 'sector']
        sp500['symbol'] = sp500['symbol'].replace('\.', '-', regex=True)
        
        print(f"✅ {len(sp500)}개 종목 로드 완료!")
        return sp500
        
    except Exception as e:
        print(f"❌ 실패: {e}")
        return pd.DataFrame()


# ============================================
# 2. 주가 데이터 다운로드
# ============================================

def get_stock_data(symbol, start_date, end_date=None):
    """
    개별 종목 1개의 주가 데이터를 가져옵니다.
    
    Args:
        symbol: 종목코드 (예: "AAPL")
        start_date: 시작일 (예: "2020-01-01")
        end_date: 종료일 (None이면 오늘)
    
    Returns:
        DataFrame: date, open, high, low, close, volume, symbol
    """
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')
    
    try:
        data = yf.download(symbol, start=start_date, end=end_date, progress=False)
        
        if data.empty:
            return None
        
        data = data.reset_index()
        data.columns = ['date', 'close', 'high', 'low', 'open', 'volume']
        data['symbol'] = symbol
        
        return data
        
    except:
        return None


def get_multiple_stocks(symbols, start_date, end_date=None):
    """
    여러 종목의 주가 데이터를 한번에 다운로드합니다.
    yfinance의 bulk download 기능 사용 (빠름)
    
    Args:
        symbols: 종목코드 리스트 (예: ["AAPL", "MSFT", "GOOGL"])
        start_date: 시작일
        end_date: 종료일 (None이면 오늘)
    
    Returns:
        DataFrame: 모든 종목이 하나의 테이블로 합쳐진 데이터
    """
    print(f"{len(symbols)}개 종목 데이터 다운로드 중...")
    
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')
    
    try:
        data = yf.download(symbols, start=start_date, end=end_date, progress=True)
        
        if data.empty:
            print("❌ 데이터 없음")
            return pd.DataFrame()
        
        # 종목별로 분리해서 다시 합치기 (테이블 구조 통일)
        all_data = []
        
        for symbol in symbols:
            try:
                if len(symbols) == 1:
                    stock_data = data.copy()
                else:
                    stock_data = data.xs(symbol, level=1, axis=1).copy()
                
                stock_data = stock_data.reset_index()
                stock_data.columns = ['date', 'close', 'high', 'low', 'open', 'volume']
                stock_data['symbol'] = symbol
                all_data.append(stock_data)
            except:
                continue
        
        result = pd.concat(all_data, ignore_index=True)
        print(f"✅ {result['symbol'].nunique()}개 종목, {len(result):,}행 완료!")
        
        return result
        
    except Exception as e:
        print(f"❌ 실패: {e}")
        return pd.DataFrame()


# ============================================
# 3. 기술적 지표 계산
# ============================================

def add_indicators(df):
    """
    주가 데이터에 기술적 지표를 추가합니다.
    종목별로 그룹화해서 계산합니다.
    
    추가되는 지표:
    - ma_5, ma_20, ma_60: 이동평균선
    - rsi: 상대강도지수 (14일)
    - macd, macd_signal: MACD
    - bb_upper, bb_middle, bb_lower: 볼린저밴드
    - daily_return: 일일 수익률
    
    Args:
        df: 주가 데이터
    
    Returns:
        DataFrame: 지표가 추가된 데이터
    """
    print("기술적 지표 계산 중...")
    
    df = df.copy()
    df = df.sort_values(['symbol', 'date']).reset_index(drop=True)
    
    result = []
    
    for symbol in df['symbol'].unique():
        stock = df[df['symbol'] == symbol].copy()
        
        # 이동평균선
        stock['ma_5'] = stock['close'].rolling(5).mean()
        stock['ma_20'] = stock['close'].rolling(20).mean()
        stock['ma_60'] = stock['close'].rolling(60).mean()
        
        # RSI (14일)
        delta = stock['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        stock['rsi'] = 100 - (100 / (1 + rs))
        
        # MACD
        ema_12 = stock['close'].ewm(span=12).mean()
        ema_26 = stock['close'].ewm(span=26).mean()
        stock['macd'] = ema_12 - ema_26
        stock['macd_signal'] = stock['macd'].ewm(span=9).mean()
        
        # 볼린저밴드
        stock['bb_middle'] = stock['close'].rolling(20).mean()
        bb_std = stock['close'].rolling(20).std()
        stock['bb_upper'] = stock['bb_middle'] + (2 * bb_std)
        stock['bb_lower'] = stock['bb_middle'] - (2 * bb_std)
        
        # 일일 수익률
        stock['daily_return'] = stock['close'].pct_change()
        
        result.append(stock)
    
    final = pd.concat(result, ignore_index=True)
    print(f"✅ 지표 추가 완료! ({len(final.columns)}개 컬럼)")
    
    return final


# ============================================
# 4. 용도별 데이터셋 생성
# ============================================

def get_training_data(symbols=None):
    """
    [Colab에서 사용]
    AI 모델 학습용 데이터셋을 생성합니다.
    
    기간: config.AI_TRAIN_START ~ config.AI_TRAIN_END (예: 2020-01 ~ 2023-12)
    용도: LSTM 등 AI 모델 훈련에 사용
    
    Args:
        symbols: 종목 리스트 (None이면 S&P 500 전체)
    
    Returns:
        DataFrame: AI 학습에 바로 사용 가능한 데이터
        
    사용 예시:
        df = get_training_data()  # S&P 500 전체
        df = get_training_data(["AAPL", "MSFT"])  # 특정 종목만
    """
    print("=" * 50)
    print("[AI 학습용 데이터 생성]")
    print(f"기간: {config.AI_TRAIN_START} ~ {config.AI_TRAIN_END}")
    print("=" * 50)
    
    # 종목 리스트 (없으면 S&P 500 전체)
    if symbols is None:
        sp500 = get_sp500_list()
        symbols = sp500['symbol'].tolist()
    
    # 주가 다운로드
    df = get_multiple_stocks(symbols, config.AI_TRAIN_START, config.AI_TRAIN_END)
    
    if df.empty:
        return pd.DataFrame()
    
    # 지표 추가
    df = add_indicators(df)
    
    # 섹터 정보 추가
    sp500 = get_sp500_list()
    sector_map = dict(zip(sp500['symbol'], sp500['sector']))
    df['sector'] = df['symbol'].map(sector_map).fillna('Unknown')
    
    # 결측치 제거
    df = df.dropna().reset_index(drop=True)
    
    print("=" * 50)
    print(f"✅ 완료: {len(df):,}행, {df['symbol'].nunique()}개 종목")
    print(f"컬럼: {df.columns.tolist()}")
    print("=" * 50)
    
    return df


def get_backtest_data(symbols=None):
    """
    [Colab에서 사용]
    백테스트용 데이터셋을 생성합니다.
    학습에 사용하지 않은 기간으로 전략 검증에 사용합니다.
    
    기간: config.BACKTEST_START ~ config.BACKTEST_END (예: 2024-01 ~ 현재)
    용도: 학습된 모델의 성능 검증, 전략 백테스트
    
    Args:
        symbols: 종목 리스트 (None이면 S&P 500 전체)
    
    Returns:
        DataFrame: 백테스트에 사용할 데이터
    """
    end_date = config.BACKTEST_END or datetime.now().strftime('%Y-%m-%d')
    
    print("=" * 50)
    print("[백테스트용 데이터 생성]")
    print(f"기간: {config.BACKTEST_START} ~ {end_date}")
    print("=" * 50)
    
    if symbols is None:
        sp500 = get_sp500_list()
        symbols = sp500['symbol'].tolist()
    
    df = get_multiple_stocks(symbols, config.BACKTEST_START, end_date)
    
    if df.empty:
        return pd.DataFrame()
    
    df = add_indicators(df)
    
    sp500 = get_sp500_list()
    sector_map = dict(zip(sp500['symbol'], sp500['sector']))
    df['sector'] = df['symbol'].map(sector_map).fillna('Unknown')
    
    df = df.dropna().reset_index(drop=True)
    
    print("=" * 50)
    print(f"✅ 완료: {len(df):,}행, {df['symbol'].nunique()}개 종목")
    print("=" * 50)
    
    return df


def get_prediction_data(symbols=None, days=60):
    """
    [GitHub Actions에서 매일 사용]
    페이퍼트레이딩(가상매매)용 최근 데이터를 생성합니다.
    매일 자동 실행되어 오늘의 매매 신호를 예측하는 데 사용합니다.
    
    기간: 최근 N일
    용도: 오늘 매수/매도 신호 예측, 페이퍼트레이딩
    
    Args:
        symbols: 종목 리스트 (None이면 S&P 500 전체)
        days: 가져올 일수 (기본 60일)
    
    Returns:
        DataFrame: 예측에 사용할 최근 데이터
    """
    # 지표 계산을 위해 여유있게 데이터 가져옴
    start_date = (datetime.now() - timedelta(days=days + 100)).strftime('%Y-%m-%d')
    end_date = datetime.now().strftime('%Y-%m-%d')
    
    print("=" * 50)
    print("[페이퍼트레이딩용 데이터 생성]")
    print(f"최근 {days}일 데이터")
    print("=" * 50)
    
    if symbols is None:
        sp500 = get_sp500_list()
        symbols = sp500['symbol'].tolist()
    
    df = get_multiple_stocks(symbols, start_date, end_date)
    
    if df.empty:
        return pd.DataFrame()
    
    df = add_indicators(df)
    
    sp500 = get_sp500_list()
    sector_map = dict(zip(sp500['symbol'], sp500['sector']))
    df['sector'] = df['symbol'].map(sector_map).fillna('Unknown')
    
    df = df.dropna().reset_index(drop=True)
    
    # 최근 N일만 남기기
    latest_date = df['date'].max()
    cutoff = latest_date - pd.Timedelta(days=days)
    df = df[df['date'] >= cutoff].reset_index(drop=True)
    
    print("=" * 50)
    print(f"✅ 완료: {len(df):,}행, {df['symbol'].nunique()}개 종목")
    print("=" * 50)
    
    return df


# ============================================
# 테스트
# ============================================

if __name__ == "__main__":
    print("\n[테스트] 3개 종목 학습 데이터")
    df = get_training_data(["AAPL", "MSFT", "GOOGL"])
    print(df.head())
    print(f"\n데이터 형태: {df.shape}")
