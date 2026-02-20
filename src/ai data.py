# ============================================
# 파일명: src/ai_data.py
# 설명: AI 모델용 피처 엔지니어링
# 
# 기능:
# - 기술적 지표 계산
# - 피처 생성
# - 라벨 생성 (다음 주 +3% 여부)
# - 학습/테스트 데이터 분리
# ============================================

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# 기존 data.py 함수 재사용
from data import get_sp500_list, download_stock_data, get_backtest_data


# ============================================
# [1] 설정
# ============================================

# 학습/테스트 기간
TRAIN_START = "2020-01-01"
TRAIN_END = "2023-12-31"
TEST_START = "2024-01-01"
TEST_END = None  # None = 현재까지

# 라벨 기준
TARGET_RETURN = 0.03  # +3%
TARGET_DAYS = 5       # 5일 (1주일)

# 피처 기간
FEATURE_WINDOWS = [5, 10, 20, 60, 120]  # 이동평균 등


# ============================================
# [2] 기술적 지표 계산
# ============================================

def calc_returns(df, periods=[5, 10, 20, 60, 120]):
    """
    다양한 기간의 수익률 계산
    
    Args:
        df: 종목별 종가 데이터 (pivot된 형태, index=date, columns=symbol)
        periods: 수익률 계산 기간 리스트
    
    Returns:
        dict: {period: DataFrame} 형태의 수익률 데이터
    """
    returns = {}
    for p in periods:
        returns[f'ret_{p}d'] = df.pct_change(p)
    return returns


def calc_moving_averages(df, windows=[5, 20, 60, 120]):
    """
    이동평균 계산
    
    Args:
        df: 종목별 종가 데이터
        windows: 이동평균 기간 리스트
    
    Returns:
        dict: {window: DataFrame} 형태의 이동평균 데이터
    """
    mas = {}
    for w in windows:
        mas[f'ma_{w}d'] = df.rolling(w).mean()
    return mas


def calc_ma_ratios(price_df, ma_dict):
    """
    현재가 대비 이동평균 비율 (price / MA - 1)
    
    Args:
        price_df: 종가 데이터
        ma_dict: calc_moving_averages() 결과
    
    Returns:
        dict: 이동평균 대비 비율
    """
    ratios = {}
    for key, ma in ma_dict.items():
        ratio_key = key.replace('ma_', 'ma_ratio_')
        ratios[ratio_key] = (price_df / ma) - 1
    return ratios


def calc_volatility(df, windows=[5, 20, 60]):
    """
    변동성 (일별 수익률의 표준편차)
    
    Args:
        df: 종가 데이터
        windows: 변동성 계산 기간
    
    Returns:
        dict: 변동성 데이터
    """
    daily_ret = df.pct_change()
    vols = {}
    for w in windows:
        vols[f'vol_{w}d'] = daily_ret.rolling(w).std()
    return vols


def calc_rsi(df, period=14):
    """
    RSI (Relative Strength Index) 계산
    
    Args:
        df: 종가 데이터
        period: RSI 기간 (기본 14일)
    
    Returns:
        DataFrame: RSI 값 (0~100)
    """
    delta = df.diff()
    gain = delta.where(delta > 0, 0)
    loss = (-delta).where(delta < 0, 0)
    
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    
    return rsi


def calc_macd(df, fast=12, slow=26, signal=9):
    """
    MACD 계산
    
    Args:
        df: 종가 데이터
        fast: 단기 EMA 기간
        slow: 장기 EMA 기간
        signal: 시그널 기간
    
    Returns:
        tuple: (MACD, Signal, Histogram)
    """
    ema_fast = df.ewm(span=fast).mean()
    ema_slow = df.ewm(span=slow).mean()
    
    macd = ema_fast - ema_slow
    macd_signal = macd.ewm(span=signal).mean()
    macd_hist = macd - macd_signal
    
    return macd, macd_signal, macd_hist


def calc_bollinger_bands(df, window=20, std_mult=2):
    """
    볼린저 밴드 계산
    
    Args:
        df: 종가 데이터
        window: 이동평균 기간
        std_mult: 표준편차 배수
    
    Returns:
        tuple: (upper, middle, lower, %b)
    """
    middle = df.rolling(window).mean()
    std = df.rolling(window).std()
    
    upper = middle + (std * std_mult)
    lower = middle - (std * std_mult)
    
    # %B: 현재가가 밴드 내 어디에 위치하는지 (0~1)
    pct_b = (df - lower) / (upper - lower)
    
    return upper, middle, lower, pct_b


def calc_volume_features(volume_df, windows=[5, 20]):
    """
    거래량 피처 계산
    
    Args:
        volume_df: 거래량 데이터
        windows: 이동평균 기간
    
    Returns:
        dict: 거래량 피처들
    """
    features = {}
    
    for w in windows:
        vol_ma = volume_df.rolling(w).mean()
        features[f'vol_ratio_{w}d'] = volume_df / vol_ma  # 거래량 / 평균거래량
    
    return features


# ============================================
# [3] SPY 상관관계
# ============================================

def calc_spy_correlation(price_df, window=60):
    """
    각 종목과 SPY의 상관관계 계산
    
    Args:
        price_df: 종가 데이터 (SPY 포함)
        window: 상관관계 계산 기간
    
    Returns:
        DataFrame: 종목별 SPY 상관관계
    """
    if 'SPY' not in price_df.columns:
        return pd.DataFrame()
    
    returns = price_df.pct_change()
    spy_ret = returns['SPY']
    
    corr_df = pd.DataFrame(index=price_df.index)
    
    for col in returns.columns:
        if col == 'SPY':
            continue
        corr_df[col] = returns[col].rolling(window).corr(spy_ret)
    
    return corr_df


# ============================================
# [4] 라벨 생성
# ============================================

def create_labels(price_df, target_return=TARGET_RETURN, target_days=TARGET_DAYS):
    """
    다음 N일 수익률 기반 라벨 생성
    
    Args:
        price_df: 종가 데이터
        target_return: 목표 수익률 (기본 3%)
        target_days: 목표 기간 (기본 5일)
    
    Returns:
        DataFrame: 라벨 (1 = 매수, 0 = 패스)
    """
    # 미래 수익률 (shift로 과거로 당김)
    future_ret = price_df.pct_change(target_days).shift(-target_days)
    
    # 목표 수익률 달성 여부
    labels = (future_ret >= target_return).astype(int)
    
    return labels, future_ret


# ============================================
# [5] 피처 통합 생성
# ============================================

def create_features(df):
    """
    모든 피처를 생성하고 통합
    
    Args:
        df: 원본 데이터프레임 (date, symbol, open, high, low, close, volume)
    
    Returns:
        DataFrame: 종목-날짜별 피처 (long format)
    """
    print("=" * 60)
    print("피처 생성 시작...")
    print("=" * 60)
    
    # 피벗 테이블 생성
    price_df = df.pivot(index='date', columns='symbol', values='close')
    volume_df = df.pivot(index='date', columns='symbol', values='volume')
    high_df = df.pivot(index='date', columns='symbol', values='high')
    low_df = df.pivot(index='date', columns='symbol', values='low')
    
    symbols = [col for col in price_df.columns if col != 'SPY']
    
    print(f"종목 수: {len(symbols)}")
    print(f"기간: {price_df.index.min()} ~ {price_df.index.max()}")
    
    # ----- 피처 계산 -----
    
    # 1. 수익률
    print("  - 수익률 계산...")
    returns = calc_returns(price_df)
    
    # 2. 이동평균
    print("  - 이동평균 계산...")
    mas = calc_moving_averages(price_df)
    ma_ratios = calc_ma_ratios(price_df, mas)
    
    # 3. 변동성
    print("  - 변동성 계산...")
    vols = calc_volatility(price_df)
    
    # 4. RSI
    print("  - RSI 계산...")
    rsi = calc_rsi(price_df)
    
    # 5. MACD
    print("  - MACD 계산...")
    macd, macd_signal, macd_hist = calc_macd(price_df)
    
    # 6. 볼린저 밴드
    print("  - 볼린저 밴드 계산...")
    bb_upper, bb_middle, bb_lower, bb_pct = calc_bollinger_bands(price_df)
    
    # 7. 거래량
    print("  - 거래량 피처 계산...")
    vol_features = calc_volume_features(volume_df)
    
    # 8. SPY 상관관계
    print("  - SPY 상관관계 계산...")
    spy_corr = calc_spy_correlation(price_df)
    
    # 9. 라벨 생성
    print("  - 라벨 생성...")
    labels, future_ret = create_labels(price_df)
    
    # ----- 데이터 통합 (long format) -----
    print("  - 데이터 통합...")
    
    all_features = []
    
    for symbol in symbols:
        if symbol not in price_df.columns:
            continue
        
        feature_df = pd.DataFrame({
            'date': price_df.index,
            'symbol': symbol,
            'close': price_df[symbol].values,
        })
        
        # 수익률 추가
        for key, ret_df in returns.items():
            if symbol in ret_df.columns:
                feature_df[key] = ret_df[symbol].values
        
        # 이동평균 비율 추가
        for key, ratio_df in ma_ratios.items():
            if symbol in ratio_df.columns:
                feature_df[key] = ratio_df[symbol].values
        
        # 변동성 추가
        for key, vol_df in vols.items():
            if symbol in vol_df.columns:
                feature_df[key] = vol_df[symbol].values
        
        # RSI 추가
        if symbol in rsi.columns:
            feature_df['rsi'] = rsi[symbol].values
        
        # MACD 추가
        if symbol in macd.columns:
            feature_df['macd'] = macd[symbol].values
            feature_df['macd_signal'] = macd_signal[symbol].values
            feature_df['macd_hist'] = macd_hist[symbol].values
        
        # 볼린저 밴드 추가
        if symbol in bb_pct.columns:
            feature_df['bb_pct'] = bb_pct[symbol].values
        
        # 거래량 피처 추가
        for key, vf_df in vol_features.items():
            if symbol in vf_df.columns:
                feature_df[key] = vf_df[symbol].values
        
        # SPY 상관관계 추가
        if symbol in spy_corr.columns:
            feature_df['spy_corr'] = spy_corr[symbol].values
        
        # 라벨 추가
        if symbol in labels.columns:
            feature_df['label'] = labels[symbol].values
            feature_df['future_ret'] = future_ret[symbol].values
        
        all_features.append(feature_df)
    
    # 합치기
    final_df = pd.concat(all_features, ignore_index=True)
    
    # NaN 제거
    feature_cols = [col for col in final_df.columns if col not in ['date', 'symbol', 'label', 'future_ret', 'close']]
    final_df = final_df.dropna(subset=feature_cols + ['label'])
    
    print(f"\n✅ 피처 생성 완료!")
    print(f"  총 샘플 수: {len(final_df):,}")
    print(f"  피처 수: {len(feature_cols)}")
    print(f"  피처 목록: {feature_cols}")
    
    return final_df


# ============================================
# [6] 학습/테스트 데이터 분리
# ============================================

def split_train_test(feature_df, train_end=TRAIN_END, test_start=TEST_START):
    """
    학습/테스트 데이터 분리
    
    Args:
        feature_df: 피처 데이터프레임
        train_end: 학습 데이터 종료일
        test_start: 테스트 데이터 시작일
    
    Returns:
        tuple: (train_df, test_df)
    """
    feature_df['date'] = pd.to_datetime(feature_df['date'])
    
    train_df = feature_df[feature_df['date'] <= train_end].copy()
    test_df = feature_df[feature_df['date'] >= test_start].copy()
    
    print(f"\n학습 데이터: {len(train_df):,}개 ({train_df['date'].min()} ~ {train_df['date'].max()})")
    print(f"테스트 데이터: {len(test_df):,}개 ({test_df['date'].min()} ~ {test_df['date'].max()})")
    
    # 라벨 분포
    print(f"\n학습 라벨 분포:")
    print(f"  0 (패스): {(train_df['label'] == 0).sum():,} ({(train_df['label'] == 0).mean()*100:.1f}%)")
    print(f"  1 (매수): {(train_df['label'] == 1).sum():,} ({(train_df['label'] == 1).mean()*100:.1f}%)")
    
    return train_df, test_df


def get_feature_columns(df):
    """
    피처 컬럼 목록 반환
    """
    exclude = ['date', 'symbol', 'label', 'future_ret', 'close']
    return [col for col in df.columns if col not in exclude]


# ============================================
# [7] 전체 파이프라인
# ============================================

def prepare_ai_data(symbols=None, train_start=TRAIN_START, test_end=TEST_END):
    """
    AI 학습용 데이터 전체 준비
    
    Args:
        symbols: 종목 리스트 (None이면 S&P 500)
        train_start: 학습 시작일
        test_end: 테스트 종료일
    
    Returns:
        tuple: (train_df, test_df, feature_columns)
    
    사용 예시 (Colab):
        from ai_data import prepare_ai_data
        train_df, test_df, features = prepare_ai_data()
    """
    # 1. 데이터 다운로드
    print("=" * 60)
    print("AI 데이터 준비")
    print("=" * 60)
    
    df = get_backtest_data(symbols, start_date=train_start, end_date=test_end)
    
    if df.empty:
        print("❌ 데이터 다운로드 실패")
        return None, None, None
    
    # 2. 피처 생성
    feature_df = create_features(df)
    
    # 3. 학습/테스트 분리
    train_df, test_df = split_train_test(feature_df)
    
    # 4. 피처 컬럼 목록
    feature_cols = get_feature_columns(feature_df)
    
    return train_df, test_df, feature_cols


# ============================================
# [8] 테스트
# ============================================

if __name__ == "__main__":
    print("AI Data 모듈 테스트")
    print("=" * 60)
    
    # 소규모 테스트 (5종목)
    test_symbols = ['AAPL', 'MSFT', 'GOOGL', 'NVDA', 'SPY']
    
    train_df, test_df, features = prepare_ai_data(
        symbols=test_symbols,
        train_start="2022-01-01",
        test_end=None
    )
    
    if train_df is not None:
        print(f"\n피처 샘플:")
        print(train_df.head())