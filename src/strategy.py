# ============================================
# 파일명: src/strategy.py
# 설명: 매매 전략 (커스텀 전략)
# 
# 전략 요약:
# - SPY 상관관계 > 0.5 필터
# - 중장기 모멘텀 (1개월, 3개월, 6개월)
# - Top 3 종목 선정
# ============================================

import pandas as pd
import numpy as np


# ============================================
# [설정] 전략 파라미터
# ============================================

# ----- 중장기 모멘텀 가중치 -----
WEIGHT_1M = 3.5              # 1개월 수익률 가중치
WEIGHT_3M = 2.5              # 3개월 수익률 가중치
WEIGHT_6M = 1.5              # 6개월 수익률 가중치

# ----- 포트폴리오 구성 -----
TOP_N = 3                    # 상위 N개 종목 선정
ALLOCATIONS = [0.4, 0.3, 0.3]  # 투자 비중

# ----- SPY 상관관계 필터 -----
CORRELATION_PERIOD = 60      # 상관관계 계산 기간 (60일)
CORRELATION_THRESHOLD = 0.5  # 최소 상관관계


# ============================================
# [1] 커스텀 전략 클래스
# ============================================

class CustomStrategy:
    """
    커스텀 모멘텀 전략
    
    전략 로직:
    1. SPY와 상관관계 > 0.5인 종목만 필터링
    2. 중장기 모멘텀 점수 계산 (1개월, 3개월, 6개월)
    3. Top 3 종목 선정
    
    사용 예시:
        strategy = CustomStrategy()
        picks = strategy.select_stocks(price_df, date)
    """
    
    def __init__(self, 
                 weight_1m=WEIGHT_1M, 
                 weight_3m=WEIGHT_3M, 
                 weight_6m=WEIGHT_6M,
                 top_n=TOP_N,
                 allocations=ALLOCATIONS,
                 correlation_period=CORRELATION_PERIOD,
                 correlation_threshold=CORRELATION_THRESHOLD):
        """
        전략 초기화
        
        Args:
            weight_1m: 1개월 수익률 가중치
            weight_3m: 3개월 수익률 가중치
            weight_6m: 6개월 수익률 가중치
            top_n: 선정할 종목 수
            allocations: 종목별 투자 비중
            correlation_period: 상관관계 계산 기간
            correlation_threshold: 최소 상관관계
        """
        self.weight_1m = weight_1m
        self.weight_3m = weight_3m
        self.weight_6m = weight_6m
        self.top_n = top_n
        self.allocations = allocations
        self.correlation_period = correlation_period
        self.correlation_threshold = correlation_threshold
        
        # 캐시 (계산 결과 저장)
        self._correlation_df = None
        self._score_df = None
        self._ret_1m = None
    
    # ============================================
    # [2] 상관관계 계산
    # ============================================
    
    def calc_correlation(self, price_df):
        """
        각 종목과 SPY의 상관관계 계산
        
        Args:
            price_df: 피벗된 종가 테이블 (날짜 × 종목)
        
        Returns:
            DataFrame: 날짜별 종목별 상관관계 (-1 ~ +1)
        """
        if 'SPY' not in price_df.columns:
            return pd.DataFrame()
        
        # 일별 수익률
        returns = price_df.pct_change()
        spy_returns = returns['SPY']
        
        # 롤링 상관관계
        correlation_df = pd.DataFrame(index=price_df.index)
        
        for col in returns.columns:
            if col == 'SPY':
                continue
            correlation_df[col] = returns[col].rolling(self.correlation_period).corr(spy_returns)
        
        self._correlation_df = correlation_df
        return correlation_df
    
    def get_high_correlation_stocks(self, date, correlation_df=None):
        """
        특정 날짜에 SPY와 상관관계 높은 종목 리스트 반환
        
        Args:
            date: 조회할 날짜
            correlation_df: 상관관계 DataFrame (없으면 캐시 사용)
        
        Returns:
            list: 상관관계 높은 종목 리스트
        """
        if correlation_df is None:
            correlation_df = self._correlation_df
        
        if correlation_df is None or date not in correlation_df.index:
            return []
        
        corr_values = correlation_df.loc[date].dropna()
        high_corr = corr_values[corr_values > self.correlation_threshold]
        
        return high_corr.index.tolist()
    
    # ============================================
    # [3] 모멘텀 점수 계산
    # ============================================
    
    def calc_momentum_scores(self, weekly_df):
        """
        중장기 모멘텀 점수 계산
        
        공식: (1개월 × 3.5) + (3개월 × 2.5) + (6개월 × 1.5)
        
        Args:
            weekly_df: 화요일만 필터링된 종가 테이블
        
        Returns:
            tuple: (점수 DataFrame, 1개월 수익률 DataFrame)
        """
        # 수익률 계산 (주 1회 데이터 기준)
        ret_1m = weekly_df.pct_change(4)    # 4주 = 1개월
        ret_3m = weekly_df.pct_change(12)   # 12주 = 3개월
        ret_6m = weekly_df.pct_change(24)   # 24주 = 6개월
        
        # 가중 점수
        score_df = (ret_1m * self.weight_1m) + (ret_3m * self.weight_3m) + (ret_6m * self.weight_6m)
        
        self._score_df = score_df
        self._ret_1m = ret_1m
        
        return score_df, ret_1m
    
    # ============================================
    # [4] 종목 선정 (핵심 함수)
    # ============================================
    
    def select_stocks(self, score_df, correlation_df, date, ret_1m=None):
        """
        특정 날짜에 매수할 종목 선정
        
        선정 과정:
        1. 시장 필터 (평균 1개월 수익률 > 0)
        2. 상관관계 필터 (SPY와 > 0.5)
        3. 모멘텀 점수 순위
        4. Top N 선정
        
        Args:
            score_df: 모멘텀 점수 DataFrame
            correlation_df: 상관관계 DataFrame
            date: 선정 날짜
            ret_1m: 1개월 수익률 DataFrame (시장 필터용)
        
        Returns:
            dict: {
                'picks': [종목 리스트],
                'scores': [점수 리스트],
                'allocations': [비중 리스트]
            } 또는 None (매수 안 함)
        """
        date_ts = pd.Timestamp(date)
        
        # ----- 시장 필터 -----
        if ret_1m is not None and date_ts in ret_1m.index:
            market_momentum = ret_1m.loc[date_ts].mean()
            if market_momentum <= 0:
                return None  # 시장 하락 추세 → 매수 안 함
        
        # ----- 점수 가져오기 -----
        if date_ts not in score_df.index:
            return None
        
        current_scores = score_df.loc[date_ts].drop(labels=['SPY'], errors='ignore').dropna()
        
        if current_scores.empty:
            return None
        
        # ----- 상관관계 필터 -----
        high_corr_stocks = self.get_high_correlation_stocks(date_ts, correlation_df)
        
        if high_corr_stocks:
            filtered_scores = current_scores[current_scores.index.isin(high_corr_stocks)]
        else:
            filtered_scores = current_scores
        
        if filtered_scores.empty:
            return None
        
        # ----- Top N 선정 -----
        top_n = filtered_scores.nlargest(min(self.top_n, len(filtered_scores)))
        
        # ----- 비중 계산 -----
        n_picks = len(top_n)
        if n_picks >= 3:
            allocations = self.allocations[:3]
        elif n_picks == 2:
            allocations = [0.5, 0.5]
        elif n_picks == 1:
            allocations = [1.0]
        else:
            return None
        
        return {
            'picks': top_n.index.tolist(),
            'scores': top_n.values.tolist(),
            'allocations': allocations[:n_picks]
        }
    
    # ============================================
    # [5] 전체 프로세스 (편의 함수)
    # ============================================
    
    def prepare(self, price_df, tuesday_df):
        """
        전략 실행 전 데이터 준비
        
        Args:
            price_df: 전체 일별 종가 테이블
            tuesday_df: 화요일만 필터링된 종가 테이블
        
        Returns:
            tuple: (score_df, correlation_df, ret_1m)
        """
        # 상관관계 계산
        correlation_df = self.calc_correlation(price_df)
        
        # 모멘텀 점수 계산
        score_df, ret_1m = self.calc_momentum_scores(tuesday_df)
        
        return score_df, correlation_df, ret_1m
    
    def get_allocations(self, n_picks):
        """
        종목 수에 따른 비중 반환
        
        Args:
            n_picks: 선정된 종목 수
        
        Returns:
            list: 비중 리스트
        """
        if n_picks >= 3:
            return self.allocations[:3]
        elif n_picks == 2:
            return [0.5, 0.5]
        elif n_picks == 1:
            return [1.0]
        else:
            return []


# ============================================
# [6] 유틸리티 함수
# ============================================

def prepare_price_data(df):
    """
    DataFrame을 피벗 테이블로 변환
    
    Args:
        df: 원본 데이터프레임 (date, symbol, close)
    
    Returns:
        DataFrame: 피벗된 종가 테이블
    """
    return df.pivot(index='date', columns='symbol', values='close')


def filter_tuesday(price_df):
    """
    화요일 데이터만 필터링
    
    Args:
        price_df: 피벗된 종가 테이블
    
    Returns:
        DataFrame: 화요일만 포함된 테이블
    """
    price_df = price_df.copy()
    mask = price_df.index.day_name() == 'Tuesday'
    return price_df[mask]


def create_trade_mapping(df):
    """
    화요일 → 수요일 매수일 매핑 생성
    
    Args:
        df: 원본 데이터프레임
    
    Returns:
        dict: {화요일: 수요일} 매핑
    """
    dates = sorted(df['date'].unique())
    date_weekday = {d: pd.Timestamp(d).day_name() for d in dates}
    
    trade_map = {}
    for i, date in enumerate(dates):
        if date_weekday[date] == 'Tuesday':
            for j in range(i+1, len(dates)):
                if date_weekday[dates[j]] == 'Wednesday':
                    trade_map[date] = dates[j]
                    break
    
    return trade_map


# ============================================
# [7] 테스트
# ============================================

if __name__ == "__main__":
    print("CustomStrategy 테스트")
    print("=" * 50)
    
    # 전략 인스턴스 생성
    strategy = CustomStrategy()
    
    print(f"모멘텀 가중치: 1M={strategy.weight_1m}, 3M={strategy.weight_3m}, 6M={strategy.weight_6m}")
    print(f"Top N: {strategy.top_n}")
    print(f"비중: {strategy.allocations}")
    print(f"상관관계 기간: {strategy.correlation_period}일")
    print(f"상관관계 기준: {strategy.correlation_threshold}")
    
    print("\n사용법:")
    print("  from src.strategy import CustomStrategy, prepare_price_data, filter_tuesday")
    print("  strategy = CustomStrategy()")
    print("  price_df = prepare_price_data(df)")
    print("  tuesday_df = filter_tuesday(price_df)")
    print("  score_df, correlation_df, ret_1m = strategy.prepare(price_df, tuesday_df)")
    print("  result = strategy.select_stocks(score_df, correlation_df, date, ret_1m)")
