# ============================================
# 파일명: src/hybrid_strategy.py
# 설명: 하이브리드 전략 (모멘텀 + AI 결합)
#
# 역할 요약:
#   모멘텀 점수와 AI 확률을 가중 평균하여 최종 종목 선정.
#   두 전략의 장점을 결합: 모멘텀의 안정성 + AI의 패턴 인식.
#
# 핵심 알고리즘 (predict 메서드):
#   1. 모멘텀 점수(score_df) 추출 및 0~1 정규화 (min-max scaling)
#      정규화 공식: (x - min) / (max - min + 1e-8)
#   2. AI 확률(probability) 추출 (이미 0~1 범위)
#   3. 두 점수를 가중 합산:
#      hybrid_score = m_score × 0.35 + probability × 0.65
#   4. hybrid_score 내림차순으로 Top N 선정
#
# 가중치 최적화 결과:
#   모멘텀 35% + AI 65% → 수익률 +352.73%, 샤프 2.51
#   (백테스트 기반 최적값, hybrid_trading.py에서도 동일 사용)
#
# 준비 과정 (prepare 메서드):
#   1. AIStrategy(XGBoost) 학습: train_df, feature_cols
#   2. CustomStrategy 모멘텀 계산: price_df의 화요일 필터 → score_df, correlation_df
#
# 주요 클래스/함수:
#   HybridStrategy           → 전략 클래스
#   create_hybrid_strategy() → 팩토리 함수 (초기화 + prepare 일괄)
#
# 의존 관계:
#   ← strategy.py  (CustomStrategy, prepare_price_data, filter_tuesday)
#   ← ai_strategy.py (AIStrategy, XGB_PARAMS)
#   → ai_backtest.py (run_ai_backtest의 strategy 인자로 전달)
#   → hybrid_trading.py (라이브 트레이딩)
#   → run_hybrid_backtest.py (백테스트 실행기)
# ============================================

import pandas as pd
import numpy as np
from datetime import datetime

# 전략 모듈
from strategy import CustomStrategy, prepare_price_data, filter_tuesday
from ai_strategy import AIStrategy, XGB_PARAMS


# ============================================
# [1] 설정
# ============================================

# 가중치 (최적값)
WEIGHT_MOMENTUM = 0.35    # 모멘텀 35%
WEIGHT_AI = 0.65          # AI 65%

# 종목 선정
TOP_N = 3
ALLOCATIONS = [0.4, 0.3, 0.3]
MIN_PROBABILITY = 0.5


# ============================================
# [2] 하이브리드 전략 클래스
# ============================================

class HybridStrategy:
    """
    하이브리드 전략: 모멘텀 점수 + AI 확률 가중 평균
    
    사용 예시:
        strategy = HybridStrategy()
        strategy.prepare(train_df, test_df, features)
        picks = strategy.select_stocks(test_df, features, date)
    """
    
    def __init__(self,
                 weight_momentum=WEIGHT_MOMENTUM,
                 weight_ai=WEIGHT_AI,
                 top_n=TOP_N,
                 allocations=ALLOCATIONS,
                 min_probability=MIN_PROBABILITY,
                 xgb_params=None):
        """
        전략 초기화

        Args:
            weight_momentum: 모멘텀 가중치 (기본 35%)
            weight_ai: AI 가중치 (기본 65%)
            top_n: 선정할 종목 수
            allocations: 종목별 투자 비중
            min_probability: 최소 매수 확률
            xgb_params: XGBoost 파라미터 딕셔너리 (None이면 기본값)
        """
        self.weight_m = weight_momentum
        self.weight_ai = weight_ai
        self.top_n = top_n
        self.allocations = allocations
        self.min_probability = min_probability
        self.xgb_params = xgb_params
        
        # 전략 인스턴스
        self.ai_strategy = None
        self.momentum_strategy = None
        
        # 모멘텀 데이터
        self.score_df = None
        self.correlation_df = None
        self.ret_1m = None
        
        # 피처 컬럼
        self.feature_cols = None
        
        # 상태
        self.is_prepared = False
    
    # ============================================
    # [3] 데이터 준비
    # ============================================
    
    def prepare(self, train_df, price_df, feature_cols):
        """
        전략 실행 전 데이터 준비
        
        Args:
            train_df: AI 학습용 데이터
            price_df: 모멘텀 계산용 가격 데이터 (피벗된 형태)
            feature_cols: AI 피처 컬럼 리스트
        """
        print("=" * 60)
        print("하이브리드 전략 준비")
        print("=" * 60)
        
        self.feature_cols = feature_cols
        
        # ----- AI 전략 준비 -----
        print("\n[1] AI 전략 (XGBoost) 학습...")
        self.ai_strategy = AIStrategy(params=self.xgb_params)
        self.ai_strategy.train(train_df, feature_cols)
        
        # ----- 모멘텀 전략 준비 -----
        print("\n[2] 모멘텀 전략 준비...")
        self.momentum_strategy = CustomStrategy()
        
        tuesday_df = filter_tuesday(price_df)
        self.score_df, self.correlation_df, self.ret_1m = \
            self.momentum_strategy.prepare(price_df, tuesday_df)
        
        print(f"  모멘텀 점수 계산 완료: {len(self.score_df)}일")
        
        self.is_prepared = True
        
        print("\n✅ 하이브리드 전략 준비 완료!")
        print(f"  가중치: 모멘텀 {self.weight_m*100:.0f}% + AI {self.weight_ai*100:.0f}%")
    
    # ============================================
    # [4] 종목 선정
    # ============================================
    
    def predict(self, df, feature_cols=None, date=None):
        """
        종목별 하이브리드 점수 계산
        
        Args:
            df: 예측용 데이터
            feature_cols: 피처 컬럼 (None이면 학습 시 사용한 것)
            date: 예측 날짜
        
        Returns:
            DataFrame: 종목별 점수
        """
        if not self.is_prepared:
            raise ValueError("prepare() 먼저 실행하세요.")
        
        feature_cols = feature_cols or self.feature_cols
        
        if date is None:
            return pd.DataFrame()
        
        date_ts = pd.Timestamp(date)
        date_df = df[df['date'] == date_ts].copy()
        
        if date_df.empty:
            return pd.DataFrame()
        
        # ----- 모멘텀 점수 추출 (SPY 제외) -----
        if date_ts not in self.score_df.index:
            return pd.DataFrame()

        m_scores = self.score_df.loc[date_ts].drop(labels=['SPY'], errors='ignore').dropna()

        if m_scores.empty:
            return pd.DataFrame()

        # ----- AI 확률 예측 (XGBoost) -----
        ai_pred = self.ai_strategy.predict(date_df, feature_cols, date)

        if ai_pred.empty:
            return pd.DataFrame()

        # ----- 모멘텀 점수 정규화 (0~1, min-max scaling) -----
        # AI 확률은 이미 0~1 범위이므로 모멘텀만 정규화하여 동일 스케일로 맞춤
        m_min, m_max = m_scores.min(), m_scores.max()
        m_norm = (m_scores - m_min) / (m_max - m_min + 1e-8)  # 1e-8: 분모 0 방지

        # ----- 두 점수 합치기 (symbol 기준 join) -----
        merged = ai_pred.copy()
        merged['m_score'] = merged['symbol'].map(m_norm)
        merged = merged.dropna()  # 양쪽 점수 모두 있는 종목만 유지

        if merged.empty:
            return pd.DataFrame()

        # ----- 가중 평균으로 최종 hybrid_score 계산 -----
        # hybrid_score = 모멘텀 × 35% + AI확률 × 65%
        merged['hybrid_score'] = (merged['m_score'] * self.weight_m +
                                   merged['probability'] * self.weight_ai)

        # 내림차순 정렬 (가장 높은 점수가 Top 순위)
        merged = merged.sort_values('hybrid_score', ascending=False)
        
        return merged
    
    def select_stocks(self, df, feature_cols=None, date=None):
        """
        매수할 종목 선정
        
        Args:
            df: 예측용 데이터
            feature_cols: 피처 컬럼
            date: 예측 날짜
        
        Returns:
            dict: {
                'picks': [종목 리스트],
                'scores': [점수 리스트],
                'allocations': [비중 리스트],
                'prices': {종목: 가격}
            } 또는 None
        """
        pred_df = self.predict(df, feature_cols, date)
        
        if pred_df.empty:
            return None
        
        # Top N 선정
        top_picks = pred_df.head(self.top_n)
        
        n_picks = len(top_picks)
        if n_picks == 0:
            return None
        
        # 비중 계산
        if n_picks >= 3:
            allocs = self.allocations[:3]
        elif n_picks == 2:
            allocs = [0.5, 0.5]
        else:
            allocs = [1.0]
        
        return {
            'picks': top_picks['symbol'].tolist(),
            'scores': top_picks['hybrid_score'].tolist(),
            'allocations': allocs[:n_picks],
            'prices': dict(zip(top_picks['symbol'], top_picks['close']))
        }
    
    # ============================================
    # [5] 유틸리티
    # ============================================
    
    def get_weights(self):
        """현재 가중치 반환"""
        return {
            'momentum': self.weight_m,
            'ai': self.weight_ai
        }
    
    def set_weights(self, weight_momentum, weight_ai):
        """가중치 변경"""
        if abs(weight_momentum + weight_ai - 1.0) > 0.01:
            raise ValueError("가중치 합이 1이어야 합니다.")
        
        self.weight_m = weight_momentum
        self.weight_ai = weight_ai
        
        print(f"가중치 변경: 모멘텀 {weight_momentum*100:.0f}% + AI {weight_ai*100:.0f}%")


# ============================================
# [6] 간편 실행 함수
# ============================================

def create_hybrid_strategy(train_df, price_df, feature_cols,
                           weight_momentum=WEIGHT_MOMENTUM,
                           weight_ai=WEIGHT_AI,
                           xgb_params=None):
    """
    하이브리드 전략 생성 및 준비

    Args:
        train_df: AI 학습용 데이터
        price_df: 모멘텀 계산용 가격 데이터
        feature_cols: AI 피처 컬럼
        weight_momentum: 모멘텀 가중치
        weight_ai: AI 가중치
        xgb_params: XGBoost 파라미터 딕셔너리 (None이면 기본값)

    Returns:
        HybridStrategy: 준비된 전략 인스턴스

    사용 예시:
        strategy = create_hybrid_strategy(train_df, price_df, features)
        picks = strategy.select_stocks(test_df, features, date)
    """
    strategy = HybridStrategy(
        weight_momentum=weight_momentum,
        weight_ai=weight_ai,
        xgb_params=xgb_params,
    )
    strategy.prepare(train_df, price_df, feature_cols)
    
    return strategy


# ============================================
# [7] 테스트
# ============================================

if __name__ == "__main__":
    print("Hybrid Strategy 모듈")
    print("=" * 60)
    print("\n최적 파라미터:")
    print(f"  모멘텀 가중치: {WEIGHT_MOMENTUM*100:.0f}%")
    print(f"  AI 가중치: {WEIGHT_AI*100:.0f}%")
    print("\n백테스트 성과:")
    print("  수익률: +352.73%")
    print("  승률: 61.2%")
    print("  MDD: -38.87%")
    print("  샤프비율: 2.51")
    print("\n사용법:")
    print("  from hybrid_strategy import HybridStrategy, create_hybrid_strategy")
    print("  strategy = create_hybrid_strategy(train_df, price_df, features)")
    print("  picks = strategy.select_stocks(test_df, features, date)")