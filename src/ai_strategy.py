# ============================================
# 파일명: src/ai_strategy.py
# 설명: XGBoost / LightGBM 기반 AI 매매 전략
#
# 역할 요약:
#   ai_data.py가 생성한 피처를 사용해 이진 분류 모델을 학습하고,
#   특정 날짜에 "다음 5일 내 +3% 달성" 확률이 높은 종목을 선정.
#
# 구성:
#   AIStrategy  (XGBoost): 하이브리드 전략의 AI 컴포넌트로 사용
#   LGBStrategy (LightGBM): 비교/앙상블용 (현재 하이브리드에서는 미사용)
#   evaluate_model()     : XGBoost 테스트셋 평가
#   evaluate_lgb_model() : LightGBM 테스트셋 평가
#
# 모델 파라미터 설계 의도:
#   - max_depth=4, num_leaves=10: 과적합 방지를 위한 얕은 트리
#   - n_estimators=1000: 많은 트리로 분산 감소 (early stopping 미사용)
#   - scale_pos_weight=3: 라벨 불균형 보정 (0이 1보다 약 3배 많음)
#   - subsample=0.7, colsample_bytree=0.7: 배깅 효과로 일반화
#
# 종목 선정 논리:
#   predict() → probability ≥ min_probability(0.5) 필터 → 확률 내림차순 Top N
#
# [주의] 데드 코드:
#   - AIStrategy.train()의 y_prob 변수: 계산 후 사용되지 않음
#   - LGBStrategy.train()의 y_prob 변수: 동일하게 미사용
#   - evaluate_model()의 y_prob 변수: 동일하게 미사용
#   - evaluate_lgb_model()의 y_prob 변수: 동일하게 미사용
#   (추후 ROC-AUC 등 추가 평가를 위해 남겨둔 것으로 보임)
#
# 의존 관계:
#   ← ai_data.py의 피처 DataFrame
#   → hybrid_strategy.py (HybridStrategy 내부에서 AIStrategy 사용)
#   → ai_backtest.py (run_ai_backtest의 strategy 인자로 전달)
# ============================================

import pandas as pd
import numpy as np
import pickle
import os
from datetime import datetime

# XGBoost
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("⚠️ XGBoost 설치 필요: pip install xgboost")

# LightGBM
try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    print("⚠️ LightGBM 설치 필요: pip install lightgbm")

# sklearn
try:
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import (
        accuracy_score, 
        precision_score, 
        recall_score, 
        f1_score,
        confusion_matrix,
        classification_report
    )
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("⚠️ scikit-learn 설치 필요: pip install scikit-learn")


# ============================================
# [1] 설정
# ============================================

# 모델 파라미터 (최적화된 값 - n_estimators=1000)
XGB_PARAMS = {
    'objective': 'binary:logistic',  # 이진 분류
    'eval_metric': 'logloss',
    'max_depth': 3,                  # 트리 깊이 (얕은 트리 → 과적합 방지)
    'learning_rate': 0.03,           # 학습률
    'n_estimators': 1000,            # ⭐ 트리 개수 (최적값!)
    'min_child_weight': 5,           # 과적합 방지
    'subsample': 0.5,                # 데이터 샘플링 (강한 배깅)
    'colsample_bytree': 0.5,         # 피처 샘플링 (강한 배깅)
    'scale_pos_weight': 3,           # 클래스 가중치
    'random_state': 42,
    'n_jobs': -1,                    # 병렬 처리
}

# LightGBM 파라미터 (최적화된 값 - num_leaves=10)
LGB_PARAMS = {
    'objective': 'binary',
    'metric': 'binary_logloss',
    'max_depth': 4,
    'learning_rate': 0.03,
    'n_estimators': 1000,            # ⭐ 트리 개수
    'num_leaves': 10,                # ⭐ 핵심! 단순한 모델
    'min_child_weight': 5,
    'subsample': 0.7,
    'colsample_bytree': 0.7,
    'scale_pos_weight': 3,
    'random_state': 42,
    'n_jobs': -1,
    'verbose': -1,                   # 로그 숨김
}

# 종목 선정
TOP_N = 3
ALLOCATIONS = [0.4, 0.3, 0.3]
MIN_PROBABILITY = 0.5  # 최소 확률 (이상이어야 매수)

# 모델 저장 경로
MODEL_PATH = "models/xgb_model.pkl"


# ============================================
# [2] AI 전략 클래스
# ============================================

class AIStrategy:
    """
    XGBoost 기반 AI 매매 전략
    
    사용 예시:
        strategy = AIStrategy()
        strategy.train(train_df, feature_cols)
        picks = strategy.predict(test_df, feature_cols)
    """
    
    def __init__(self, params=None, top_n=TOP_N, allocations=ALLOCATIONS, 
                 min_probability=MIN_PROBABILITY):
        """
        전략 초기화
        
        Args:
            params: XGBoost 파라미터 (None이면 기본값)
            top_n: 선정할 종목 수
            allocations: 종목별 투자 비중
            min_probability: 최소 매수 확률
        """
        if not XGBOOST_AVAILABLE:
            raise ImportError("XGBoost가 설치되어 있지 않습니다.")
        
        self.params = params or XGB_PARAMS.copy()
        self.top_n = top_n
        self.allocations = allocations
        self.min_probability = min_probability
        
        self.model = None
        self.feature_cols = None
        self.feature_importance = None
        self.train_metrics = None
    
    # ============================================
    # [3] 모델 학습
    # ============================================
    
    def train(self, train_df, feature_cols, valid_ratio=0.2):
        """
        XGBoost 모델 학습
        
        Args:
            train_df: 학습 데이터
            feature_cols: 피처 컬럼 리스트
            valid_ratio: 검증 데이터 비율
        
        Returns:
            dict: 학습 결과 메트릭
        """
        print("=" * 60)
        print("XGBoost 모델 학습")
        print("=" * 60)
        
        self.feature_cols = feature_cols
        
        # 데이터 준비
        X = train_df[feature_cols].values
        y = train_df['label'].values
        
        print(f"학습 데이터: {len(X):,}개")
        print(f"피처 수: {len(feature_cols)}")
        print(f"라벨 분포: 0={sum(y==0):,}, 1={sum(y==1):,}")
        
        # 학습/검증 분리
        X_train, X_valid, y_train, y_valid = train_test_split(
            X, y, test_size=valid_ratio, random_state=42, stratify=y
        )
        
        print(f"\n학습셋: {len(X_train):,}개")
        print(f"검증셋: {len(X_valid):,}개")
        
        # 모델 학습
        print("\n모델 학습 중...")
        
        early_stopping = self.params.pop('early_stopping_rounds', 20)
        
        self.model = xgb.XGBClassifier(**self.params)
        
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_valid, y_valid)],
            verbose=False
        )
        
        # 검증 성능
        y_pred = self.model.predict(X_valid)
        # [주의] y_prob는 현재 사용되지 않음 (미래에 ROC-AUC 등 추가 시 활용 가능)
        y_prob = self.model.predict_proba(X_valid)[:, 1]

        metrics = {
            'accuracy': accuracy_score(y_valid, y_pred),
            'precision': precision_score(y_valid, y_pred, zero_division=0),
            'recall': recall_score(y_valid, y_pred, zero_division=0),
            'f1': f1_score(y_valid, y_pred, zero_division=0)
        }

        self.train_metrics = metrics

        print("\n✅ 학습 완료!")
        print(f"\n📊 검증 성능:")
        print(f"  Accuracy:  {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")  # 매수 신호 정확도 (실제 승률과 직결)
        print(f"  Recall:    {metrics['recall']:.4f}")
        print(f"  F1 Score:  {metrics['f1']:.4f}")
        
        # 피처 중요도
        self.feature_importance = pd.DataFrame({
            'feature': feature_cols,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print(f"\n🔍 Top 10 중요 피처:")
        for i, row in self.feature_importance.head(10).iterrows():
            print(f"  {row['feature']:20} : {row['importance']:.4f}")
        
        return metrics
    
    # ============================================
    # [4] 예측 및 종목 선정
    # ============================================
    
    def predict(self, df, feature_cols=None, date=None):
        """
        종목 예측 및 매수 추천
        
        Args:
            df: 예측용 데이터
            feature_cols: 피처 컬럼 (None이면 학습 시 사용한 것)
            date: 특정 날짜만 예측 (None이면 전체)
        
        Returns:
            DataFrame: 종목별 예측 확률
        """
        if self.model is None:
            raise ValueError("모델이 학습되지 않았습니다. train() 먼저 실행하세요.")
        
        feature_cols = feature_cols or self.feature_cols
        
        # 특정 날짜 필터링
        if date is not None:
            df = df[df['date'] == pd.Timestamp(date)].copy()
        
        if df.empty:
            return pd.DataFrame()
        
        # 예측
        X = df[feature_cols].values
        probs = self.model.predict_proba(X)[:, 1]
        
        # 결과 정리
        result = df[['date', 'symbol', 'close']].copy()
        result['probability'] = probs
        result['prediction'] = (probs >= self.min_probability).astype(int)
        
        return result.sort_values('probability', ascending=False)
    
    def select_stocks(self, df, feature_cols=None, date=None):
        """
        매수할 종목 선정 (모멘텀 전략과 동일한 인터페이스)
        
        Args:
            df: 예측용 데이터
            feature_cols: 피처 컬럼
            date: 예측 날짜
        
        Returns:
            dict: {
                'picks': [종목 리스트],
                'scores': [확률 리스트],
                'allocations': [비중 리스트]
            } 또는 None
        """
        pred_df = self.predict(df, feature_cols, date)
        
        if pred_df.empty:
            return None
        
        # 최소 확률 이상인 종목만
        candidates = pred_df[pred_df['probability'] >= self.min_probability]
        
        if candidates.empty:
            return None
        
        # Top N 선정
        top_picks = candidates.head(self.top_n)
        
        n_picks = len(top_picks)
        if n_picks == 0:
            return None
        
        # 비중 계산
        if n_picks >= 3:
            allocations = self.allocations[:3]
        elif n_picks == 2:
            allocations = [0.5, 0.5]
        elif n_picks == 1:
            allocations = [1.0]
        
        return {
            'picks': top_picks['symbol'].tolist(),
            'scores': top_picks['probability'].tolist(),
            'allocations': allocations[:n_picks],
            'prices': dict(zip(top_picks['symbol'], top_picks['close']))
        }
    
    # ============================================
    # [5] 모델 저장/로드
    # ============================================
    
    def save(self, path=MODEL_PATH):
        """
        모델 저장
        
        Args:
            path: 저장 경로
        """
        if self.model is None:
            raise ValueError("저장할 모델이 없습니다.")
        
        # 디렉토리 생성
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        save_data = {
            'model': self.model,
            'feature_cols': self.feature_cols,
            'feature_importance': self.feature_importance,
            'train_metrics': self.train_metrics,
            'params': self.params,
            'top_n': self.top_n,
            'allocations': self.allocations,
            'min_probability': self.min_probability,
            'saved_at': datetime.now().isoformat()
        }
        
        with open(path, 'wb') as f:
            pickle.dump(save_data, f)
        
        print(f"✅ 모델 저장 완료: {path}")
    
    def load(self, path=MODEL_PATH):
        """
        모델 로드
        
        Args:
            path: 로드 경로
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"모델 파일이 없습니다: {path}")
        
        with open(path, 'rb') as f:
            save_data = pickle.load(f)
        
        self.model = save_data['model']
        self.feature_cols = save_data['feature_cols']
        self.feature_importance = save_data['feature_importance']
        self.train_metrics = save_data['train_metrics']
        self.params = save_data['params']
        self.top_n = save_data['top_n']
        self.allocations = save_data['allocations']
        self.min_probability = save_data['min_probability']
        
        print(f"✅ 모델 로드 완료: {path}")
        print(f"  저장 시점: {save_data.get('saved_at', 'Unknown')}")
        print(f"  Precision: {self.train_metrics.get('precision', 0):.4f}")


# ============================================
# [6] 평가 함수
# ============================================

def evaluate_model(strategy, test_df, feature_cols):
    """
    테스트 데이터로 모델 평가
    
    Args:
        strategy: AIStrategy 인스턴스
        test_df: 테스트 데이터
        feature_cols: 피처 컬럼
    
    Returns:
        dict: 평가 메트릭
    """
    print("=" * 60)
    print("모델 평가 (테스트셋)")
    print("=" * 60)
    
    pred_df = strategy.predict(test_df, feature_cols)

    y_true = test_df['label'].values
    y_pred = (pred_df['probability'] >= strategy.min_probability).astype(int).values
    # [주의] y_prob는 현재 사용되지 않음 (미래에 ROC-AUC 등 추가 시 활용 가능)
    y_prob = pred_df['probability'].values

    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0)
    }

    print(f"\n📊 테스트 성능:")
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")  # 실제 매매 승률과 직결
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1 Score:  {metrics['f1']:.4f}")

    # Confusion Matrix: TP=실제로 상승했고 매수 신호, FP=매수 신호인데 실제 안 오름
    cm = confusion_matrix(y_true, y_pred)
    print(f"\n📊 Confusion Matrix:")
    print(f"  TN={cm[0,0]:,}  FP={cm[0,1]:,}")
    print(f"  FN={cm[1,0]:,}  TP={cm[1,1]:,}")
    
    # 매수 신호 분석
    buy_signals = pred_df[pred_df['prediction'] == 1]
    print(f"\n📊 매수 신호 분석:")
    print(f"  총 매수 신호: {len(buy_signals):,}개")
    
    if len(buy_signals) > 0:
        actual_wins = test_df.loc[buy_signals.index, 'label'].sum()
        win_rate = actual_wins / len(buy_signals) * 100
        print(f"  실제 +3% 달성: {actual_wins:,}개 ({win_rate:.1f}%)")
        
        avg_future_ret = test_df.loc[buy_signals.index, 'future_ret'].mean() * 100
        print(f"  평균 수익률: {avg_future_ret:.2f}%")
    
    return metrics


# ============================================
# [7] LightGBM 전략 클래스
# ============================================

class LGBStrategy:
    """
    LightGBM 기반 AI 매매 전략
    
    사용 예시:
        strategy = LGBStrategy()
        strategy.train(train_df, feature_cols)
        picks = strategy.predict(test_df, feature_cols)
    """
    
    def __init__(self, params=None, top_n=TOP_N, allocations=ALLOCATIONS, 
                 min_probability=MIN_PROBABILITY):
        """
        전략 초기화
        """
        if not LIGHTGBM_AVAILABLE:
            raise ImportError("LightGBM이 설치되어 있지 않습니다.")
        
        self.params = params or LGB_PARAMS.copy()
        self.top_n = top_n
        self.allocations = allocations
        self.min_probability = min_probability
        
        self.model = None
        self.feature_cols = None
        self.feature_importance = None
        self.train_metrics = None
    
    def train(self, train_df, feature_cols, valid_ratio=0.2):
        """
        LightGBM 모델 학습 (AIStrategy.train()과 동일한 로직, LightGBM 버전)
        """
        print("=" * 60)
        print("LightGBM 모델 학습")
        print("=" * 60)

        self.feature_cols = feature_cols

        # 데이터 준비
        X = train_df[feature_cols].values
        y = train_df['label'].values

        print(f"학습 데이터: {len(X):,}개")
        print(f"피처 수: {len(feature_cols)}")
        print(f"라벨 분포: 0={sum(y==0):,}, 1={sum(y==1):,}")

        # 학습/검증 분리 (stratify=y: 라벨 비율 유지)
        X_train, X_valid, y_train, y_valid = train_test_split(
            X, y, test_size=valid_ratio, random_state=42, stratify=y
        )

        print(f"\n학습셋: {len(X_train):,}개")
        print(f"검증셋: {len(X_valid):,}개")

        # 모델 학습
        print("\n모델 학습 중...")

        self.model = lgb.LGBMClassifier(**self.params)

        self.model.fit(
            X_train, y_train,
            eval_set=[(X_valid, y_valid)],
        )

        # 검증 성능
        y_pred = self.model.predict(X_valid)
        # [주의] y_prob는 현재 사용되지 않음 (미래에 ROC-AUC 등 추가 시 활용 가능)
        y_prob = self.model.predict_proba(X_valid)[:, 1]
        
        metrics = {
            'accuracy': accuracy_score(y_valid, y_pred),
            'precision': precision_score(y_valid, y_pred, zero_division=0),
            'recall': recall_score(y_valid, y_pred, zero_division=0),
            'f1': f1_score(y_valid, y_pred, zero_division=0)
        }
        
        self.train_metrics = metrics
        
        print("\n✅ 학습 완료!")
        print(f"\n📊 검증 성능:")
        print(f"  Accuracy:  {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall:    {metrics['recall']:.4f}")
        print(f"  F1 Score:  {metrics['f1']:.4f}")
        
        # 피처 중요도
        self.feature_importance = pd.DataFrame({
            'feature': feature_cols,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print(f"\n🔍 Top 10 중요 피처:")
        for i, row in self.feature_importance.head(10).iterrows():
            print(f"  {row['feature']:20} : {row['importance']:.4f}")
        
        return metrics
    
    def predict(self, df, feature_cols=None, date=None):
        """
        종목 예측 및 매수 추천
        """
        if self.model is None:
            raise ValueError("모델이 학습되지 않았습니다.")
        
        feature_cols = feature_cols or self.feature_cols
        
        if date is not None:
            df = df[df['date'] == pd.Timestamp(date)].copy()
        
        if df.empty:
            return pd.DataFrame()
        
        X = df[feature_cols].values
        probs = self.model.predict_proba(X)[:, 1]
        
        result = df[['date', 'symbol', 'close']].copy()
        result['probability'] = probs
        result['prediction'] = (probs >= self.min_probability).astype(int)
        
        return result.sort_values('probability', ascending=False)
    
    def select_stocks(self, df, feature_cols=None, date=None):
        """
        매수할 종목 선정
        """
        pred_df = self.predict(df, feature_cols, date)
        
        if pred_df.empty:
            return None
        
        candidates = pred_df[pred_df['probability'] >= self.min_probability]
        
        if candidates.empty:
            return None
        
        top_picks = candidates.head(self.top_n)
        
        n_picks = len(top_picks)
        if n_picks == 0:
            return None
        
        if n_picks >= 3:
            allocations = self.allocations[:3]
        elif n_picks == 2:
            allocations = [0.5, 0.5]
        elif n_picks == 1:
            allocations = [1.0]
        
        return {
            'picks': top_picks['symbol'].tolist(),
            'scores': top_picks['probability'].tolist(),
            'allocations': allocations[:n_picks],
            'prices': dict(zip(top_picks['symbol'], top_picks['close']))
        }
    
    def save(self, path="models/lgb_model.pkl"):
        """모델 저장"""
        if self.model is None:
            raise ValueError("저장할 모델이 없습니다.")
        
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        save_data = {
            'model': self.model,
            'feature_cols': self.feature_cols,
            'feature_importance': self.feature_importance,
            'train_metrics': self.train_metrics,
            'params': self.params,
            'saved_at': datetime.now().isoformat()
        }
        
        with open(path, 'wb') as f:
            pickle.dump(save_data, f)
        
        print(f"✅ 모델 저장 완료: {path}")
    
    def load(self, path="models/lgb_model.pkl"):
        """모델 로드"""
        if not os.path.exists(path):
            raise FileNotFoundError(f"모델 파일이 없습니다: {path}")
        
        with open(path, 'rb') as f:
            save_data = pickle.load(f)
        
        self.model = save_data['model']
        self.feature_cols = save_data['feature_cols']
        self.feature_importance = save_data['feature_importance']
        self.train_metrics = save_data['train_metrics']
        self.params = save_data['params']
        
        print(f"✅ 모델 로드 완료: {path}")


def evaluate_lgb_model(strategy, test_df, feature_cols):
    """
    LightGBM 테스트 데이터 평가
    """
    print("=" * 60)
    print("LightGBM 모델 평가 (테스트셋)")
    print("=" * 60)
    
    pred_df = strategy.predict(test_df, feature_cols)

    y_true = test_df['label'].values
    y_pred = (pred_df['probability'] >= strategy.min_probability).astype(int).values
    # [주의] y_prob는 현재 사용되지 않음 (미래에 ROC-AUC 등 추가 시 활용 가능)
    y_prob = pred_df['probability'].values

    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0)
    }

    print(f"\n📊 테스트 성능:")
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1 Score:  {metrics['f1']:.4f}")

    cm = confusion_matrix(y_true, y_pred)
    print(f"\n📊 Confusion Matrix:")
    print(f"  TN={cm[0,0]:,}  FP={cm[0,1]:,}")
    print(f"  FN={cm[1,0]:,}  TP={cm[1,1]:,}")
    
    buy_signals = pred_df[pred_df['prediction'] == 1]
    print(f"\n📊 매수 신호 분석:")
    print(f"  총 매수 신호: {len(buy_signals):,}개")
    
    if len(buy_signals) > 0:
        actual_wins = test_df.loc[buy_signals.index, 'label'].sum()
        win_rate = actual_wins / len(buy_signals) * 100
        print(f"  실제 +3% 달성: {actual_wins:,}개 ({win_rate:.1f}%)")
        
        avg_future_ret = test_df.loc[buy_signals.index, 'future_ret'].mean() * 100
        print(f"  평균 수익률: {avg_future_ret:.2f}%")
    
    return metrics


# ============================================
# [8] 테스트
# ============================================

if __name__ == "__main__":
    print("AI Strategy 모듈 테스트")
    print("=" * 60)
    
    if XGBOOST_AVAILABLE:
        print("✅ XGBoost 사용 가능")
    else:
        print("❌ XGBoost 설치 필요")
    
    if LIGHTGBM_AVAILABLE:
        print("✅ LightGBM 사용 가능")
    else:
        print("❌ LightGBM 설치 필요")
    
    print("\n사용법:")
    print("  from ai_strategy import AIStrategy, LGBStrategy")
    print("  from ai_strategy import evaluate_model, evaluate_lgb_model")