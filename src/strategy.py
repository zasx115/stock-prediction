# ============================================
# 파일명: src/strategy.py
# 설명: 매매 전략 (모멘텀 기반)
# ============================================

import pandas as pd
import numpy as np


# ============================================
# 설정 (수정하기 쉽게 변수로 분리)
# ============================================

# ----- 모멘텀 점수 가중치 -----
WEIGHT_2DAY = 3.5    # 2일전 수익률 가중치
WEIGHT_4DAY = 2.5    # 4일전 수익률 가중치
WEIGHT_6DAY = 1.5    # 6일전 수익률 가중치

# ----- 종목 선정 -----
TOP_N = 3            # 상위 몇 개 종목 선정
ALLOCATIONS = [0.4, 0.3, 0.3]  # 투자 비중 (1위, 2위, 3위)

# ----- 필터링 조건 -----
MIN_SCORE = 0.01     # 최소 점수 (이 점수 이상이어야 매수)
                     # 0.01 = 1% 수익률 기준

# ----- 시장 필터링 -----
MARKET_FILTER = True  # 시장 필터 사용 여부 (False면 항상 매수)


# ============================================
# 1. 모멘텀 점수 계산
# ============================================

def calculate_momentum_score(df):
    """
    각 종목의 모멘텀 점수를 계산합니다.
    
    공식:
    score = (2일전 수익률 × 3.5) + (4일전 수익률 × 2.5) + (6일전 수익률 × 1.5)
    
    Args:
        df: 주가 데이터 (date, symbol, close 필수)
    
    Returns:
        DataFrame: symbol, score, return_2d, return_4d, return_6d
    """
    df = df.copy()
    df = df.sort_values(['symbol', 'date']).reset_index(drop=True)
    
    results = []
    
    for symbol in df['symbol'].unique():
        stock = df[df['symbol'] == symbol].copy()
        
        # 최소 7일 데이터 필요
        if len(stock) < 7:
            continue
        
        # 최신 데이터 (오늘)
        today = stock.iloc[-1]
        today_close = today['close']
        today_date = today['date']
        
        # N일전 종가 가져오기
        close_2d = stock.iloc[-3]['close']  # 2일전
        close_4d = stock.iloc[-5]['close']  # 4일전
        close_6d = stock.iloc[-7]['close']  # 6일전
        
        # 수익률 계산
        return_2d = (today_close - close_2d) / close_2d
        return_4d = (today_close - close_4d) / close_4d
        return_6d = (today_close - close_6d) / close_6d
        
        # 모멘텀 점수
        score = (return_2d * WEIGHT_2DAY) + (return_4d * WEIGHT_4DAY) + (return_6d * WEIGHT_6DAY)
        
        results.append({
            'date': today_date,
            'symbol': symbol,
            'close': today_close,
            'return_2d': return_2d,
            'return_4d': return_4d,
            'return_6d': return_6d,
            'score': score
        })
    
    # 결과가 없으면 빈 DataFrame 반환 (컬럼은 유지)
    if not results:
        return pd.DataFrame(columns=['date', 'symbol', 'close', 'return_2d', 'return_4d', 'return_6d', 'score'])
    
    result_df = pd.DataFrame(results)
    result_df = result_df.sort_values('score', ascending=False).reset_index(drop=True)
    
    return result_df


# ============================================
# 2. 시장 필터링 (S&P 500 평균 수익률 + SPY)
# ============================================

def check_market_condition(df):
    """
    시장 전체 상황을 체크합니다.
    1. SPY 수익률 확인
    2. S&P 500 전체 종목의 평균 일일 수익률이 양수인지 확인
    
    Args:
        df: 주가 데이터 (date, symbol, close 필수)
    
    Returns:
        tuple: (통과 여부, 평균 수익률, SPY 수익률)
    """
    print("시장 상황 체크 중...")
    
    df = df.copy()
    df = df.sort_values(['symbol', 'date']).reset_index(drop=True)
    
    # ----- SPY 수익률 계산 -----
    spy_return = 0
    if 'SPY' in df['symbol'].unique():
        spy = df[df['symbol'] == 'SPY']
        if len(spy) >= 2:
            spy_today = spy.iloc[-1]['close']
            spy_yesterday = spy.iloc[-2]['close']
            spy_return = (spy_today - spy_yesterday) / spy_yesterday
    
    # ----- 전체 종목 평균 수익률 계산 -----
    returns = []
    
    for symbol in df['symbol'].unique():
        stock = df[df['symbol'] == symbol]
        
        if len(stock) < 2:
            continue
        
        # 오늘 일일 수익률
        today_close = stock.iloc[-1]['close']
        yesterday_close = stock.iloc[-2]['close']
        daily_return = (today_close - yesterday_close) / yesterday_close
        
        returns.append(daily_return)
    
    avg_return = np.mean(returns) if returns else 0
    is_positive = avg_return > 0
    
    # 출력
    print(f"  SPY 수익률: {spy_return:.4f} ({spy_return*100:.2f}%)")
    print(f"  시장 평균 수익률: {avg_return:.4f} ({avg_return*100:.2f}%)")
    
    status = "✅ 양호 (매수 가능)" if is_positive else "❌ 부정적 (매수 보류)"
    print(f"  시장 상태: {status}")
    
    return is_positive, avg_return, spy_return


# ============================================
# 3. 커스텀 전략 실행 (메인)
# ============================================

def run_custom_strategy(df):
    """
    커스텀 모멘텀 전략을 실행합니다.
    
    실행 순서:
    1. 모멘텀 점수 계산
    2. 시장 필터링 체크
    3. 최소 점수 필터링
    4. 상위 종목 선정 및 비중 배분
    
    Args:
        df: 주가 데이터
    
    Returns:
        dict: {
            'signal': 'BUY' or 'HOLD',
            'picks': [선정 종목 리스트],
            'allocations': [비중 리스트],
            'pick_scores': [선정 종목 점수 리스트],
            'scores': DataFrame (전체 점수),
            'market_return': 시장 평균 수익률,
            'spy_return': SPY 수익률,
            'reason': 판단 이유
        }
    """
    print("=" * 50)
    print("[커스텀 전략 실행]")
    print("=" * 50)
    
    result = {
        'signal': 'HOLD',
        'picks': [],
        'allocations': [],
        'pick_scores': [],
        'scores': None,
        'market_return': 0,
        'spy_return': 0,
        'reason': ''
    }
    
    # 1. 모멘텀 점수 계산
    scores_df = calculate_momentum_score(df)
    result['scores'] = scores_df
    
    if scores_df.empty:
        result['reason'] = '점수 계산 실패 (데이터 부족)'
        print(f"결과: {result['reason']}")
        return result
    
    # 2. 시장 필터링
    if MARKET_FILTER:
        is_positive, avg_return, spy_return = check_market_condition(df)
        result['market_return'] = avg_return
        result['spy_return'] = spy_return
        
        if not is_positive:
            result['reason'] = f'시장 부정적 (평균 수익률: {avg_return*100:.2f}%)'
            print(f"\n결과: HOLD - {result['reason']}")
            return result
    
    # 3. 상위 종목 추출
    top_stocks = scores_df.head(TOP_N)
    
    # 4. 최소 점수 필터링
    qualified = top_stocks[top_stocks['score'] >= MIN_SCORE]
    
    if len(qualified) == 0:
        result['reason'] = f'최소 점수 미달 (기준: {MIN_SCORE}, 최고점: {top_stocks.iloc[0]["score"]:.4f})'
        print(f"\n결과: HOLD - {result['reason']}")
        return result
    
    # 5. 매수 신호 생성
    result['signal'] = 'BUY'
    result['picks'] = qualified['symbol'].tolist()
    result['pick_scores'] = qualified['score'].tolist()
    
    # 비중 배분 (종목 수에 맞게)
    n_picks = len(result['picks'])
    if n_picks >= 3:
        result['allocations'] = ALLOCATIONS[:3]
    elif n_picks == 2:
        result['allocations'] = [0.5, 0.5]
    else:
        result['allocations'] = [1.0]
    
    result['reason'] = f'조건 충족 (시장 양호, 점수 충족)'
    
    # 결과 출력
    print("\n" + "=" * 50)
    print("결과: BUY")
    print("=" * 50)
    print(f"선정 종목:")
    for i, (symbol, score, alloc) in enumerate(zip(result['picks'], result['pick_scores'], result['allocations'])):
        print(f"  {i+1}위: {symbol} (점수: {score:.4f}, 비중: {alloc*100:.0f}%)")
    
    return result


# ============================================
# 4. AI 전략 (나중에 구현)
# ============================================

def run_ai_strategy(df, model=None):
    """
    [추후 구현]
    AI 모델 기반 전략을 실행합니다.
    """
    print("=" * 50)
    print("[AI 전략 실행]")
    print("=" * 50)
    print("⚠️ 아직 구현되지 않았습니다.")
    
    return {
        'signal': 'HOLD',
        'picks': [],
        'allocations': [],
        'pick_scores': [],
        'scores': None,
        'reason': 'AI 모델 미구현'
    }


# ============================================
# 5. 하이브리드 전략 (나중에 구현)
# ============================================

def run_hybrid_strategy(df, model=None):
    """
    [추후 구현]
    커스텀 + AI 결합 전략을 실행합니다.
    """
    print("=" * 50)
    print("[하이브리드 전략 실행]")
    print("=" * 50)
    print("⚠️ 아직 구현되지 않았습니다.")
    
    return {
        'signal': 'HOLD',
        'picks': [],
        'allocations': [],
        'pick_scores': [],
        'scores': None,
        'reason': '하이브리드 전략 미구현'
    }


# ============================================
# 6. 전략 실행 (통합)
# ============================================

def run_strategy(df, strategy_type='custom', model=None):
    """
    전략을 실행합니다.
    
    Args:
        df: 주가 데이터
        strategy_type: 'custom', 'ai', 'hybrid' 중 선택
        model: AI 모델 (ai, hybrid 전략 시 필요)
    
    Returns:
        dict: 전략 실행 결과
    """
    if strategy_type == 'custom':
        return run_custom_strategy(df)
    elif strategy_type == 'ai':
        return run_ai_strategy(df, model)
    elif strategy_type == 'hybrid':
        return run_hybrid_strategy(df, model)
    else:
        print(f"❌ 알 수 없는 전략: {strategy_type}")
        return {'signal': 'HOLD', 'reason': '알 수 없는 전략'}


# ============================================
# 테스트
# ============================================

if __name__ == "__main__":
    print("\n[테스트] 전략 실행")
    print("실제 테스트는 Colab에서 data.py와 함께 실행하세요.")
