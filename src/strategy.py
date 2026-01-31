# ============================================
# 파일명: src/strategy.py
# 설명: 매매 전략 (모멘텀 기반)
# ============================================

import pandas as pd
import numpy as np


# ============================================
# 설정 (수정하기 쉽게 변수로 분리)
# ============================================

WEIGHT_2DAY = 3.5
WEIGHT_4DAY = 2.5
WEIGHT_6DAY = 1.5

TOP_N = 3
ALLOCATIONS = [0.4, 0.3, 0.3]

MIN_SCORE = 0.01
MARKET_FILTER = True


# ============================================
# 1. 모멘텀 점수 계산
# ============================================

def calculate_momentum_score(df):
    """
    각 종목의 모멘텀 점수를 계산합니다.
    """
    df = df.copy()
    df = df.sort_values(['symbol', 'date']).reset_index(drop=True)
    
    results = []
    
    for symbol in df['symbol'].unique():
        stock = df[df['symbol'] == symbol].copy()
        
        if len(stock) < 7:
            continue
        
        today = stock.iloc[-1]
        today_close = today['close']
        today_date = today['date']
        
        close_2d = stock.iloc[-3]['close']
        close_4d = stock.iloc[-5]['close']
        close_6d = stock.iloc[-7]['close']
        
        return_2d = (today_close - close_2d) / close_2d
        return_4d = (today_close - close_4d) / close_4d
        return_6d = (today_close - close_6d) / close_6d
        
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
    
    if len(results) == 0:
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
    """
    df = df.copy()
    df = df.sort_values(['symbol', 'date']).reset_index(drop=True)
    
    spy_return = 0
    if 'SPY' in df['symbol'].unique():
        spy = df[df['symbol'] == 'SPY']
        if len(spy) >= 2:
            spy_today = spy.iloc[-1]['close']
            spy_yesterday = spy.iloc[-2]['close']
            spy_return = (spy_today - spy_yesterday) / spy_yesterday
    
    returns = []
    for symbol in df['symbol'].unique():
        stock = df[df['symbol'] == symbol]
        if len(stock) < 2:
            continue
        today_close = stock.iloc[-1]['close']
        yesterday_close = stock.iloc[-2]['close']
        daily_return = (today_close - yesterday_close) / yesterday_close
        returns.append(daily_return)
    
    avg_return = np.mean(returns) if returns else 0
    is_positive = avg_return > 0
    
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
    
    scores_df = calculate_momentum_score(df)
    result['scores'] = scores_df
    
    if scores_df.empty:
        result['reason'] = '점수 계산 실패 (데이터 부족)'
        print(f"결과: {result['reason']}")
        return result
    
    if MARKET_FILTER:
        is_positive, avg_return, spy_return = check_market_condition(df)
        result['market_return'] = avg_return
        result['spy_return'] = spy_return
        
        if not is_positive:
            result['reason'] = f'시장 부정적 (평균 수익률: {avg_return*100:.2f}%)'
            print(f"\n결과: HOLD - {result['reason']}")
            return result
    
    top_stocks = scores_df.head(TOP_N)
    qualified = top_stocks[top_stocks['score'] >= MIN_SCORE]
    
    if len(qualified) == 0:
        result['reason'] = f'최소 점수 미달 (기준: {MIN_SCORE}, 최고점: {top_stocks.iloc[0]["score"]:.4f})'
        print(f"\n결과: HOLD - {result['reason']}")
        return result
    
    result['signal'] = 'BUY'
    result['picks'] = qualified['symbol'].tolist()
    result['pick_scores'] = qualified['score'].tolist()
    
    n_picks = len(result['picks'])
    if n_picks >= 3:
        result['allocations'] = ALLOCATIONS[:3]
    elif n_picks == 2:
        result['allocations'] = [0.5, 0.5]
    else:
        result['allocations'] = [1.0]
    
    result['reason'] = '조건 충족 (시장 양호, 점수 충족)'
    
    print("\n" + "=" * 50)
    print("결과: BUY")
    print("=" * 50)
    for i, (symbol, score, alloc) in enumerate(zip(result['picks'], result['pick_scores'], result['allocations'])):
        print(f"  {i+1}위: {symbol} (점수: {score:.4f}, 비중: {alloc*100:.0f}%)")
    
    return result


# ============================================
# 4. AI 전략 (나중에 구현)
# ============================================

def run_ai_strategy(df, model=None):
    print("=" * 50)
    print("[AI 전략 실행]")
    print("⚠️ 아직 구현되지 않았습니다.")
    print("=" * 50)
    return {'signal': 'HOLD', 'picks': [], 'allocations': [], 'pick_scores': [], 'scores': None, 'reason': 'AI 모델 미구현'}


# ============================================
# 5. 하이브리드 전략 (나중에 구현)
# ============================================

def run_hybrid_strategy(df, model=None):
    print("=" * 50)
    print("[하이브리드 전략 실행]")
    print("⚠️ 아직 구현되지 않았습니다.")
    print("=" * 50)
    return {'signal': 'HOLD', 'picks': [], 'allocations': [], 'pick_scores': [], 'scores': None, 'reason': '하이브리드 전략 미구현'}


# ============================================
# 6. 전략 실행 (통합)
# ============================================

def run_strategy(df, strategy_type='custom', model=None):
    if strategy_type == 'custom':
        return run_custom_strategy(df)
    elif strategy_type == 'ai':
        return run_ai_strategy(df, model)
    elif strategy_type == 'hybrid':
        return run_hybrid_strategy(df, model)
    else:
        print(f"❌ 알 수 없는 전략: {strategy_type}")
        return {'signal': 'HOLD', 'reason': '알 수 없는 전략'}
