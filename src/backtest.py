# ============================================
# 파일명: src/backtest.py
# 설명: 모멘텀 전략 백테스트 (최종본 - 버전 C)
# 
# 전략 요약:
# - SPY 상관관계 > 0.5 필터 (SPY와 같은 방향 종목만)
# - 중장기 모멘텀 (1개월, 3개월, 6개월)
# - 화요일 점수 계산 → 수요일 종가 매수
# - 같은 종목이면 비중만 조절 (거래 최소화)
# - 손절 -7%
# 
# 성과 (5년 백테스트):
# - 총수익률: ~500%
# - MDD: -38%
# - 샤프비율: 0.97
# ============================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# ============================================
# [설정] 전략 파라미터
# ============================================

# ----- 자본금 설정 -----
INITIAL_CAPITAL = 2000       # 초기 자본금 ($2000)

# ----- 거래 비용 -----
BUY_COMMISSION = 0.0025      # 매수 수수료 (0.25%)
SELL_COMMISSION = 0.0025     # 매도 수수료 (0.25%)
SLIPPAGE = 0.001             # 슬리피지 (0.1%) - 매수시 더 비싸게, 매도시 더 싸게

# ----- 리스크 관리 -----
STOP_LOSS = -0.07            # 손절 기준 (-7%)

# ----- 중장기 모멘텀 가중치 -----
# 점수 = (1개월 수익률 × 3.5) + (3개월 수익률 × 2.5) + (6개월 수익률 × 1.5)
WEIGHT_1M = 3.5              # 1개월 수익률 가중치 (최근 → 높은 가중치)
WEIGHT_3M = 2.5              # 3개월 수익률 가중치
WEIGHT_6M = 1.5              # 6개월 수익률 가중치 (오래된 → 낮은 가중치)

# ----- 포트폴리오 구성 -----
TOP_N = 3                    # 상위 N개 종목 선정
ALLOCATIONS = [0.4, 0.3, 0.3]  # 투자 비중 (1위 40%, 2위 30%, 3위 30%)

# ----- SPY 상관관계 필터 -----
CORRELATION_PERIOD = 60      # 상관관계 계산 기간 (60 거래일 ≈ 3개월)
CORRELATION_THRESHOLD = 0.5  # 최소 상관관계 (0.5 이상만 투자)


# ============================================
# [1] 데이터 전처리 함수
# ============================================

def prepare_price_data(df):
    """
    DataFrame을 피벗 테이블로 변환합니다.
    
    변환 전: date, symbol, close, ... (long format)
    변환 후: 날짜(index) × 종목(columns) 형태의 종가 테이블
    
    Args:
        df: 원본 데이터프레임 (date, symbol, close 컬럼 필요)
    
    Returns:
        DataFrame: 피벗된 종가 테이블
    
    예시:
        변환 전:
        | date       | symbol | close |
        | 2024-01-01 | AAPL   | 185.0 |
        | 2024-01-01 | MSFT   | 375.0 |
        
        변환 후:
        |            | AAPL  | MSFT  |
        | 2024-01-01 | 185.0 | 375.0 |
    """
    price_df = df.pivot(index='date', columns='symbol', values='close')
    return price_df


def filter_tuesday(price_df):
    """
    화요일 데이터만 필터링합니다.
    
    이유:
    - 화요일에 점수 계산 → 수요일에 매수
    - 월요일 종가 확인 후 충분한 분석 시간 확보
    - 주 1회 리밸런싱으로 거래 비용 절감
    
    Args:
        price_df: 피벗된 종가 테이블
    
    Returns:
        DataFrame: 화요일만 포함된 종가 테이블
    """
    price_df = price_df.copy()
    
    # 요일 확인 (day_name()은 'Monday', 'Tuesday', ... 반환)
    mask = price_df.index.day_name() == 'Tuesday'
    
    return price_df[mask]


# ============================================
# [2] SPY 상관관계 계산
# ============================================

def calc_spy_correlation(price_df, period=CORRELATION_PERIOD):
    """
    각 종목과 SPY의 상관관계를 계산합니다.
    
    목적:
    - SPY와 같은 방향으로 움직이는 종목만 선택
    - 시장과 반대로 움직이는 종목 제외
    - 2020~2021 구간에서 방향성 일치 확보
    
    계산 방법:
    1. 일별 수익률 계산
    2. 60일 롤링 상관관계 계산 (피어슨 상관계수)
    
    Args:
        price_df: 피벗된 종가 테이블
        period: 상관관계 계산 기간 (기본 60일)
    
    Returns:
        DataFrame: 날짜별 종목별 상관관계 (-1 ~ +1)
        
    해석:
        +1.0: 완벽한 양의 상관관계 (SPY 오르면 같이 오름)
        +0.5: 중간 정도 양의 상관관계
         0.0: 상관관계 없음
        -0.5: 중간 정도 음의 상관관계
        -1.0: 완벽한 음의 상관관계 (SPY 오르면 내림)
    """
    # SPY가 없으면 빈 DataFrame 반환
    if 'SPY' not in price_df.columns:
        return pd.DataFrame()
    
    # 일별 수익률 계산 (오늘 종가 / 어제 종가 - 1)
    returns = price_df.pct_change()
    spy_returns = returns['SPY']
    
    # 각 종목별 SPY와의 롤링 상관관계 계산
    correlation_df = pd.DataFrame(index=price_df.index)
    
    for col in returns.columns:
        if col == 'SPY':
            continue
        
        # 60일 롤링 상관관계 (피어슨 상관계수)
        correlation_df[col] = returns[col].rolling(period).corr(spy_returns)
    
    return correlation_df


def get_high_correlation_stocks(date, correlation_df, threshold=CORRELATION_THRESHOLD):
    """
    특정 날짜에 SPY와 상관관계가 높은 종목 리스트를 반환합니다.
    
    Args:
        date: 조회할 날짜 (Timestamp)
        correlation_df: 상관관계 데이터프레임
        threshold: 최소 상관관계 기준 (기본 0.5)
    
    Returns:
        list: 상관관계가 threshold 이상인 종목 리스트
    
    예시:
        threshold = 0.5일 때:
        - AAPL 상관관계 0.7 → 포함 ✅
        - TSLA 상관관계 0.3 → 제외 ❌
        - XOM 상관관계 0.6 → 포함 ✅
    """
    # 해당 날짜 데이터가 없으면 빈 리스트 반환
    if date not in correlation_df.index:
        return []
    
    # 해당 날짜의 상관관계 값 가져오기
    corr_values = correlation_df.loc[date].dropna()
    
    # threshold 이상인 종목만 필터링
    high_corr = corr_values[corr_values > threshold]
    
    return high_corr.index.tolist()


# ============================================
# [3] 중장기 모멘텀 점수 계산
# ============================================

def calc_momentum_scores(weekly_df):
    """
    중장기 모멘텀 점수를 계산합니다.
    
    공식:
    점수 = (1개월 수익률 × 3.5) + (3개월 수익률 × 2.5) + (6개월 수익률 × 1.5)
    
    왜 중장기 모멘텀인가?
    - 단기 (1주, 2주, 3주): 급등 후 급락하는 종목 선택 → 손실
    - 중장기 (1개월, 3개월, 6개월): 꾸준히 오르는 종목 선택 → 수익
    
    계산 방법 (주 1회 데이터 기준):
    - 4회 전 = 약 1개월 (4주)
    - 12회 전 = 약 3개월 (12주)
    - 24회 전 = 약 6개월 (24주)
    
    Args:
        weekly_df: 화요일만 필터링된 종가 테이블
    
    Returns:
        tuple: (점수 DataFrame, 1개월 수익률 DataFrame)
        
    예시:
        AAPL의 점수 계산:
        - 1개월 수익률: +5%
        - 3개월 수익률: +15%
        - 6개월 수익률: +25%
        - 점수 = (0.05 × 3.5) + (0.15 × 2.5) + (0.25 × 1.5)
               = 0.175 + 0.375 + 0.375 = 0.925
    """
    # 수익률 계산 (pct_change(n) = n회 전 대비 수익률)
    ret_1m = weekly_df.pct_change(4)    # 4회 전 = 약 1개월
    ret_3m = weekly_df.pct_change(12)   # 12회 전 = 약 3개월
    ret_6m = weekly_df.pct_change(24)   # 24회 전 = 약 6개월
    
    # 가중 점수 계산
    score_df = (ret_1m * WEIGHT_1M) + (ret_3m * WEIGHT_3M) + (ret_6m * WEIGHT_6M)
    
    return score_df, ret_1m


# ============================================
# [4] 매수일 매핑 생성
# ============================================

def create_trade_mapping(df):
    """
    화요일 → 수요일 매수일 매핑을 생성합니다.
    
    매매 타이밍:
    - 화요일: 월요일 종가 확인 후 점수 계산, 종목 선정
    - 수요일: 종가 매수 (한국 시간 목요일 새벽)
    
    Args:
        df: 원본 데이터프레임
    
    Returns:
        dict: {화요일 날짜: 수요일 날짜} 매핑
        
    예시:
        {
            2024-01-02 (화): 2024-01-03 (수),
            2024-01-09 (화): 2024-01-10 (수),
            ...
        }
    """
    # 모든 날짜와 요일 매핑
    dates = sorted(df['date'].unique())
    date_weekday = {d: pd.Timestamp(d).day_name() for d in dates}
    
    trade_map = {}
    
    for i, date in enumerate(dates):
        # 화요일인 경우
        if date_weekday[date] == 'Tuesday':
            # 다음 수요일 찾기
            for j in range(i+1, len(dates)):
                if date_weekday[dates[j]] == 'Wednesday':
                    trade_map[date] = dates[j]
                    break
    
    return trade_map


# ============================================
# [5] 백테스트 메인 함수
# ============================================

def run_backtest(df):
    """
    백테스트를 실행합니다.
    
    전략 로직:
    1. 매일: 포트폴리오 가치 계산, 손절 체크
    2. 화요일: 모멘텀 점수 계산, 상관관계 필터, 종목 선정
    3. 수요일: 매수 주문 실행 (같은 종목이면 비중만 조절)
    
    Args:
        df: 원본 데이터프레임 (date, symbol, close, sector 컬럼)
    
    Returns:
        dict: {
            'portfolio': 일별 포트폴리오 가치,
            'trades': 거래 내역,
            'metrics': 성과 지표
        }
    """
    
    # ===== 초기 설정 출력 =====
    print("=" * 60)
    print("[백테스트 실행]")
    print("=" * 60)
    print(f"전략: 상관관계 필터 + 중장기 모멘텀")
    print(f"점수: (1개월×{WEIGHT_1M}) + (3개월×{WEIGHT_3M}) + (6개월×{WEIGHT_6M})")
    print(f"상관관계: SPY와 {CORRELATION_THRESHOLD} 이상인 종목만")
    print(f"초기 자본금: ${INITIAL_CAPITAL:,}")
    print(f"수수료: 매수 {BUY_COMMISSION*100:.2f}% + 매도 {SELL_COMMISSION*100:.2f}%")
    print(f"슬리피지: {SLIPPAGE*100:.2f}%")
    print(f"손절: {STOP_LOSS*100:.1f}%")
    print("=" * 60)
    
    # ===== 데이터 준비 =====
    
    # 일별 데이터 (손절 체크용)
    df_daily = df.copy().sort_values('date').reset_index(drop=True)
    daily_dates = sorted(df_daily['date'].unique())
    
    # 백테스트 기간 출력
    print(f"데이터 기간: {daily_dates[0].strftime('%Y-%m-%d')} ~ {daily_dates[-1].strftime('%Y-%m-%d')}")
    print(f"총 {len(daily_dates)}일")
    
    # 피벗 테이블 생성
    price_df = prepare_price_data(df)
    
    # 화요일만 필터링 (점수 계산용)
    tuesday_df = filter_tuesday(price_df)
    if 'SPY' in tuesday_df.columns:
        tuesday_df = tuesday_df.dropna(subset=['SPY'])
    print(f"화요일 데이터: {len(tuesday_df)}개")
    
    # 모멘텀 점수 계산
    score_df, ret_1m = calc_momentum_scores(tuesday_df)
    
    # SPY 상관관계 계산
    correlation_df = calc_spy_correlation(price_df)
    
    # 화요일 → 수요일 매핑
    trade_map = create_trade_mapping(df)
    print(f"매핑된 거래일: {len(trade_map)}개")
    
    # 점수가 있는 날짜 리스트
    score_dates = score_df.dropna(how='all').index.tolist()
    
    # ===== 시뮬레이션 변수 초기화 =====
    
    portfolio_values = []  # 일별 포트폴리오 가치 저장
    trades = []            # 거래 내역 저장
    
    cash = INITIAL_CAPITAL  # 현재 현금
    holdings = {}           # 현재 보유 종목 {symbol: {'shares': int, 'avg_price': float}}
    pending_order = None    # 대기 중인 매수 주문
    
    print(f"\n{len(daily_dates)}일 시뮬레이션 시작...")
    
    # ===== 매일 시뮬레이션 루프 =====
    
    for i, date in enumerate(daily_dates):
        
        # 진행 상황 출력 (100일마다)
        if (i + 1) % 100 == 0:
            print(f"  진행중... {i+1}/{len(daily_dates)} ({(i+1)/len(daily_dates)*100:.1f}%)")
        
        # 오늘 데이터 가져오기
        today_data = df_daily[df_daily['date'] == date]
        date_ts = pd.Timestamp(date)
        
        # ----- [5-1] 포트폴리오 가치 계산 -----
        # 현금 + 보유 주식 평가액
        
        portfolio_value = cash
        
        for symbol, info in holdings.items():
            stock = today_data[today_data['symbol'] == symbol]
            if not stock.empty:
                current_price = stock.iloc[0]['close']
                portfolio_value += info['shares'] * current_price
        
        # 일별 기록 저장
        portfolio_values.append({
            'date': date,
            'value': portfolio_value,
            'cash': cash
        })
        
        # ----- [5-2] 손절 체크 (매일) -----
        # 보유 종목 중 -7% 이하면 즉시 매도
        
        for symbol, info in list(holdings.items()):
            stock = today_data[today_data['symbol'] == symbol]
            if stock.empty:
                continue
            
            current_price = stock.iloc[0]['close']
            
            # 수익률 계산 (현재가 / 매수가 - 1)
            return_rate = (current_price - info['avg_price']) / info['avg_price']
            
            # 손절 조건 충족
            if return_rate <= STOP_LOSS:
                # 슬리피지 적용 (매도 시 더 낮은 가격에 체결)
                sell_price = current_price * (1 - SLIPPAGE)
                sell_amount = info['shares'] * sell_price
                commission = sell_amount * SELL_COMMISSION
                
                # 현금 증가 (매도 금액 - 수수료)
                cash += sell_amount - commission
                
                # 거래 기록
                trades.append({
                    'date': date,
                    'symbol': symbol,
                    'action': 'STOP_LOSS',
                    'shares': info['shares'],
                    'price': sell_price,
                    'amount': sell_amount,
                    'commission': commission,
                    'slippage': current_price * SLIPPAGE * info['shares'],
                    'return_rate': return_rate
                })
                
                # 보유 목록에서 제거
                del holdings[symbol]
        
        # ----- [5-3] 대기 중인 매수 주문 실행 (수요일) -----
        
        if pending_order is not None and pending_order['trade_date'] == date:
            order = pending_order
            pending_order = None
            
            new_picks = order['picks']      # 새로 선정된 종목
            new_scores = order['scores']    # 새 종목들의 점수
            
            # 현재 보유 vs 새 종목 비교
            current_holdings = set(holdings.keys())
            new_holdings_set = set(new_picks)
            
            to_sell = current_holdings - new_holdings_set  # 매도할 종목 (새 리스트에 없음)
            to_buy = new_holdings_set - current_holdings   # 신규 매수 (기존에 없음)
            to_keep = current_holdings & new_holdings_set  # 유지할 종목 (둘 다 있음)
            
            # --- [5-3-1] 매도할 종목 처리 ---
            
            for symbol in to_sell:
                if symbol not in holdings:
                    continue
                
                info = holdings[symbol]
                stock = today_data[today_data['symbol'] == symbol]
                
                if not stock.empty:
                    base_price = stock.iloc[0]['close']
                    
                    # 슬리피지 적용 (매도 시 더 낮은 가격)
                    sell_price = base_price * (1 - SLIPPAGE)
                    sell_amount = info['shares'] * sell_price
                    commission = sell_amount * SELL_COMMISSION
                    
                    # 현금 증가
                    cash += sell_amount - commission
                    
                    # 수익률 계산
                    return_rate = (sell_price - info['avg_price']) / info['avg_price']
                    
                    # 거래 기록
                    trades.append({
                        'date': date,
                        'symbol': symbol,
                        'action': 'SELL',
                        'shares': info['shares'],
                        'price': sell_price,
                        'amount': sell_amount,
                        'commission': commission,
                        'slippage': base_price * SLIPPAGE * info['shares'],
                        'return_rate': return_rate
                    })
                    
                    # 보유 목록에서 제거
                    del holdings[symbol]
            
            # --- [5-3-2] 비중 계산 ---
            
            n_picks = len(new_picks)
            
            if n_picks >= 3:
                allocations = ALLOCATIONS[:3]  # [0.4, 0.3, 0.3]
            elif n_picks == 2:
                allocations = [0.5, 0.5]
            elif n_picks == 1:
                allocations = [1.0]
            else:
                allocations = []
            
            # 종목별 목표 비중
            target_allocations = {}
            for j, symbol in enumerate(new_picks):
                if j < len(allocations):
                    target_allocations[symbol] = allocations[j]
            
            # --- [5-3-3] 유지 종목 비중 조절 ---
            # 같은 종목이면 비중만 조절 (거래 최소화)
            
            for symbol in to_keep:
                if symbol not in holdings or symbol not in target_allocations:
                    continue
                
                stock = today_data[today_data['symbol'] == symbol]
                if stock.empty:
                    continue
                
                current_price = stock.iloc[0]['close']
                
                # 현재 가치 vs 목표 가치
                current_value = holdings[symbol]['shares'] * current_price
                target_value = portfolio_value * target_allocations[symbol]
                
                diff_value = target_value - current_value
                diff_shares = int(abs(diff_value) / current_price)
                
                # 점수 찾기
                score_idx = new_picks.index(symbol) if symbol in new_picks else -1
                score = new_scores[score_idx] if 0 <= score_idx < len(new_scores) else 0
                
                # 비중 차이가 5% 이상일 때만 조절 (너무 작은 조절은 비용만 발생)
                if abs(diff_value) / portfolio_value > 0.05 and diff_shares > 0:
                    
                    if diff_value > 0:
                        # --- 추가 매수 ---
                        buy_price = current_price * (1 + SLIPPAGE)
                        buy_amount = diff_shares * buy_price
                        commission = buy_amount * BUY_COMMISSION
                        
                        if cash >= buy_amount + commission:
                            cash -= (buy_amount + commission)
                            
                            # 평균 단가 재계산
                            old_shares = holdings[symbol]['shares']
                            old_avg = holdings[symbol]['avg_price']
                            new_shares = old_shares + diff_shares
                            new_avg = (old_avg * old_shares + buy_amount) / new_shares
                            
                            holdings[symbol]['shares'] = new_shares
                            holdings[symbol]['avg_price'] = new_avg
                            
                            trades.append({
                                'date': date,
                                'symbol': symbol,
                                'action': 'ADD',
                                'shares': diff_shares,
                                'price': buy_price,
                                'amount': buy_amount,
                                'commission': commission,
                                'slippage': current_price * SLIPPAGE * diff_shares,
                                'return_rate': 0,
                                'score': score
                            })
                    
                    else:
                        # --- 일부 매도 ---
                        sell_price = current_price * (1 - SLIPPAGE)
                        sell_amount = diff_shares * sell_price
                        commission = sell_amount * SELL_COMMISSION
                        
                        cash += sell_amount - commission
                        holdings[symbol]['shares'] -= diff_shares
                        
                        trades.append({
                            'date': date,
                            'symbol': symbol,
                            'action': 'REDUCE',
                            'shares': diff_shares,
                            'price': sell_price,
                            'amount': sell_amount,
                            'commission': commission,
                            'slippage': current_price * SLIPPAGE * diff_shares,
                            'return_rate': 0,
                            'score': score
                        })
            
            # --- [5-3-4] 신규 매수 ---
            
            for symbol in to_buy:
                if symbol not in target_allocations:
                    continue
                
                stock = today_data[today_data['symbol'] == symbol]
                if stock.empty:
                    continue
                
                base_price = stock.iloc[0]['close']
                
                # 슬리피지 적용 (매수 시 더 높은 가격)
                buy_price = base_price * (1 + SLIPPAGE)
                
                if pd.isna(buy_price):
                    continue
                
                # 투자 금액 계산
                allocation = target_allocations[symbol]
                invest_amount = portfolio_value * allocation
                shares = int(invest_amount / buy_price)
                
                if shares <= 0:
                    continue
                
                buy_amount = shares * buy_price
                commission = buy_amount * BUY_COMMISSION
                
                # 현금 충분한지 확인
                if cash >= buy_amount + commission:
                    cash -= (buy_amount + commission)
                    
                    # 보유 목록에 추가
                    holdings[symbol] = {
                        'shares': shares,
                        'avg_price': buy_price
                    }
                    
                    # 점수 찾기
                    score_idx = new_picks.index(symbol) if symbol in new_picks else -1
                    score = new_scores[score_idx] if 0 <= score_idx < len(new_scores) else 0
                    
                    trades.append({
                        'date': date,
                        'symbol': symbol,
                        'action': 'BUY',
                        'shares': shares,
                        'price': buy_price,
                        'amount': buy_amount,
                        'commission': commission,
                        'slippage': base_price * SLIPPAGE * shares,
                        'return_rate': 0,
                        'score': score
                    })
        
        # ----- [5-4] 화요일: 점수 계산 & 종목 선정 -----
        
        # 점수 계산일이 아니면 스킵
        if date_ts not in score_dates:
            continue
        
        # 매수일 매핑이 없으면 스킵
        if date not in trade_map:
            continue
        
        trade_date = trade_map[date]  # 수요일 날짜
        
        # --- [5-4-1] 시장 필터 ---
        # 1개월 평균 수익률 > 0 일 때만 매수
        
        if date_ts not in ret_1m.index:
            continue
        
        market_momentum = ret_1m.loc[date_ts].mean()
        
        if market_momentum <= 0:
            continue  # 시장이 하락 추세면 매수 안 함
        
        # --- [5-4-2] 모멘텀 점수 가져오기 ---
        
        if date_ts not in score_df.index:
            continue
        
        current_scores = score_df.loc[date_ts].drop(labels=['SPY'], errors='ignore').dropna()
        
        if current_scores.empty:
            continue
        
        # --- [5-4-3] 상관관계 필터 ---
        # SPY와 상관관계 > 0.5인 종목만 투자 대상
        
        high_corr_stocks = get_high_correlation_stocks(date_ts, correlation_df)
        
        if high_corr_stocks:
            # 상관관계 높은 종목만 필터링
            filtered_scores = current_scores[current_scores.index.isin(high_corr_stocks)]
        else:
            filtered_scores = current_scores
        
        if filtered_scores.empty:
            continue
        
        # --- [5-4-4] Top N 종목 선정 ---
        
        top_n = filtered_scores.nlargest(min(TOP_N, len(filtered_scores)))
        
        # --- [5-4-5] 매수 주문 대기 등록 ---
        # 수요일에 실행됨
        
        pending_order = {
            'score_date': date,
            'trade_date': trade_date,
            'picks': top_n.index.tolist(),
            'scores': top_n.values.tolist()
        }
    
    # ===== 결과 정리 =====
    
    portfolio_df = pd.DataFrame(portfolio_values)
    trades_df = pd.DataFrame(trades) if trades else pd.DataFrame()
    metrics = calculate_metrics(portfolio_df, trades_df, df)
    
    print("\n" + "=" * 60)
    print("✅ 백테스트 완료!")
    print("=" * 60)
    
    return {
        'portfolio': portfolio_df,
        'trades': trades_df,
        'metrics': metrics
    }


# ============================================
# [6] 성과 지표 계산
# ============================================

def calculate_metrics(portfolio_df, trades_df, df):
    """
    백테스트 성과 지표를 계산합니다.
    
    계산 지표:
    - 총 수익률: (최종 - 초기) / 초기
    - CAGR: 연환산 복리 수익률
    - 변동성: 일별 수익률 표준편차 × √252
    - 샤프 비율: (CAGR - 무위험수익률) / 변동성
    - MDD: 최대 낙폭
    - 승률: 양수 수익 일수 비율
    
    Args:
        portfolio_df: 일별 포트폴리오 가치
        trades_df: 거래 내역
        df: 원본 데이터
    
    Returns:
        dict: 성과 지표 딕셔너리
    """
    values = portfolio_df['value'].values
    dates = portfolio_df['date']
    
    # ----- 수익률 계산 -----
    initial = values[0]
    final = values[-1]
    total_return = (final - initial) / initial
    
    # 일별 수익률
    daily_returns = pd.Series(values).pct_change().dropna()
    
    # ----- CAGR (연환산 복리 수익률) -----
    days = (dates.iloc[-1] - dates.iloc[0]).days
    years = days / 365
    cagr = (final / initial) ** (1 / years) - 1 if years > 0 else 0
    
    # ----- 변동성 & 샤프 비율 -----
    volatility = daily_returns.std() * np.sqrt(252)  # 연환산
    risk_free_rate = 0.03  # 무위험 수익률 3%
    sharpe = (cagr - risk_free_rate) / volatility if volatility > 0 else 0
    
    # ----- MDD (최대 낙폭) -----
    peak = pd.Series(values).cummax()
    drawdown = (pd.Series(values) - peak) / peak
    mdd = drawdown.min()
    
    # ----- 승률 -----
    win_rate = (daily_returns > 0).mean()
    
    # ----- SPY 수익률 (벤치마크) -----
    spy_return = 0
    if 'SPY' in df['symbol'].unique():
        spy = df[df['symbol'] == 'SPY'].sort_values('date')
        if len(spy) >= 2:
            spy_initial = spy.iloc[0]['close']
            spy_final = spy.iloc[-1]['close']
            spy_return = (spy_final - spy_initial) / spy_initial
    
    # ----- 거래 통계 -----
    total_trades = len(trades_df) if not trades_df.empty else 0
    total_commission = trades_df['commission'].sum() if not trades_df.empty else 0
    total_slippage = trades_df['slippage'].sum() if not trades_df.empty and 'slippage' in trades_df.columns else 0
    stop_loss_count = len(trades_df[trades_df['action'] == 'STOP_LOSS']) if not trades_df.empty else 0
    
    # 거래 유형별 카운트
    buy_count = len(trades_df[trades_df['action'] == 'BUY']) if not trades_df.empty else 0
    sell_count = len(trades_df[trades_df['action'] == 'SELL']) if not trades_df.empty else 0
    add_count = len(trades_df[trades_df['action'] == 'ADD']) if not trades_df.empty else 0
    reduce_count = len(trades_df[trades_df['action'] == 'REDUCE']) if not trades_df.empty else 0
    
    return {
        'initial_capital': initial,
        'final_capital': final,
        'total_return': total_return,
        'cagr': cagr,
        'volatility': volatility,
        'sharpe_ratio': sharpe,
        'mdd': mdd,
        'win_rate': win_rate,
        'spy_return': spy_return,
        'alpha': total_return - spy_return,
        'total_trades': total_trades,
        'buy_count': buy_count,
        'sell_count': sell_count,
        'add_count': add_count,
        'reduce_count': reduce_count,
        'total_commission': total_commission,
        'total_slippage': total_slippage,
        'stop_loss_count': stop_loss_count
    }


# ============================================
# [7] 결과 출력
# ============================================

def print_metrics(metrics, trades_df=None):
    """
    백테스트 성과를 보기 좋게 출력합니다.
    
    Args:
        metrics: calculate_metrics()의 반환값
        trades_df: 거래 내역 (최근 매수 내역 표시용)
    """
    print("\n" + "=" * 60)
    print("📊 백테스트 성과")
    print("=" * 60)
    
    # ----- 수익 -----
    print(f"\n💰 수익")
    print(f"  초기 자본금: ${metrics['initial_capital']:,.2f}")
    print(f"  최종 자본금: ${metrics['final_capital']:,.2f}")
    print(f"  총 수익률: {metrics['total_return']*100:.2f}%")
    print(f"  연환산 수익률 (CAGR): {metrics['cagr']*100:.2f}%")
    
    # ----- 벤치마크 비교 -----
    print(f"\n📈 벤치마크 비교")
    print(f"  SPY 수익률: {metrics['spy_return']*100:.2f}%")
    print(f"  초과 수익 (Alpha): {metrics['alpha']*100:.2f}%")
    
    # ----- 위험 지표 -----
    print(f"\n⚠️ 위험 지표")
    print(f"  변동성: {metrics['volatility']*100:.2f}%")
    print(f"  최대 낙폭 (MDD): {metrics['mdd']*100:.2f}%")
    print(f"  샤프 비율: {metrics['sharpe_ratio']:.2f}")
    
    # ----- 거래 통계 -----
    print(f"\n🎯 거래 통계")
    print(f"  총 거래 횟수: {metrics['total_trades']}회")
    print(f"    - 신규 매수 (BUY): {metrics['buy_count']}회")
    print(f"    - 전량 매도 (SELL): {metrics['sell_count']}회")
    print(f"    - 추가 매수 (ADD): {metrics['add_count']}회")
    print(f"    - 일부 매도 (REDUCE): {metrics['reduce_count']}회")
    print(f"    - 손절 (STOP_LOSS): {metrics['stop_loss_count']}회")
    print(f"  총 수수료: ${metrics['total_commission']:,.2f}")
    print(f"  총 슬리피지: ${metrics['total_slippage']:,.2f}")
    print(f"  총 비용: ${metrics['total_commission'] + metrics['total_slippage']:,.2f}")
    
    # ----- 기타 -----
    print(f"\n📅 기타")
    print(f"  승률 (일 기준): {metrics['win_rate']*100:.2f}%")
    
    # ----- 최근 매수 내역 -----
    if trades_df is not None and not trades_df.empty:
        buy_trades = trades_df[trades_df['action'].isin(['BUY', 'ADD'])].copy()
        
        if not buy_trades.empty:
            recent_dates = buy_trades['date'].drop_duplicates().sort_values(ascending=False).head(10)
            
            print(f"\n🛒 최근 매수 내역 (최근 10회)")
            print("-" * 60)
            
            for buy_date in recent_dates:
                date_buys = buy_trades[buy_trades['date'] == buy_date]
                
                # 점수 기준 정렬 (점수 없는 경우 금액 기준)
                if 'score' in date_buys.columns:
                    date_buys = date_buys.sort_values('score', ascending=False)
                
                print(f"\n📅 {buy_date.strftime('%Y-%m-%d')}")
                
                for i, (_, row) in enumerate(date_buys.iterrows()):
                    score = row.get('score', 0)
                    action = row['action']
                    print(f"  {action:5} {row['symbol']:5} | 점수: {score:.4f} | 가격: ${row['price']:.2f} | 금액: ${row['amount']:,.2f}")
    
    print("\n" + "=" * 60)


# ============================================
# [8] 그래프 출력
# ============================================

def plot_results(portfolio_df, trades_df, df, figsize=(14, 12)):
    """
    백테스트 결과를 4개의 그래프로 시각화합니다.
    
    그래프 구성:
    1. 포트폴리오 vs SPY (좌상단)
    2. 일별 수익률 (우상단)
    3. 누적 수익률 (좌하단)
    4. Drawdown (우하단)
    
    Args:
        portfolio_df: 일별 포트폴리오 가치
        trades_df: 거래 내역
        df: 원본 데이터
        figsize: 그래프 크기
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    # ===== [8-1] 포트폴리오 vs SPY =====
    ax1 = axes[0, 0]
    
    # 정규화 (시작 = 100)
    portfolio_df = portfolio_df.copy()
    portfolio_df['normalized'] = portfolio_df['value'] / portfolio_df['value'].iloc[0] * 100
    
    # 포트폴리오 라인
    ax1.plot(portfolio_df['date'], portfolio_df['normalized'], 
             label='Portfolio', linewidth=2, color='blue')
    
    # SPY 라인
    if 'SPY' in df['symbol'].unique():
        spy = df[df['symbol'] == 'SPY'].sort_values('date').copy()
        spy['normalized'] = spy['close'] / spy['close'].iloc[0] * 100
        ax1.plot(spy['date'], spy['normalized'], 
                 label='SPY', linewidth=2, linestyle='--', color='orange')
    
    # 매수 시점 빨간 점
    if not trades_df.empty:
        buy_trades = trades_df[trades_df['action'] == 'BUY']
        for _, trade in buy_trades.iterrows():
            trade_date = trade['date']
            port_value = portfolio_df[portfolio_df['date'] == trade_date]['normalized']
            if not port_value.empty:
                ax1.scatter(trade_date, port_value.values[0], 
                           color='red', s=30, zorder=5, label='_nolegend_')
    
    ax1.set_title('Portfolio vs SPY', fontsize=12)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # ===== [8-2] 일별 수익률 =====
    ax2 = axes[0, 1]
    daily_returns = portfolio_df['value'].pct_change().dropna()
    colors = ['green' if r > 0 else 'red' for r in daily_returns]
    ax2.bar(range(len(daily_returns)), daily_returns, color=colors, alpha=0.7)
    ax2.axhline(y=0, color='black', linewidth=0.5)
    ax2.set_title('Daily Returns', fontsize=12)
    ax2.grid(True, alpha=0.3)
    
    # ===== [8-3] 누적 수익률 =====
    ax3 = axes[1, 0]
    cumulative = (1 + daily_returns).cumprod() - 1
    ax3.fill_between(range(len(cumulative)), cumulative, alpha=0.3, color='blue')
    ax3.plot(range(len(cumulative)), cumulative, linewidth=2, color='blue')
    ax3.axhline(y=0, color='black', linewidth=0.5)
    ax3.set_title('Cumulative Returns', fontsize=12)
    ax3.grid(True, alpha=0.3)
    
    # ===== [8-4] Drawdown =====
    ax4 = axes[1, 1]
    peak = portfolio_df['value'].cummax()
    drawdown = (portfolio_df['value'] - peak) / peak
    ax4.fill_between(portfolio_df['date'], drawdown, 0, color='red', alpha=0.3)
    ax4.plot(portfolio_df['date'], drawdown, color='red', linewidth=1)
    ax4.set_title('Drawdown', fontsize=12)
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


# ============================================
# [9] 테스트 실행
# ============================================

if __name__ == "__main__":
    # 이 파일을 직접 실행하면 테스트 수행
    print("백테스트 모듈 테스트")
    print("Colab에서 다음 코드로 실행하세요:")
    print()
    print("from src.data import get_backtest_data")
    print("from src.backtest import run_backtest, print_metrics, plot_results")
    print()
    print("df = get_backtest_data()")
    print("result = run_backtest(df)")
    print("print_metrics(result['metrics'], result['trades'])")
    print("plot_results(result['portfolio'], result['trades'], df)")