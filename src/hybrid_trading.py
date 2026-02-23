# ============================================
# 파일명: src/hybrid_trading.py
# 설명: 하이브리드 전략 페이퍼 트레이딩
# 
# 전략: 모멘텀 35% + AI 65%
# 백테스트 성과:
# - 수익률: +352.73%
# - 승률: 61.2%
# - 샤프비율: 2.51
# ============================================

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# 상위 폴더의 config.py 임포트
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    INITIAL_CAPITAL,
    STOP_LOSS,
    TOP_N,
    BUY_COMMISSION,
    SELL_COMMISSION,
    SLIPPAGE
)

# 데이터 및 전략
from data import get_sp500_list, download_stock_data, get_backtest_data
from strategy import CustomStrategy, prepare_price_data, filter_tuesday
from ai_data import create_features, get_feature_columns
from ai_strategy import AIStrategy, XGB_PARAMS

# Google Sheets (선택적)
try:
    from sheets import SheetsManager
    SHEETS_AVAILABLE = True
except ImportError:
    SHEETS_AVAILABLE = False
    print("⚠️ Sheets 모듈 없음 (선택적)")

# Telegram
from telegram import send_message


# ============================================
# [1] 설정
# ============================================

# Hybrid 전용 Google Sheets 이름
HYBRID_SPREADSHEET = "Hybrid_Paper_Trading"

# 시트 이름
HYBRID_HOLDINGS_SHEET = "Holdings"
HYBRID_TRADES_SHEET = "Trades"
HYBRID_SIGNALS_SHEET = "Signals"

# 가중치
WEIGHT_MOMENTUM = 0.35
WEIGHT_AI = 0.65

# AI 학습 기간 (자동 롤링)
# - 학습: 5년 전 ~ 1년 전
_today = datetime.now()
TRAIN_START = (_today - timedelta(days=365*5)).strftime('%Y-%m-%d')  # 5년 전
TRAIN_END = (_today - timedelta(days=365)).strftime('%Y-%m-%d')      # 1년 전


# ============================================
# [1-1] Hybrid Sheets Manager
# ============================================

class HybridSheetsManager:
    """
    Hybrid 전용 Google Sheets 관리
    기존 SheetsManager를 Hybrid 전용 스프레드시트로 사용
    """
    
    def __init__(self):
        self.sheets = None
        self._connect()
    
    def _connect(self):
        """Sheets 연결"""
        if not SHEETS_AVAILABLE:
            print("⚠️ Sheets 모듈 없음")
            return
        
        try:
            self.sheets = SheetsManager(spreadsheet_name=HYBRID_SPREADSHEET)
            print(f"✅ Hybrid Sheets 연결: {HYBRID_SPREADSHEET}")
        except Exception as e:
            print(f"⚠️ Sheets 연결 실패: {e}")
            self.sheets = None
    
    # ============================================
    # 현금 추적 시스템
    # ============================================
    
    def get_cash(self):
        """
        현재 현금 잔고 가져오기
        Cash 시트의 마지막 행에서 조회
        
        Returns:
            float: 현금 잔고
        """
        if not self.sheets:
            return INITIAL_CAPITAL
        
        try:
            # Cash 시트 가져오기/생성
            try:
                ws = self.sheets.spreadsheet.worksheet("Cash")
            except:
                # 시트 없으면 생성하고 초기 자본금 입력
                ws = self.sheets.spreadsheet.add_worksheet(title="Cash", rows=5000, cols=5)
                ws.update("A1", [["Date", "Cash", "Change", "Reason", "Balance_Check"]])
                ws.append_row([
                    datetime.now().strftime('%Y-%m-%d'),
                    INITIAL_CAPITAL,
                    0,
                    "초기 자본금",
                    INITIAL_CAPITAL
                ])
                return INITIAL_CAPITAL
            
            # 마지막 행 가져오기
            data = ws.get_all_values()
            
            if len(data) <= 1:
                # 헤더만 있으면 초기 자본금 입력
                ws.append_row([
                    datetime.now().strftime('%Y-%m-%d'),
                    INITIAL_CAPITAL,
                    0,
                    "초기 자본금",
                    INITIAL_CAPITAL
                ])
                return INITIAL_CAPITAL
            
            last_row = data[-1]
            cash = float(last_row[1]) if last_row[1] else INITIAL_CAPITAL
            print(f"💰 현재 현금: ${cash:,.2f}")
            return cash
            
        except Exception as e:
            print(f"⚠️ Cash 로드 실패: {e}")
            return INITIAL_CAPITAL
    
    def update_cash(self, amount, reason=""):
        """
        현금 변동 기록
        
        Args:
            amount: 변동 금액 (양수: 입금, 음수: 출금)
            reason: 변동 사유
        """
        if not self.sheets:
            return
        
        try:
            # 현재 현금 가져오기
            current_cash = self.get_cash()
            new_cash = current_cash + amount
            
            # Cash 시트에 기록
            try:
                ws = self.sheets.spreadsheet.worksheet("Cash")
            except:
                ws = self.sheets.spreadsheet.add_worksheet(title="Cash", rows=5000, cols=5)
                ws.update("A1", [["Date", "Cash", "Change", "Reason", "Balance_Check"]])
            
            row = [
                datetime.now().strftime('%Y-%m-%d %H:%M'),
                round(new_cash, 2),
                round(amount, 2),
                reason,
                round(new_cash, 2)
            ]
            ws.append_row(row)
            print(f"💰 현금 변동: ${amount:+,.2f} → ${new_cash:,.2f} ({reason})")
            
        except Exception as e:
            print(f"⚠️ Cash 업데이트 실패: {e}")
    
    def get_holdings(self):
        """
        현재 보유 종목 가져오기
        
        Returns:
            dict: {symbol: {shares, avg_price, sector, buy_date}}
        """
        if not self.sheets:
            return {}
        
        try:
            df = self.sheets.load_holdings()
            
            if df.empty:
                return {}
            
            holdings = {}
            for _, row in df.iterrows():
                symbol = row['Symbol']
                if symbol:
                    holdings[symbol] = {
                        'shares': int(float(row.get('Shares', 0) or 0)),
                        'avg_price': float(row.get('Avg_Price', 0) or 0),
                        'sector': row.get('Sector', ''),
                        'buy_date': row.get('Buy_Date', '')
                    }
            
            print(f"📊 보유 종목: {len(holdings)}개")
            return holdings
            
        except Exception as e:
            print(f"⚠️ Holdings 로드 실패: {e}")
            return {}
    
    def update_holdings(self, actions, current_prices):
        """
        리밸런싱 후 Holdings 업데이트
        
        Args:
            actions: 리밸런싱 액션 리스트
            current_prices: 현재 가격 dict
        """
        if not self.sheets:
            return
        
        try:
            for action in actions:
                symbol = action['symbol']
                act_type = action['action']
                shares = action['shares']
                price = action['price']
                
                if act_type == 'BUY':
                    # 신규 매수
                    self.sheets.save_holding({
                        'symbol': symbol,
                        'shares': shares,
                        'avg_price': price,
                        'sector': '',
                        'buy_date': datetime.now().strftime('%Y-%m-%d')
                    })
                
                elif act_type == 'SELL':
                    # 전량 매도
                    self.sheets.remove_holding(symbol)
                
                elif act_type == 'ADD':
                    # 추가 매수 - 평균 단가 재계산
                    holdings = self.get_holdings()
                    if symbol in holdings:
                        old_shares = holdings[symbol]['shares']
                        old_price = holdings[symbol]['avg_price']
                        new_shares = old_shares + shares
                        new_avg = (old_shares * old_price + shares * price) / new_shares
                        self.sheets.update_holding(symbol, shares=new_shares, avg_price=new_avg)
                
                elif act_type == 'REDUCE':
                    # 일부 매도
                    holdings = self.get_holdings()
                    if symbol in holdings:
                        new_shares = holdings[symbol]['shares'] - shares
                        if new_shares <= 0:
                            self.sheets.remove_holding(symbol)
                        else:
                            self.sheets.update_holding(symbol, shares=new_shares)
            
            print("✅ Holdings 업데이트 완료")
            
        except Exception as e:
            print(f"⚠️ Holdings 업데이트 실패: {e}")
    
    def save_trade(self, action, memo="Hybrid"):
        """
        거래 기록 저장
        
        Args:
            action: 거래 액션 dict
            memo: 메모
        """
        if not self.sheets:
            return
        
        try:
            # Trades 시트에 직접 추가
            ws = self.sheets.spreadsheet.worksheet("Trades")
            row = [
                datetime.now().strftime('%Y-%m-%d'),
                action['symbol'],
                action['action'],
                action['shares'],
                round(action['price'], 2),
                round(action['amount'], 2),
                round(action['amount'] * BUY_COMMISSION, 2),
                round(action.get('return_pct', 0), 2),
                0,  # realized_pnl
                '',  # sector
                memo
            ]
            ws.append_row(row)
            print(f"✅ Trade 저장: {action['action']} {action['symbol']}")
        except Exception as e:
            print(f"⚠️ Trade 저장 실패: {e}")
    
    def save_signal(self, signal):
        """
        신호 기록 저장
        
        Args:
            signal: 신호 dict
        """
        if not self.sheets:
            return
        
        try:
            # 시장 필터링 발동 체크
            if signal.get('market_filter', False):
                self.sheets.save_signal({
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M'),
                    'analysis_date': datetime.now().strftime('%Y-%m-%d'),
                    'signal': 'MARKET_FILTER',
                    'picks': '없음 (시장 필터링)',
                    'scores': '',
                    'allocations': '',
                    'market_momentum': '',
                    'spy_price': signal.get('spy_price', 0),
                    'market_trend': 'BEARISH'
                })
                print("✅ Signal 저장 완료 (시장 필터링)")
                return
            
            # 빈 signal 체크
            if not signal.get('picks'):
                print("⚠️ Signal 저장 스킵: 선정 종목 없음")
                return
            
            # scores를 문자열로 변환
            scores_str = ', '.join([str(round(s, 4)) for s in signal['scores']])
            allocs_str = ', '.join([str(int(a*100)) + '%' for a in signal['allocations']])
            
            self.sheets.save_signal({
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M'),
                'analysis_date': datetime.now().strftime('%Y-%m-%d'),
                'signal': 'HYBRID',
                'picks': ', '.join(signal['picks']),
                'scores': scores_str,
                'allocations': allocs_str,
                'market_momentum': '',
                'spy_price': signal.get('spy_price', 0),
                'market_trend': 'BULLISH'
            })
            print("✅ Signal 저장 완료")
        except Exception as e:
            print(f"⚠️ Signal 저장 실패: {e}")
    
    def save_daily_value(self, holdings, current_prices, cash, spy_price=0):
        """
        Daily_Value 시트에 일일 포트폴리오 가치 기록
        
        Args:
            holdings: 보유 종목 dict
            current_prices: 현재 가격 dict
            cash: 현금
            spy_price: SPY 가격
        """
        if not self.sheets:
            return
        
        try:
            today = datetime.now().strftime('%Y-%m-%d')
            
            # 주식 가치 계산
            stocks_value = 0
            if holdings:
                for symbol, info in holdings.items():
                    shares = info.get('shares', 0)
                    price = current_prices.get(symbol, info.get('avg_price', 0))
                    stocks_value += shares * price
            
            # 총 가치
            total_value = stocks_value + cash
            
            # Daily_Value 시트 가져오기/생성
            try:
                ws = self.sheets.spreadsheet.worksheet("Daily_Value")
            except:
                ws = self.sheets.spreadsheet.add_worksheet(title="Daily_Value", rows=5000, cols=10)
                # 헤더 추가
                ws.update("A1", [["Date", "Total_Value", "Cash", "Stocks_Value", "Daily_Return%", "SPY_Price", "SPY_Return%", "Alpha"]])
                print("✅ Daily_Value 시트 자동 생성")
            
            # 데이터 가져오기
            data = ws.get_all_values()
            
            # 중복 체크 (오늘 이미 기록되어 있으면 업데이트)
            today_row_idx = None
            if len(data) > 1:
                for i, row in enumerate(data[1:], start=2):  # 1-indexed, 헤더 제외
                    if row[0] == today:
                        today_row_idx = i
                        break
            
            # 이전 데이터에서 수익률 계산 (오늘 제외)
            prev_value = None
            prev_spy = None
            
            if len(data) > 1:
                for row in reversed(data[1:]):
                    if row[0] != today:
                        try:
                            prev_value = float(row[1]) if row[1] else None
                            prev_spy = float(row[5]) if row[5] else None
                        except:
                            pass
                        break
            
            # 수익률 계산
            daily_return = 0
            spy_return = 0
            alpha = 0
            
            if prev_value and prev_value > 0:
                daily_return = (total_value - prev_value) / prev_value * 100
            
            if prev_spy and prev_spy > 0 and spy_price > 0:
                spy_return = (spy_price - prev_spy) / prev_spy * 100
                alpha = daily_return - spy_return
            
            # 행 데이터
            row = [
                today,
                round(total_value, 2),
                round(cash, 2),
                round(stocks_value, 2),
                round(daily_return, 2),
                round(spy_price, 2),
                round(spy_return, 2),
                round(alpha, 2)
            ]
            
            if today_row_idx:
                # 오늘 데이터 업데이트
                ws.update(f"A{today_row_idx}:H{today_row_idx}", [row])
                print(f"✅ Daily_Value 업데이트: ${total_value:,.2f}")
            else:
                # 새 행 추가
                ws.append_row(row)
                print(f"✅ Daily_Value 저장: ${total_value:,.2f}")
            
        except Exception as e:
            print(f"⚠️ Daily_Value 저장 실패: {e}")


# ============================================
# [2] Hybrid 전략 클래스 (간소화 버전)
# ============================================

# 시장 필터링 설정
MARKET_FILTER_MA_PERIOD = 20  # 20일 이동평균

class HybridTradingStrategy:
    """
    하이브리드 트레이딩 전략
    모멘텀 점수 + AI 확률 결합
    + 시장 필터링 (SPY > 20일 MA)
    """
    
    def __init__(self, weight_momentum=WEIGHT_MOMENTUM, weight_ai=WEIGHT_AI,
                 use_market_filter=True):
        self.weight_m = weight_momentum
        self.weight_ai = weight_ai
        self.use_market_filter = use_market_filter
        
        self.ai_strategy = None
        self.momentum_strategy = None
        self.score_df = None
        self.feature_cols = None
        self.spy_df = None  # SPY 데이터
        
        self.is_prepared = False
    
    def prepare(self, train_df, price_df, feature_cols):
        """
        전략 준비 (AI 학습 + 모멘텀 계산)
        """
        print("=" * 60)
        print("Hybrid 전략 준비")
        print("=" * 60)
        
        self.feature_cols = feature_cols
        
        # AI 학습
        print("\n[1] AI (XGBoost) 학습...")
        self.ai_strategy = AIStrategy()
        self.ai_strategy.train(train_df, feature_cols)
        
        # 모멘텀 준비
        print("\n[2] 모멘텀 점수 계산...")
        self.momentum_strategy = CustomStrategy()
        tuesday_df = filter_tuesday(price_df)
        self.score_df, _, _ = self.momentum_strategy.prepare(price_df, tuesday_df)
        
        # SPY 데이터 저장 (시장 필터링용)
        if 'SPY' in price_df.columns:
            self.spy_df = price_df[['SPY']].copy()
            self.spy_df.columns = ['close']
            print(f"\n[3] SPY 데이터 로드: {len(self.spy_df)}일")
        
        self.is_prepared = True
        print("\n✅ Hybrid 전략 준비 완료!")
        if self.use_market_filter:
            print(f"   시장 필터링: ON (SPY > {MARKET_FILTER_MA_PERIOD}일 MA)")
        else:
            print("   시장 필터링: OFF")
    
    def check_market_condition(self, date):
        """
        시장 상황 체크: SPY > 20일 이동평균
        
        Args:
            date: 체크할 날짜
        
        Returns:
            tuple: (매수가능 여부, SPY가격, MA가격)
        """
        if not self.use_market_filter:
            return True, 0, 0
        
        if self.spy_df is None or self.spy_df.empty:
            return True, 0, 0
        
        date_ts = pd.Timestamp(date)
        
        # 해당 날짜까지의 SPY 데이터
        spy_data = self.spy_df[self.spy_df.index <= date_ts]
        
        if len(spy_data) < MARKET_FILTER_MA_PERIOD:
            return True, 0, 0  # 데이터 부족하면 매수 허용
        
        # 20일 이동평균 계산
        spy_ma = spy_data['close'].rolling(MARKET_FILTER_MA_PERIOD).mean().iloc[-1]
        spy_price = spy_data['close'].iloc[-1]
        
        # SPY > 20일 MA면 매수 가능
        is_bullish = spy_price > spy_ma
        
        return is_bullish, spy_price, spy_ma
    
    def select_stocks(self, current_df, price_df, date):
        """
        오늘 날짜 기준 종목 선정
        
        Args:
            current_df: 피처가 포함된 데이터프레임
            price_df: 가격 데이터 (피벗)
            date: 기준 날짜
        
        Returns:
            dict: picks, scores, allocations, prices, market_status
        """
        if not self.is_prepared:
            raise ValueError("prepare() 먼저 실행하세요.")
        
        date_ts = pd.Timestamp(date)
        
        # ----- 시장 필터링 체크 -----
        is_bullish, spy_price, spy_ma = self.check_market_condition(date)
        
        if not is_bullish:
            print(f"⚠️ 시장 필터링 발동: SPY ${spy_price:.2f} < MA20 ${spy_ma:.2f}")
            print("   → 매수 보류 (현금 보유)")
            return {
                'picks': [],
                'scores': [],
                'allocations': [],
                'prices': {'SPY': spy_price},
                'market_filter': True,
                'spy_price': spy_price,
                'spy_ma': spy_ma
            }
        
        # 해당 날짜 데이터
        date_df = current_df[current_df['date'] == date_ts].copy()
        if date_df.empty:
            return None
        
        # ----- 모멘텀 점수 -----
        if date_ts not in self.score_df.index:
            # 가장 최근 화요일 점수 사용
            available_dates = self.score_df.index[self.score_df.index <= date_ts]
            if len(available_dates) == 0:
                return None
            date_ts_momentum = available_dates[-1]
        else:
            date_ts_momentum = date_ts
        
        m_scores = self.score_df.loc[date_ts_momentum].drop(labels=['SPY'], errors='ignore').dropna()
        
        if m_scores.empty:
            return None
        
        # ----- AI 확률 -----
        ai_pred = self.ai_strategy.predict(date_df, self.feature_cols)
        
        if ai_pred.empty:
            return None
        
        # ----- 정규화 -----
        m_min, m_max = m_scores.min(), m_scores.max()
        m_norm = (m_scores - m_min) / (m_max - m_min + 1e-8)
        
        # ----- 결합 -----
        merged = ai_pred.copy()
        merged['m_score'] = merged['symbol'].map(m_norm)
        merged = merged.dropna()
        
        if merged.empty:
            return None
        
        # 가중 평균
        merged['hybrid_score'] = (merged['m_score'] * self.weight_m + 
                                   merged['probability'] * self.weight_ai)
        
        merged = merged.sort_values('hybrid_score', ascending=False)
        
        # Top 3 선정
        top_picks = merged.head(TOP_N)
        n_picks = len(top_picks)
        
        if n_picks == 0:
            return None
        
        if n_picks >= 3:
            allocations = [0.4, 0.3, 0.3]
        elif n_picks == 2:
            allocations = [0.5, 0.5]
        else:
            allocations = [1.0]
        
        # SPY 가격 추가
        prices = dict(zip(top_picks['symbol'], top_picks['close']))
        prices['SPY'] = spy_price
        
        return {
            'picks': top_picks['symbol'].tolist(),
            'scores': top_picks['hybrid_score'].tolist(),
            'allocations': allocations[:n_picks],
            'prices': prices,
            'market_filter': False,
            'spy_price': spy_price,
            'spy_ma': spy_ma
        }


# ============================================
# [3] 데이터 준비
# ============================================

def prepare_hybrid_data():
    """
    Hybrid 전략용 데이터 준비
    
    Returns:
        tuple: (train_df, current_df, price_df, features)
    """
    print("=" * 60)
    print("Hybrid 데이터 준비")
    print("=" * 60)
    
    # S&P 500 종목
    sp500 = get_sp500_list()
    symbols = sp500['symbol'].tolist() + ['SPY']
    
    # 학습 데이터 (2020-2023)
    print("\n[1] 학습 데이터 다운로드...")
    train_raw = get_backtest_data(symbols, start_date=TRAIN_START, end_date=TRAIN_END)
    
    # 현재 데이터 (최근 6개월)
    print("\n[2] 현재 데이터 다운로드...")
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=180)).strftime('%Y-%m-%d')
    current_raw = get_backtest_data(symbols, start_date=start_date, end_date=end_date)
    
    # 피처 생성
    print("\n[3] 피처 생성...")
    from ai_data import create_features
    
    train_df = create_features(train_raw)
    current_df = create_features(current_raw)
    
    features = get_feature_columns(train_df)
    
    # 가격 데이터 (피벗)
    price_df = current_raw.pivot(index='date', columns='symbol', values='close')
    
    print(f"\n✅ 데이터 준비 완료!")
    print(f"  학습 데이터: {len(train_df):,}개")
    print(f"  현재 데이터: {len(current_df):,}개")
    print(f"  피처 수: {len(features)}개")
    
    return train_df, current_df, price_df, features


# ============================================
# [4] 오늘의 신호 생성
# ============================================

def get_hybrid_signal():
    """
    오늘의 Hybrid 신호 생성
    
    Returns:
        dict: 종목 선정 결과
    """
    print("=" * 60)
    print("Hybrid 신호 생성")
    print("=" * 60)
    
    # 데이터 준비
    train_df, current_df, price_df, features = prepare_hybrid_data()
    
    # 전략 준비 (시장 필터링 ON)
    strategy = HybridTradingStrategy(use_market_filter=True)
    strategy.prepare(train_df, price_df, features)
    
    # 오늘 신호
    today = datetime.now().strftime('%Y-%m-%d')
    
    # 가장 최근 거래일 찾기
    available_dates = current_df['date'].unique()
    available_dates = sorted(available_dates)
    
    if len(available_dates) == 0:
        print("❌ 데이터 없음")
        return None
    
    latest_date = available_dates[-1]
    print(f"\n기준일: {latest_date}")
    
    # 종목 선정
    result = strategy.select_stocks(current_df, price_df, latest_date)
    
    if result is None:
        print("❌ 선정된 종목 없음")
        return None
    
    # 시장 필터링 발동 체크
    if result.get('market_filter', False):
        print(f"\n⚠️ 시장 필터링 발동!")
        print(f"   SPY: ${result.get('spy_price', 0):.2f} < MA20: ${result.get('spy_ma', 0):.2f}")
        print(f"   → 이번 주 매수 보류 (현금 보유)")
        return result
    
    print(f"\n✅ 선정 종목:")
    for i, (symbol, score) in enumerate(zip(result['picks'], result['scores'])):
        price = result['prices'].get(symbol, 0)
        alloc = result['allocations'][i]
        print(f"  {i+1}. {symbol}: 점수 {score:.4f}, 가격 ${price:.2f}, 비중 {alloc*100:.0f}%")
    
    return result


# ============================================
# [5] 리밸런싱 계산
# ============================================

def calculate_hybrid_rebalancing(portfolio, signal, total_capital, available_cash=None, min_trade_amount=50):
    """
    리밸런싱 계산
    
    Args:
        portfolio: 현재 보유 {symbol: {shares, avg_price, current_price}}
        signal: 새 신호 {picks, scores, allocations, prices}
        total_capital: 총 자본금
        available_cash: 사용 가능한 현금 (None이면 total_capital 사용)
        min_trade_amount: 최소 거래 금액
    
    Returns:
        dict: 리밸런싱 액션
    """
    actions = []
    
    new_symbols = set(signal['picks']) if signal else set()
    current_symbols = set(portfolio.keys()) if portfolio else set()
    
    # 매도 금액 먼저 계산 (현금 추정용)
    sell_amount = 0
    
    # 1. 매도 (신호에서 제외된 종목)
    for symbol in current_symbols - new_symbols:
        info = portfolio[symbol]
        current_price = info.get('current_price', info['avg_price'])
        ret_pct = (current_price - info['avg_price']) / info['avg_price'] * 100
        amount = info['shares'] * current_price
        sell_amount += amount
        
        actions.append({
            'action': 'SELL',
            'symbol': symbol,
            'shares': info['shares'],
            'price': current_price,
            'amount': amount,
            'reason': '신호에서 제외',
            'return_pct': ret_pct
        })
    
    # 사용 가능한 현금 계산
    if available_cash is None:
        available_cash = total_capital
    cash_available = available_cash + sell_amount
    
    # 2. 매수/조정 (신규 및 기존)
    if signal:
        for i, symbol in enumerate(signal['picks']):
            target_alloc = signal['allocations'][i]
            target_amount = total_capital * target_alloc
            price = signal['prices'].get(symbol, 0)
            score = signal['scores'][i]  # 점수 추가
            
            if price <= 0:
                continue
            
            current_amount = 0
            current_shares = 0
            
            if symbol in portfolio:
                current_shares = portfolio[symbol]['shares']
                current_price = portfolio[symbol].get('current_price', price)
                current_amount = current_shares * current_price
            
            diff = target_amount - current_amount
            
            if abs(diff) < min_trade_amount:
                # 유지
                if current_shares > 0:
                    actions.append({
                        'action': 'HOLD',
                        'symbol': symbol,
                        'shares': current_shares,
                        'price': price,
                        'amount': current_amount,
                        'reason': '유지',
                        'score': score,
                        'allocation': target_alloc
                    })
            elif diff > 0:
                # 매수 - 현금 체크
                shares_to_buy = int(diff / price)
                buy_amount = shares_to_buy * price
                
                # 현금 부족 시 조정
                if buy_amount > cash_available:
                    shares_to_buy = int(cash_available / price)
                    buy_amount = shares_to_buy * price
                    print(f"⚠️ {symbol}: 현금 부족으로 {shares_to_buy}주로 조정")
                
                if shares_to_buy > 0:
                    action_type = 'ADD' if current_shares > 0 else 'BUY'
                    actions.append({
                        'action': action_type,
                        'symbol': symbol,
                        'shares': shares_to_buy,
                        'price': price,
                        'amount': buy_amount,
                        'reason': '비중 증가' if action_type == 'ADD' else '신규 매수',
                        'score': score,
                        'allocation': target_alloc
                    })
                    cash_available -= buy_amount  # 남은 현금 업데이트
            else:
                # 비중 축소
                shares_to_sell = int(abs(diff) / price)
                shares_to_sell = min(shares_to_sell, current_shares)
                if shares_to_sell > 0:
                    ret_pct = (price - portfolio[symbol]['avg_price']) / portfolio[symbol]['avg_price'] * 100
                    sell_amt = shares_to_sell * price
                    actions.append({
                        'action': 'REDUCE',
                        'symbol': symbol,
                        'shares': shares_to_sell,
                        'price': price,
                        'amount': sell_amt,
                        'reason': '비중 축소',
                        'return_pct': ret_pct,
                        'score': score,
                        'allocation': target_alloc
                    })
                    cash_available += sell_amt  # 현금 증가
    
    # 요약 계산
    total_buy = sum(a['amount'] for a in actions if a['action'] in ['BUY', 'ADD'])
    total_sell = sum(a['amount'] for a in actions if a['action'] in ['SELL', 'REDUCE'])
    
    return {
        'actions': actions,
        'summary': {
            'total_buy': total_buy,
            'total_sell': total_sell,
            'net_cash_change': total_sell - total_buy
        }
    }


# ============================================
# [6] 리밸런싱 메시지 출력
# ============================================

def print_hybrid_rebalancing(rebalancing):
    """
    리밸런싱 결과 출력
    """
    print("\n" + "=" * 60)
    print("📊 Hybrid 리밸런싱")
    print("=" * 60)
    
    actions = rebalancing['actions']
    summary = rebalancing['summary']
    
    # 액션별 분류
    sells = [a for a in actions if a['action'] == 'SELL']
    reduces = [a for a in actions if a['action'] == 'REDUCE']
    holds = [a for a in actions if a['action'] == 'HOLD']
    adds = [a for a in actions if a['action'] == 'ADD']
    buys = [a for a in actions if a['action'] == 'BUY']
    
    if sells:
        print("\n🔴 매도 (전량)")
        for a in sells:
            ret = a.get('return_pct', 0)
            print(f"  • {a['symbol']} {a['shares']}주 @ ${a['price']:.2f} ({ret:+.1f}%)")
    
    if reduces:
        print("\n🟠 비중 축소")
        for a in reduces:
            ret = a.get('return_pct', 0)
            print(f"  • {a['symbol']} -{a['shares']}주 @ ${a['price']:.2f} ({ret:+.1f}%)")
    
    if holds:
        print("\n⚪ 유지")
        for a in holds:
            print(f"  • {a['symbol']} {a['shares']}주")
    
    if adds:
        print("\n🟢 추가 매수")
        for a in adds:
            print(f"  • {a['symbol']} +{a['shares']}주 @ ${a['price']:.2f}")
    
    if buys:
        print("\n🟢 신규 매수")
        for a in buys:
            print(f"  • {a['symbol']} {a['shares']}주 @ ${a['price']:.2f}")
    
    print(f"\n💰 요약")
    print(f"  매도 금액: ${summary['total_sell']:,.2f}")
    print(f"  매수 금액: ${summary['total_buy']:,.2f}")
    print(f"  현금 변화: ${summary['net_cash_change']:+,.2f}")


# ============================================
# [7] Telegram 메시지 전송
# ============================================

def send_hybrid_signal(signal, total_capital):
    """
    Hybrid 신호 텔레그램 전송
    """
    if signal is None:
        return
    
    today = datetime.now().strftime('%Y-%m-%d')
    
    msg = f"🤖 Hybrid 신호 ({today})\n"
    msg += f"Capital: ${total_capital:,.0f}\n"
    msg += f"가중치: M{WEIGHT_MOMENTUM*100:.0f}% + AI{WEIGHT_AI*100:.0f}%\n\n"
    
    for i, (symbol, score) in enumerate(zip(signal['picks'], signal['scores'])):
        price = signal['prices'].get(symbol, 0)
        alloc = signal['allocations'][i]
        shares = int(total_capital * alloc / price) if price > 0 else 0
        
        msg += f"{i+1}. {symbol}\n"
        msg += f"   점수: {score:.4f}\n"
        msg += f"   가격: ${price:.2f}\n"
        msg += f"   비중: {alloc*100:.0f}% ({shares}주)\n\n"
    
    send_message(msg)


def send_hybrid_rebalancing(rebalancing, total_capital, signal=None):
    """
    Hybrid 리밸런싱 텔레그램 전송
    """
    today = datetime.now().strftime('%Y-%m-%d')
    
    actions = rebalancing['actions']
    summary = rebalancing['summary']
    
    msg = f"🤖 Hybrid 리밸런싱 ({today})\n"
    msg += f"Capital: ${total_capital:,.0f}\n"
    msg += f"가중치: M{WEIGHT_MOMENTUM*100:.0f}% + AI{WEIGHT_AI*100:.0f}%\n\n"
    
    # 선정 종목 (점수 포함)
    if signal:
        msg += "📊 선정 종목:\n"
        for i, (symbol, score) in enumerate(zip(signal['picks'], signal['scores'])):
            price = signal['prices'].get(symbol, 0)
            msg += f"{i+1}. {symbol}: 점수 {score:.4f}, 가격 ${price:.2f}\n"
        msg += "\n"
    
    # 액션별 분류
    sells = [a for a in actions if a['action'] == 'SELL']
    reduces = [a for a in actions if a['action'] == 'REDUCE']
    holds = [a for a in actions if a['action'] == 'HOLD']
    adds = [a for a in actions if a['action'] == 'ADD']
    buys = [a for a in actions if a['action'] == 'BUY']
    
    if sells:
        msg += "🔴 매도 (전량)\n"
        for a in sells:
            ret = a.get('return_pct', 0)
            msg += f"• {a['symbol']} {a['shares']}주 @ ${a['price']:.2f} ({ret:+.1f}%)\n"
        msg += "\n"
    
    if reduces:
        msg += "🟠 비중 축소\n"
        for a in reduces:
            msg += f"• {a['symbol']} -{a['shares']}주 @ ${a['price']:.2f}\n"
        msg += "\n"
    
    if holds:
        msg += "⚪ 유지\n"
        for a in holds:
            msg += f"• {a['symbol']} {a['shares']}주\n"
        msg += "\n"
    
    if adds:
        msg += "🟢 추가 매수\n"
        for a in adds:
            msg += f"• {a['symbol']} +{a['shares']}주 @ ${a['price']:.2f}\n"
        msg += "\n"
    
    if buys:
        msg += "🟢 신규 매수\n"
        for a in buys:
            msg += f"• {a['symbol']} {a['shares']}주 @ ${a['price']:.2f}\n"
        msg += "\n"
    
    msg += "💰 요약\n"
    msg += f"매도: ${summary['total_sell']:,.0f}\n"
    msg += f"매수: ${summary['total_buy']:,.0f}\n"
    msg += f"현금: ${summary['net_cash_change']:+,.0f}"
    
    send_message(msg)


# ============================================
# [8] 메인 실행
# ============================================

def run_hybrid_weekly(total_capital=INITIAL_CAPITAL):
    """
    Hybrid 주간 실행
    
    Args:
        total_capital: 총 자본금
    """
    print("=" * 60)
    print("🤖 Hybrid 주간 실행")
    print("=" * 60)
    print(f"자본금: ${total_capital:,}")
    print(f"가중치: 모멘텀 {WEIGHT_MOMENTUM*100:.0f}% + AI {WEIGHT_AI*100:.0f}%")
    
    # 1. Sheets 연결
    sheets = HybridSheetsManager()
    
    # 2. 신호 생성
    signal = get_hybrid_signal()
    
    if signal is None:
        print("❌ 신호 생성 실패")
        return
    
    # 3. 시장 필터링 체크
    if signal.get('market_filter', False):
        print("\n⚠️ 시장 필터링 발동 - 매수 보류")
        
        # 현재 보유 종목 전량 매도
        portfolio = sheets.get_holdings()
        
        if portfolio:
            print("📤 보유 종목 전량 매도:")
            
            # 현재 가격 가져오기
            import yfinance as yf
            for symbol in portfolio:
                try:
                    ticker = yf.Ticker(symbol)
                    hist = ticker.history(period='1d')
                    if not hist.empty:
                        portfolio[symbol]['current_price'] = hist['Close'].iloc[-1]
                except:
                    portfolio[symbol]['current_price'] = portfolio[symbol]['avg_price']
            
            total_sell_amount = 0
            sell_symbols = []
            
            for symbol, info in portfolio.items():
                shares = info['shares']
                price = info.get('current_price', info['avg_price'])
                amount = shares * price
                total_sell_amount += amount
                sell_symbols.append(symbol)
                
                ret_pct = (price - info['avg_price']) / info['avg_price'] * 100
                print(f"  • {symbol}: {shares}주 @ ${price:.2f} ({ret_pct:+.1f}%)")
                
                # Holdings에서 제거
                sheets.sheets.remove_holding(symbol)
                
                # Trade 기록
                sheets.save_trade({
                    'symbol': symbol,
                    'action': 'SELL',
                    'shares': shares,
                    'price': price,
                    'amount': amount,
                    'return_pct': ret_pct
                })
            
            # 현금 업데이트
            if total_sell_amount > 0:
                sheets.update_cash(total_sell_amount, f"시장필터링 매도: {', '.join(sell_symbols)}")
                commission = total_sell_amount * SELL_COMMISSION
                sheets.update_cash(-commission, "수수료")
        
        # Telegram 전송
        spy_price = signal.get('spy_price', 0)
        spy_ma = signal.get('spy_ma', 0)
        
        msg = f"⚠️ Hybrid 시장 필터링 ({datetime.now().strftime('%Y-%m-%d')})\n\n"
        msg += f"SPY: ${spy_price:.2f}\n"
        msg += f"MA20: ${spy_ma:.2f}\n"
        msg += f"상태: 하락 추세 ❌\n\n"
        msg += "→ 이번 주 매수 보류\n"
        msg += "→ 현금 보유"
        
        send_message(msg)
        
        # Daily_Value 저장
        cash = sheets.get_cash()
        new_holdings = sheets.get_holdings()
        sheets.save_daily_value(new_holdings, signal['prices'], cash, spy_price)
        
        print("\n✅ Hybrid 주간 실행 완료 (시장 필터링)")
        return {'signal': signal, 'market_filter': True}
    
    # 4. 현재 포트폴리오 (Sheets에서 가져오기)
    portfolio = sheets.get_holdings()
    
    # 현재 가격 추가
    for symbol in portfolio:
        if symbol in signal['prices']:
            portfolio[symbol]['current_price'] = signal['prices'][symbol]
        else:
            portfolio[symbol]['current_price'] = portfolio[symbol]['avg_price']
    
    print(f"📊 현재 보유: {list(portfolio.keys()) if portfolio else '없음'}")
    
    # 5. 현재 현금 가져오기
    available_cash = sheets.get_cash()
    
    # 6. 리밸런싱 계산 (현금 전달)
    rebalancing = calculate_hybrid_rebalancing(portfolio, signal, total_capital, available_cash)
    
    # 7. 출력
    print_hybrid_rebalancing(rebalancing)
    
    # 8. Telegram 전송 (signal 포함)
    send_hybrid_rebalancing(rebalancing, total_capital, signal)
    
    # 9. Sheets 기록
    # 신호 저장
    sheets.save_signal(signal)
    
    # 거래 저장
    for action in rebalancing['actions']:
        if action['action'] != 'HOLD':
            sheets.save_trade(action)
    
    # Holdings 업데이트
    sheets.update_holdings(rebalancing['actions'], signal['prices'])
    
    # 9. 현금 업데이트
    # 매도 금액 입금
    if rebalancing['summary']['total_sell'] > 0:
        sheets.update_cash(
            rebalancing['summary']['total_sell'], 
            f"매도: {', '.join([a['symbol'] for a in rebalancing['actions'] if a['action'] in ['SELL', 'REDUCE']])}"
        )
    
    # 매수 금액 출금
    if rebalancing['summary']['total_buy'] > 0:
        sheets.update_cash(
            -rebalancing['summary']['total_buy'], 
            f"매수: {', '.join([a['symbol'] for a in rebalancing['actions'] if a['action'] in ['BUY', 'ADD']])}"
        )
    
    # 수수료 차감
    total_commission = (rebalancing['summary']['total_buy'] + rebalancing['summary']['total_sell']) * BUY_COMMISSION
    if total_commission > 0:
        sheets.update_cash(-total_commission, "수수료")
    
    # 10. Daily_Value 저장
    # 현재 현금 가져오기
    cash = sheets.get_cash()
    
    # SPY 가격 가져오기
    spy_price = signal['prices'].get('SPY', 0)
    
    # 새 포트폴리오로 Daily_Value 저장
    new_holdings = sheets.get_holdings()
    sheets.save_daily_value(new_holdings, signal['prices'], cash, spy_price)
    
    print("\n✅ Hybrid 주간 실행 완료!")
    
    return {
        'signal': signal,
        'rebalancing': rebalancing
    }


# ============================================
# [9] Daily 실행 (월,수,목,금)
# ============================================

def get_current_prices(symbols):
    """
    현재 가격 가져오기
    
    Args:
        symbols: 종목 리스트
    
    Returns:
        dict: {symbol: price}
    """
    import yfinance as yf
    
    prices = {}
    for symbol in symbols:
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period='1d')
            if not hist.empty:
                prices[symbol] = hist['Close'].iloc[-1]
        except Exception as e:
            print(f"⚠️ {symbol} 가격 조회 실패: {e}")
    
    return prices


def check_stop_loss(holdings, current_prices, stop_loss_pct=STOP_LOSS):
    """
    손절 체크
    
    Args:
        holdings: 보유 종목 dict
        current_prices: 현재 가격 dict
        stop_loss_pct: 손절 기준 (기본 -7%)
    
    Returns:
        list: 손절 대상 종목 리스트
    """
    stop_loss_list = []
    
    for symbol, info in holdings.items():
        avg_price = info.get('avg_price', 0)
        current_price = current_prices.get(symbol, avg_price)
        
        if avg_price > 0:
            return_pct = (current_price - avg_price) / avg_price
            
            if return_pct <= stop_loss_pct:
                stop_loss_list.append({
                    'symbol': symbol,
                    'shares': info.get('shares', 0),
                    'avg_price': avg_price,
                    'current_price': current_price,
                    'return_pct': return_pct * 100
                })
    
    return stop_loss_list


def run_hybrid_daily(total_capital=INITIAL_CAPITAL):
    """
    Hybrid Daily 실행 (월,수,목,금)
    - 손절 체크
    - 일일 가치 기록
    
    Args:
        total_capital: 총 자본금
    """
    print("=" * 60)
    print("🤖 Hybrid Daily 실행")
    print("=" * 60)
    
    today = datetime.now().strftime('%Y-%m-%d')
    
    # 1. Sheets 연결
    sheets = HybridSheetsManager()
    
    # 2. 현재 보유 종목 가져오기
    holdings = sheets.get_holdings()
    
    # 4. 현재 가격 가져오기 (보유종목 + SPY)
    symbols = list(holdings.keys()) + ['SPY'] if holdings else ['SPY']
    current_prices = get_current_prices(symbols)
    
    spy_price = current_prices.get('SPY', 0)
    print(f"📈 SPY: ${spy_price:.2f}")
    
    # 5. 보유 종목이 있으면 손절 체크
    if holdings:
        print(f"📊 보유 종목: {list(holdings.keys())}")
        
        stop_loss_list = check_stop_loss(holdings, current_prices)
        
        if stop_loss_list:
            print("\n🔴 손절 대상:")
            msg = f"🚨 Hybrid 손절 알림\n\n"
            
            total_stop_loss_amount = 0
            
            for item in stop_loss_list:
                print(f"  • {item['symbol']}: {item['return_pct']:.1f}%")
                msg += f"🔴 {item['symbol']}\n"
                msg += f"   매수가: ${item['avg_price']:.2f}\n"
                msg += f"   현재가: ${item['current_price']:.2f}\n"
                msg += f"   수익률: {item['return_pct']:.1f}%\n\n"
                
                # 손절 금액 계산
                sell_amount = item['shares'] * item['current_price']
                total_stop_loss_amount += sell_amount
                
                # Holdings에서 제거
                sheets.sheets.remove_holding(item['symbol'])
                
                # Trade 기록
                sheets.save_trade({
                    'symbol': item['symbol'],
                    'action': 'STOP_LOSS',
                    'shares': item['shares'],
                    'price': item['current_price'],
                    'amount': sell_amount,
                    'return_pct': item['return_pct']
                })
            
            # 현금 업데이트 (손절 매도 금액 입금)
            if total_stop_loss_amount > 0:
                sheets.update_cash(total_stop_loss_amount, f"손절 매도: {', '.join([i['symbol'] for i in stop_loss_list])}")
                
                # 수수료 차감
                commission = total_stop_loss_amount * SELL_COMMISSION
                sheets.update_cash(-commission, "손절 수수료")
            
            # Telegram 전송
            send_message(msg)
            
            # 손절 후 Holdings 다시 로드
            holdings = sheets.get_holdings()
        else:
            print("\n✅ 손절 대상 없음")
    else:
        print("📊 보유 종목 없음 (현금 보유 중)")
    
    # 6. 현재 현금 가져오기
    cash = sheets.get_cash()
    
    # 주식 가치 계산
    stocks_value = 0
    if holdings:
        stocks_value = sum(
            holdings.get(s, {}).get('shares', 0) * current_prices.get(s, 0)
            for s in holdings
        )
    
    # 총 포트폴리오 가치
    total_value = stocks_value + cash
    
    # 7. 이전 Daily_Value에서 수익률 계산 (오늘 제외)
    daily_return = 0
    spy_return = 0
    alpha = 0
    prev_value = total_capital
    prev_spy = spy_price
    
    try:
        ws = sheets.sheets.spreadsheet.worksheet("Daily_Value")
        data = ws.get_all_values()
        
        if len(data) > 1:
            # 오늘 날짜가 아닌 마지막 행 찾기
            for row in reversed(data[1:]):
                if row[0] != today:
                    prev_value = float(row[1]) if row[1] else total_capital
                    prev_spy = float(row[5]) if row[5] else spy_price
                    break
            
            if prev_value > 0:
                daily_return = (total_value - prev_value) / prev_value * 100
            
            if prev_spy > 0 and spy_price > 0:
                spy_return = (spy_price - prev_spy) / prev_spy * 100
                alpha = daily_return - spy_return
    except:
        pass
    
    # 8. Daily_Value 저장
    sheets.save_daily_value(holdings, current_prices, cash, spy_price)
    
    # 9. Daily Summary 텔레그램 전송
    msg = f"📊 Hybrid Daily Summary ({today})\n"
    msg += f"Portfolio: ${total_value:,.2f}\n"
    msg += f"Daily: {daily_return:+.2f}%\n"
    msg += f"SPY: {spy_return:+.2f}%\n"
    msg += f"Alpha: {alpha:+.2f}%\n\n"
    
    if holdings:
        msg += "Holdings:\n"
        for symbol, info in holdings.items():
            shares = info.get('shares', 0)
            avg_price = info.get('avg_price', 0)
            current_price = current_prices.get(symbol, avg_price)
            
            if avg_price > 0:
                return_pct = (current_price - avg_price) / avg_price * 100
            else:
                return_pct = 0
            
            msg += f"• {symbol}: {shares}주 ({return_pct:+.2f}%)\n"
    else:
        msg += "Holdings: 없음 (현금 보유)"
    
    send_message(msg)
    
    print("\n✅ Hybrid Daily 실행 완료!")


# ============================================
# [10] 테스트
# ============================================

if __name__ == "__main__":
    print("Hybrid Trading 모듈 테스트")
    print("=" * 60)
    
    # 간단 테스트: 신호만 생성
    try:
        signal = get_hybrid_signal()
        
        if signal:
            print("\n✅ 신호 생성 성공!")
            send_hybrid_signal(signal, INITIAL_CAPITAL)
    except Exception as e:
        print(f"❌ 에러: {e}")