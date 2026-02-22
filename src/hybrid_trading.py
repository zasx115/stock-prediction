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
                'spy_price': 0,
                'market_trend': ''
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
            # 주식 가치 계산
            stocks_value = 0
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
            
            # 이전 데이터 가져오기 (수익률 계산용)
            data = ws.get_all_values()
            prev_value = None
            prev_spy = None
            
            if len(data) > 1:
                last_row = data[-1]
                try:
                    prev_value = float(last_row[1]) if last_row[1] else None
                    prev_spy = float(last_row[5]) if last_row[5] else None
                except:
                    pass
            
            # 수익률 계산
            daily_return = 0
            spy_return = 0
            alpha = 0
            
            if prev_value and prev_value > 0:
                daily_return = (total_value - prev_value) / prev_value * 100
            
            if prev_spy and prev_spy > 0 and spy_price > 0:
                spy_return = (spy_price - prev_spy) / prev_spy * 100
                alpha = daily_return - spy_return
            
            # 행 추가
            row = [
                datetime.now().strftime('%Y-%m-%d'),
                round(total_value, 2),
                round(cash, 2),
                round(stocks_value, 2),
                round(daily_return, 2),
                round(spy_price, 2),
                round(spy_return, 2),
                round(alpha, 2)
            ]
            ws.append_row(row)
            print(f"✅ Daily_Value 저장: ${total_value:,.2f}")
            
        except Exception as e:
            print(f"⚠️ Daily_Value 저장 실패: {e}")


# ============================================
# [2] Hybrid 전략 클래스 (간소화 버전)
# ============================================

class HybridTradingStrategy:
    """
    하이브리드 트레이딩 전략
    모멘텀 점수 + AI 확률 결합
    """
    
    def __init__(self, weight_momentum=WEIGHT_MOMENTUM, weight_ai=WEIGHT_AI):
        self.weight_m = weight_momentum
        self.weight_ai = weight_ai
        
        self.ai_strategy = None
        self.momentum_strategy = None
        self.score_df = None
        self.feature_cols = None
        
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
        
        self.is_prepared = True
        print("\n✅ Hybrid 전략 준비 완료!")
    
    def select_stocks(self, current_df, price_df, date):
        """
        오늘 날짜 기준 종목 선정
        
        Args:
            current_df: 피처가 포함된 데이터프레임
            price_df: 가격 데이터 (피벗)
            date: 기준 날짜
        
        Returns:
            dict: picks, scores, allocations, prices
        """
        if not self.is_prepared:
            raise ValueError("prepare() 먼저 실행하세요.")
        
        date_ts = pd.Timestamp(date)
        
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
        
        return {
            'picks': top_picks['symbol'].tolist(),
            'scores': top_picks['hybrid_score'].tolist(),
            'allocations': allocations[:n_picks],
            'prices': dict(zip(top_picks['symbol'], top_picks['close']))
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
    
    # 전략 준비
    strategy = HybridTradingStrategy()
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
    
    print(f"\n✅ 선정 종목:")
    for i, (symbol, score) in enumerate(zip(result['picks'], result['scores'])):
        price = result['prices'].get(symbol, 0)
        alloc = result['allocations'][i]
        print(f"  {i+1}. {symbol}: 점수 {score:.4f}, 가격 ${price:.2f}, 비중 {alloc*100:.0f}%")
    
    return result


# ============================================
# [5] 리밸런싱 계산
# ============================================

def calculate_hybrid_rebalancing(portfolio, signal, total_capital, min_trade_amount=50):
    """
    리밸런싱 계산
    
    Args:
        portfolio: 현재 보유 {symbol: {shares, avg_price, current_price}}
        signal: 새 신호 {picks, scores, allocations, prices}
        total_capital: 총 자본금
        min_trade_amount: 최소 거래 금액
    
    Returns:
        dict: 리밸런싱 액션
    """
    actions = []
    
    new_symbols = set(signal['picks']) if signal else set()
    current_symbols = set(portfolio.keys()) if portfolio else set()
    
    # 1. 매도 (신호에서 제외된 종목)
    for symbol in current_symbols - new_symbols:
        info = portfolio[symbol]
        current_price = info.get('current_price', info['avg_price'])
        ret_pct = (current_price - info['avg_price']) / info['avg_price'] * 100
        
        actions.append({
            'action': 'SELL',
            'symbol': symbol,
            'shares': info['shares'],
            'price': current_price,
            'amount': info['shares'] * current_price,
            'reason': '신호에서 제외',
            'return_pct': ret_pct
        })
    
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
                # 매수
                shares_to_buy = int(diff / price)
                if shares_to_buy > 0:
                    action_type = 'ADD' if current_shares > 0 else 'BUY'
                    actions.append({
                        'action': action_type,
                        'symbol': symbol,
                        'shares': shares_to_buy,
                        'price': price,
                        'amount': shares_to_buy * price,
                        'reason': '비중 증가' if action_type == 'ADD' else '신규 매수',
                        'score': score,
                        'allocation': target_alloc
                    })
            else:
                # 비중 축소
                shares_to_sell = int(abs(diff) / price)
                shares_to_sell = min(shares_to_sell, current_shares)
                if shares_to_sell > 0:
                    ret_pct = (price - portfolio[symbol]['avg_price']) / portfolio[symbol]['avg_price'] * 100
                    actions.append({
                        'action': 'REDUCE',
                        'symbol': symbol,
                        'shares': shares_to_sell,
                        'price': price,
                        'amount': shares_to_sell * price,
                        'reason': '비중 축소',
                        'return_pct': ret_pct,
                        'score': score,
                        'allocation': target_alloc
                    })
    
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
    
    # 3. 현재 포트폴리오 (Sheets에서 가져오기)
    portfolio = sheets.get_holdings()
    
    # 현재 가격 추가
    for symbol in portfolio:
        if symbol in signal['prices']:
            portfolio[symbol]['current_price'] = signal['prices'][symbol]
        else:
            portfolio[symbol]['current_price'] = portfolio[symbol]['avg_price']
    
    print(f"📊 현재 보유: {list(portfolio.keys()) if portfolio else '없음'}")
    
    # 4. 리밸런싱 계산
    rebalancing = calculate_hybrid_rebalancing(portfolio, signal, total_capital)
    
    # 5. 출력
    print_hybrid_rebalancing(rebalancing)
    
    # 6. Telegram 전송 (signal 포함)
    send_hybrid_rebalancing(rebalancing, total_capital, signal)
    
    # 7. Sheets 기록
    # 신호 저장
    sheets.save_signal(signal)
    
    # 거래 저장
    for action in rebalancing['actions']:
        if action['action'] != 'HOLD':
            sheets.save_trade(action)
    
    # Holdings 업데이트
    sheets.update_holdings(rebalancing['actions'], signal['prices'])
    
    # 8. Daily_Value 저장
    # 현금 계산 (매수 후 남은 금액)
    cash = total_capital - rebalancing['summary']['total_buy'] + rebalancing['summary']['total_sell']
    
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
    
    # 1. Sheets 연결
    sheets = HybridSheetsManager()
    
    # 2. 현재 보유 종목 가져오기
    holdings = sheets.get_holdings()
    
    if not holdings:
        print("📊 보유 종목 없음")
        return
    
    print(f"📊 보유 종목: {list(holdings.keys())}")
    
    # 3. 현재 가격 가져오기
    symbols = list(holdings.keys()) + ['SPY']
    current_prices = get_current_prices(symbols)
    
    spy_price = current_prices.get('SPY', 0)
    print(f"📈 SPY: ${spy_price:.2f}")
    
    # 4. 손절 체크
    stop_loss_list = check_stop_loss(holdings, current_prices)
    
    if stop_loss_list:
        print("\n🔴 손절 대상:")
        msg = f"🚨 Hybrid 손절 알림\n\n"
        
        for item in stop_loss_list:
            print(f"  • {item['symbol']}: {item['return_pct']:.1f}%")
            msg += f"🔴 {item['symbol']}\n"
            msg += f"   매수가: ${item['avg_price']:.2f}\n"
            msg += f"   현재가: ${item['current_price']:.2f}\n"
            msg += f"   수익률: {item['return_pct']:.1f}%\n\n"
            
            # Holdings에서 제거
            sheets.sheets.remove_holding(item['symbol'])
            
            # Trade 기록
            sheets.save_trade({
                'symbol': item['symbol'],
                'action': 'STOP_LOSS',
                'shares': item['shares'],
                'price': item['current_price'],
                'amount': item['shares'] * item['current_price'],
                'return_pct': item['return_pct']
            })
        
        # Telegram 전송
        send_message(msg)
        
        # 손절 후 Holdings 다시 로드
        holdings = sheets.get_holdings()
    else:
        print("\n✅ 손절 대상 없음")
    
    # 5. Daily_Value 저장
    # 현금 계산 (간단히 총 자본금 - 주식가치로 추정)
    stocks_value = sum(
        holdings.get(s, {}).get('shares', 0) * current_prices.get(s, 0)
        for s in holdings
    )
    cash = total_capital - stocks_value
    
    # 음수면 0으로
    if cash < 0:
        cash = 0
    
    sheets.save_daily_value(holdings, current_prices, cash, spy_price)
    
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