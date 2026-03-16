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
from telegram import (
    send_hybrid_signal,
    send_hybrid_portfolio,
    send_hybrid_rebalancing,
    send_stop_loss,
    send_daily_summary,
    send_error
)


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
        Trades 시트 기반으로 계산 (모멘텀과 동일)

        Returns:
            float: 현금 잔고
        """
        if not self.sheets:
            return INITIAL_CAPITAL

        try:
            sync_result = self.sheets.sync_holdings_from_trades()
            cash = sync_result.get("cash", INITIAL_CAPITAL)
            print(f"💰 현재 현금: ${cash:,.2f}")
            return cash
        except Exception as e:
            print(f"⚠️ Cash 로드 실패: {e}")
            return INITIAL_CAPITAL
    
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
    
    def sync_holdings(self):
        """
        Trades 시트 기반으로 Holdings 동기화 (모멘텀과 동일)

        Returns:
            dict: {holdings: {...}, cash: float}
        """
        if not self.sheets:
            return {"holdings": {}, "cash": INITIAL_CAPITAL}

        try:
            sync_result = self.sheets.sync_holdings_from_trades()
            print("✅ Holdings 동기화 완료")
            return sync_result
        except Exception as e:
            print(f"⚠️ Holdings 동기화 실패: {e}")
            return {"holdings": {}, "cash": INITIAL_CAPITAL}
    
    def save_trade(self, action, memo="Hybrid"):
        """
        거래 기록 저장 (Trades 시트에 저장 - 모멘텀과 동일)

        Args:
            action: 거래 액션 dict
            memo: 메모
        """
        if not self.sheets:
            print(f"📋 [Sheets 없음] Trade: {action['action']} {action['symbol']} "
                  f"{action['shares']}주 @ ${round(action['price'], 2):.2f}")
            return

        try:
            self.sheets.save_trades([{
                "date": datetime.now().strftime("%Y-%m-%d"),
                "symbol": action.get("symbol", ""),
                "action": action.get("action", ""),
                "shares": action.get("shares", 0),
                "price": action.get("price", 0),
                "amount": action.get("amount", 0),
                "return_pct": action.get("return_pct", 0),
                "realized_pnl": action.get("realized_pnl", 0),
                "sector": action.get("sector", ""),
                "memo": memo
            }])
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
            # 시장 필터링 발동 체크 (HOLD 신호)
            if signal.get('market_filter', False):
                market_momentum = signal.get('market_momentum', 0)
                self.sheets.save_signal({
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M'),
                    'analysis_date': datetime.now().strftime('%Y-%m-%d'),
                    'signal': 'HOLD',
                    'picks': '없음 (시장 하락)',
                    'scores': '',
                    'allocations': '',
                    'market_momentum': round(market_momentum, 4),
                    'spy_price': round(signal.get('prices', {}).get('SPY', 0), 2),
                    'market_trend': 'DOWN'
                })
                print("✅ Signal 저장 완료 (HOLD)")
                return
            
            # 빈 signal 체크
            if not signal.get('picks'):
                print("⚠️ Signal 저장 스킵: 선정 종목 없음")
                return
            
            # scores를 문자열로 변환
            scores_str = ', '.join([str(round(s, 4)) for s in signal['scores']])
            allocs_str = ', '.join([str(int(a*100)) + '%' for a in signal['allocations']])
            market_momentum = signal.get('market_momentum', 0)
            
            self.sheets.save_signal({
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M'),
                'analysis_date': datetime.now().strftime('%Y-%m-%d'),
                'signal': 'BUY',
                'picks': ', '.join(signal['picks']),
                'scores': scores_str,
                'allocations': allocs_str,
                'market_momentum': round(market_momentum, 4),
                'spy_price': round(signal.get('prices', {}).get('SPY', 0), 2),
                'market_trend': 'UP'
            })
            print("✅ Signal 저장 완료")
        except Exception as e:
            print(f"⚠️ Signal 저장 실패: {e}")
    
    def save_daily_value(self, holdings, current_prices, cash, spy_price=0):
        """
        Daily_Value 시트에 일일 포트폴리오 가치 기록 (모멘텀과 동일)

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
            if holdings:
                for symbol, info in holdings.items():
                    shares = info.get('shares', 0)
                    price = current_prices.get(symbol, info.get('avg_price', 0))
                    stocks_value += shares * price

            total_value = stocks_value + cash

            # 이전 데이터에서 수익률 계산
            daily_df = self.sheets.load_daily_values()
            prev_value = INITIAL_CAPITAL
            prev_spy = spy_price
            if len(daily_df) > 0:
                try:
                    prev_value = float(daily_df.iloc[-1]["Total_Value"])
                    prev_spy = float(daily_df.iloc[-1]["SPY_Price"])
                except:
                    pass

            daily_return = (total_value - prev_value) / prev_value * 100 if prev_value > 0 else 0
            spy_return = (spy_price - prev_spy) / prev_spy * 100 if prev_spy > 0 else 0
            alpha = daily_return - spy_return

            daily_data = {
                "date": datetime.now().strftime("%Y-%m-%d"),
                "total_value": round(total_value, 2),
                "cash": round(cash, 2),
                "stocks_value": round(stocks_value, 2),
                "daily_return_pct": round(daily_return, 2),
                "spy_price": round(spy_price, 2),
                "spy_return_pct": round(spy_return, 2),
                "alpha": round(alpha, 2)
            }

            self.sheets.save_daily_value(daily_data)
            print(f"Daily value recorded: ${total_value:,.2f} ({daily_return:+.2f}%)")

        except Exception as e:
            print(f"⚠️ Daily_Value 저장 실패: {e}")

    def update_performance(self, exchange_rate=1400):
        """
        전체 성과 업데이트 (모멘텀과 동일)
        - Performance 시트 갱신
        - 월간/연간 리포트 자동 업데이트

        Args:
            exchange_rate: 원/달러 환율 (세금 계산용)
        """
        if not self.sheets:
            return

        try:
            import numpy as np

            sync_result = self.sheets.sync_holdings_from_trades()
            holdings = self.get_holdings()

            # 포트폴리오 가치 계산
            stocks_value = 0
            if holdings:
                import yfinance as yf
                for symbol, info in holdings.items():
                    try:
                        ticker = yf.Ticker(symbol)
                        hist = ticker.history(period='1d')
                        price = float(hist['Close'].iloc[-1]) if not hist.empty else info['avg_price']
                    except:
                        price = info['avg_price']
                    stocks_value += info['shares'] * price

            cash = sync_result.get("cash", INITIAL_CAPITAL)
            total_value = stocks_value + cash
            total_return = (total_value - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100

            trades_df = self.sheets.load_trades()
            daily_df = self.sheets.load_daily_values()

            # 기간 계산
            days = 0
            start_date = ""
            if len(daily_df) > 0:
                try:
                    start_date = daily_df.iloc[0]["Date"]
                    days = (datetime.now() - datetime.strptime(start_date, "%Y-%m-%d")).days
                except:
                    pass

            years = days / 365 if days > 0 else 0
            cagr = ((total_value / INITIAL_CAPITAL) ** (1 / years) - 1) * 100 if years > 0 else 0

            # SPY 수익률
            spy_return = 0
            if len(daily_df) > 1:
                try:
                    first_spy = float(daily_df.iloc[0]["SPY_Price"])
                    last_spy = float(daily_df.iloc[-1]["SPY_Price"])
                    spy_return = (last_spy - first_spy) / first_spy * 100
                except:
                    pass

            # MDD
            mdd = 0
            if len(daily_df) > 0:
                try:
                    values = daily_df["Total_Value"].astype(float)
                    peak = values.cummax()
                    drawdown = (values - peak) / peak * 100
                    mdd = drawdown.min()
                except:
                    pass

            # 승률 및 수수료/실현손익
            win_rate = 0
            total_trades = len(trades_df)
            total_commission = 0
            total_realized_pnl = 0

            if total_trades > 0:
                try:
                    returns = trades_df["Return%"].astype(float)
                    wins = (returns > 0).sum()
                    win_rate = wins / total_trades * 100
                    if "Commission" in trades_df.columns:
                        total_commission = trades_df["Commission"].astype(float).sum()
                    if "Realized_PnL" in trades_df.columns:
                        total_realized_pnl = trades_df["Realized_PnL"].astype(float).sum()
                except:
                    pass

            # Sharpe
            sharpe = 0
            if len(daily_df) > 1:
                try:
                    daily_returns = daily_df["Daily_Return%"].astype(float)
                    if daily_returns.std() > 0:
                        sharpe = (daily_returns.mean() * 252) / (daily_returns.std() * np.sqrt(252))
                except:
                    pass

            # 세금 계산 (해외주식 양도소득세: 22%, 250만원 공제)
            realized_pnl_krw = total_realized_pnl * exchange_rate
            taxable_amount = max(0, realized_pnl_krw - 2500000)
            est_tax = round(taxable_amount * 0.22)

            metrics = {
                "initial_capital": INITIAL_CAPITAL,
                "current_value": total_value,
                "total_return_pct": total_return,
                "cagr": cagr,
                "spy_return_pct": spy_return,
                "alpha": total_return - spy_return,
                "mdd": mdd,
                "sharpe_ratio": sharpe,
                "win_rate": win_rate,
                "total_trades": total_trades,
                "start_date": start_date,
                "days": days,
                "total_commission": round(total_commission, 2),
                "total_realized_pnl": round(total_realized_pnl, 2),
                "est_tax": est_tax
            }

            self.sheets.save_performance(metrics)
            self.sheets.update_monthly_summary()
            self.sheets.update_yearly_summary(exchange_rate)

            print("=" * 60)
            print("Performance Updated")
            print("=" * 60)
            print(f"Total Return: {total_return:+.2f}%")
            print(f"SPY Return: {spy_return:+.2f}%")
            print(f"Alpha: {total_return - spy_return:+.2f}%")
            print(f"MDD: {mdd:.2f}%")
            print(f"Win Rate: {win_rate:.1f}%")
            print("=" * 60)

        except Exception as e:
            print(f"⚠️ Performance 업데이트 실패: {e}")


# ============================================
# [2] Hybrid 전략 클래스 (간소화 버전)
# ============================================

# SPY 상관관계 필터 설정 (모멘텀과 동일)
CORRELATION_PERIOD = 60      # 상관관계 계산 기간 (60일)
CORRELATION_THRESHOLD = 0.5  # 최소 상관관계

class HybridTradingStrategy:
    """
    하이브리드 트레이딩 전략
    모멘텀 점수 + AI 확률 결합
    
    필터링 (모멘텀과 동일):
    - 시장 필터: 평균 1개월 수익률 <= 0 → HOLD
    - SPY 상관관계 > 0.5 필터
    """
    
    def __init__(self, weight_momentum=WEIGHT_MOMENTUM, weight_ai=WEIGHT_AI,
                 use_market_filter=True, correlation_threshold=CORRELATION_THRESHOLD):
        self.weight_m = weight_momentum
        self.weight_ai = weight_ai
        self.use_market_filter = use_market_filter
        self.correlation_threshold = correlation_threshold
        
        self.ai_strategy = None
        self.momentum_strategy = None
        self.score_df = None
        self.correlation_df = None  # SPY 상관관계
        self.ret_1m = None          # 1개월 수익률 (시장 필터용)
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
        
        # 모멘텀 준비 (상관관계, 수익률 포함)
        print("\n[2] 모멘텀 점수 계산...")
        self.momentum_strategy = CustomStrategy(
            correlation_threshold=self.correlation_threshold
        )
        tuesday_df = filter_tuesday(price_df)
        self.score_df, self.correlation_df, self.ret_1m = self.momentum_strategy.prepare(price_df, tuesday_df)
        
        self.is_prepared = True
        print("\n✅ Hybrid 전략 준비 완료!")
        print(f"   시장 필터링: {'ON (평균 1개월 수익률 <= 0 → HOLD)' if self.use_market_filter else 'OFF'}")
        print(f"   SPY 상관관계 필터: > {self.correlation_threshold}")
    
    def check_market_condition(self, date):
        """
        시장 상황 체크: 평균 1개월 수익률 (모멘텀과 동일)
        
        Args:
            date: 체크할 날짜
        
        Returns:
            tuple: (매수가능 여부, market_momentum)
        """
        if not self.use_market_filter:
            return True, 0
        
        if self.ret_1m is None or self.ret_1m.empty:
            return True, 0
        
        date_ts = pd.Timestamp(date)
        
        # 해당 날짜 또는 가장 가까운 이전 날짜
        if date_ts in self.ret_1m.index:
            market_momentum = self.ret_1m.loc[date_ts].mean()
        else:
            available_dates = self.ret_1m.index[self.ret_1m.index <= date_ts]
            if len(available_dates) == 0:
                return True, 0
            market_momentum = self.ret_1m.loc[available_dates[-1]].mean()
        
        # 평균 1개월 수익률 > 0 이면 매수 가능 (모멘텀과 동일)
        is_bullish = market_momentum > 0
        
        return is_bullish, market_momentum
    
    def get_high_correlation_stocks(self, date):
        """
        SPY와 상관관계 높은 종목 필터 (모멘텀과 동일)
        
        Args:
            date: 기준 날짜
        
        Returns:
            list: 상관관계 높은 종목 리스트
        """
        if self.correlation_df is None or self.correlation_df.empty:
            return None
        
        date_ts = pd.Timestamp(date)
        
        # 해당 날짜 또는 가장 가까운 이전 날짜
        if date_ts in self.correlation_df.index:
            corr_row = self.correlation_df.loc[date_ts]
        else:
            available_dates = self.correlation_df.index[self.correlation_df.index <= date_ts]
            if len(available_dates) == 0:
                return None
            corr_row = self.correlation_df.loc[available_dates[-1]]
        
        # 상관관계 > threshold 인 종목
        high_corr = corr_row[corr_row > self.correlation_threshold].index.tolist()
        
        # SPY 제외
        if 'SPY' in high_corr:
            high_corr.remove('SPY')
        
        return high_corr if high_corr else None
    
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
        
        # ----- 시장 필터링 체크 (모멘텀과 동일) -----
        is_bullish, market_momentum = self.check_market_condition(date)
        
        if not is_bullish:
            print(f"⚠️ 시장 필터링 발동: 평균 1개월 수익률 = {market_momentum:.4f} <= 0")
            print("   → HOLD (기존 보유 유지)")
            return {
                'picks': [],
                'scores': [],
                'allocations': [],
                'prices': {},
                'market_filter': True,
                'market_momentum': market_momentum,
                'signal': 'HOLD'
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
        
        # ----- SPY 상관관계 필터 (모멘텀과 동일) -----
        high_corr_stocks = self.get_high_correlation_stocks(date_ts_momentum)
        
        if high_corr_stocks:
            m_scores = m_scores[m_scores.index.isin(high_corr_stocks)]
            print(f"   SPY 상관관계 > {self.correlation_threshold}: {len(high_corr_stocks)}개 종목")
        
        if m_scores.empty:
            print("⚠️ 상관관계 필터 후 종목 없음")
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
        
        # 가격 dict (SPY 포함)
        prices = dict(zip(top_picks['symbol'], top_picks['close']))

        # SPY 가격 추가 (Daily_Value 및 텔레그램 수익률 계산용)
        if 'SPY' in price_df.columns:
            available_spy_dates = price_df.index[price_df.index <= date_ts]
            if len(available_spy_dates) > 0:
                prices['SPY'] = float(price_df['SPY'].loc[available_spy_dates[-1]])

        return {
            'picks': top_picks['symbol'].tolist(),
            'scores': top_picks['hybrid_score'].tolist(),
            'allocations': allocations[:n_picks],
            'prices': prices,
            'market_filter': False,
            'market_momentum': market_momentum,
            'signal': 'BUY'
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
        print(f"   평균 1개월 수익률: {result.get('market_momentum', 0):.4f} <= 0")
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
# [8] 메인 실행
# ============================================

def run_hybrid_weekly():
    """
    Hybrid 주간 실행 (모멘텀과 동일한 구조)
    - Holdings 동기화 (Trades 기반)
    - 신호 생성
    - 리밸런싱 계산 및 Telegram 알림
    - Trades 시트에 거래 기록
    - Daily_Value / Performance 업데이트
    """
    print("=" * 60)
    print("🤖 Hybrid 주간 실행")
    print("=" * 60)
    print(f"가중치: 모멘텀 {WEIGHT_MOMENTUM*100:.0f}% + AI {WEIGHT_AI*100:.0f}%")

    # 1. Sheets 연결
    sheets = HybridSheetsManager()

    # 2. Holdings 동기화 (Trades 기반) + Cash 계산
    sync_result = sheets.sync_holdings()

    # 3. 신호 생성
    signal = get_hybrid_signal()

    if signal is None:
        print("❌ 신호 생성 실패")
        return

    # 4. 시장 필터링 체크 (모멘텀과 동일: HOLD)
    if signal.get('market_filter', False):
        print("\n⚠️ 시장 필터링 발동 - HOLD (기존 보유 유지)")

        # Telegram 전송
        send_hybrid_signal(signal, INITIAL_CAPITAL, WEIGHT_MOMENTUM, WEIGHT_AI)

        # Signal 저장
        sheets.save_signal(signal)

        # Daily_Value 저장 (기존 보유 기준)
        portfolio = sheets.get_holdings()
        cash = sync_result.get("cash", INITIAL_CAPITAL)
        stocks_value = 0.0
        holdings_detail = []
        if portfolio:
            import yfinance as yf
            symbols = list(portfolio.keys()) + ['SPY']
            current_prices = {}
            for sym in symbols:
                try:
                    ticker = yf.Ticker(sym)
                    hist = ticker.history(period='1d')
                    if not hist.empty:
                        current_prices[sym] = hist['Close'].iloc[-1]
                except:
                    pass
            spy_price = current_prices.get('SPY', 0)
            sheets.save_daily_value(portfolio, current_prices, cash, spy_price)

            for sym, data in portfolio.items():
                shares = data.get('shares', 0)
                avg_price = data.get('avg_price', 0)
                cur_price = current_prices.get(sym, avg_price)
                stocks_value += shares * cur_price
                pnl_pct = ((cur_price - avg_price) / avg_price * 100) if avg_price > 0 else 0.0
                holdings_detail.append({
                    "symbol": sym,
                    "shares": shares,
                    "profit_loss_pct": pnl_pct
                })

        # Telegram 전송 2: 포트폴리오
        send_hybrid_portfolio({
            "total": cash + stocks_value,
            "cash": cash,
            "stocks": stocks_value,
            "holdings_detail": holdings_detail
        })

        # Performance 업데이트
        sheets.update_performance()

        print("\n✅ Hybrid 주간 실행 완료 (HOLD)")
        return {'signal': signal, 'market_filter': True}

    # 5. 현재 포트폴리오 (Trades 기반 동기화 결과 사용)
    portfolio = sheets.get_holdings()

    # 현재 가격 추가
    for symbol in portfolio:
        if symbol in signal['prices']:
            portfolio[symbol]['current_price'] = signal['prices'][symbol]
        else:
            portfolio[symbol]['current_price'] = portfolio[symbol]['avg_price']

    print(f"📊 현재 보유: {list(portfolio.keys()) if portfolio else '없음'}")

    # 6. 동적 자본금 계산 (현금 + 주식가치)
    available_cash = sync_result.get("cash", INITIAL_CAPITAL)
    stocks_value = sum(
        info.get('shares', 0) * info.get('current_price', info.get('avg_price', 0))
        for info in portfolio.values()
    ) if portfolio else 0

    total_capital = available_cash + stocks_value
    print(f"💰 동적 자본금: ${total_capital:,.2f} (현금 ${available_cash:,.2f} + 주식 ${stocks_value:,.2f})")

    # 7. 리밸런싱 계산 (동적 자본금 사용)
    rebalancing = calculate_hybrid_rebalancing(portfolio, signal, total_capital, available_cash)

    # 8. 출력
    print_hybrid_rebalancing(rebalancing)

    # 9. Telegram 전송 (signal 포함)
    send_hybrid_rebalancing(rebalancing, total_capital, signal, WEIGHT_MOMENTUM, WEIGHT_AI)

    # 10. Sheets 기록
    # 신호 저장
    sheets.save_signal(signal)

    # 거래 Trades 시트에 저장 → 수동 입력 (모멘텀과 동일)
    # 리밸런싱 안내는 Telegram으로 발송, 실제 매매 후 Trades에 직접 입력

    # 11. Holdings 동기화 (Trades 기반 재계산)
    sync_result = sheets.sync_holdings()
    cash = sync_result.get("cash", INITIAL_CAPITAL)

    # 12. Daily_Value 저장
    spy_price = signal['prices'].get('SPY', 0)
    new_holdings = sheets.get_holdings()
    sheets.save_daily_value(new_holdings, signal['prices'], cash, spy_price)

    # 13. Performance 업데이트 (월간/연간 포함)
    sheets.update_performance()

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


def run_hybrid_daily():
    """
    Hybrid Daily 실행 (모멘텀과 동일한 구조)
    - Holdings 동기화 (Trades 기반)
    - 손절 체크 (알림 + Trades 기록)
    - 일일 가치 기록
    - Performance 업데이트
    """
    print("=" * 60)
    print("🤖 Hybrid Daily 실행")
    print("=" * 60)

    today = datetime.now().strftime('%Y-%m-%d')

    # 1. Sheets 연결
    sheets = HybridSheetsManager()

    # 2. Holdings 동기화 (Trades 기반) + Cash 계산
    sync_result = sheets.sync_holdings()

    # 3. 현재 보유 종목 가져오기
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
            for item in stop_loss_list:
                print(f"  • {item['symbol']}: {item['return_pct']:.1f}%")
                item['profit_loss'] = item['shares'] * (item['current_price'] - item['avg_price'])

            # Telegram 전송 (알림만 - 실제 매도는 수동, 모멘텀과 동일)
            send_stop_loss(stop_loss_list)
            print("→ 한투 앱에서 수동 매도 후 Trades에 직접 입력 필요!")
        else:
            print("\n✅ 손절 대상 없음")
    else:
        print("📊 보유 종목 없음 (현금 보유 중)")

    # 6. 현재 현금 및 포트폴리오 가치 계산
    cash = sync_result.get("cash", INITIAL_CAPITAL)
    stocks_value = sum(
        holdings.get(s, {}).get('shares', 0) * current_prices.get(s, 0)
        for s in holdings
    ) if holdings else 0
    total_value = stocks_value + cash
    print(f"💰 포트폴리오: ${total_value:,.2f} (현금 ${cash:,.2f} + 주식 ${stocks_value:,.2f})")

    # 7. Daily_Value 저장
    sheets.save_daily_value(holdings, current_prices, cash, spy_price)

    # 8. Performance 업데이트 (월간/연간 포함)
    sheets.update_performance()

    # 9. Daily Summary 텔레그램 전송
    daily_df = sheets.sheets.load_daily_values()
    daily_return = 0
    spy_return = 0
    alpha = 0
    if len(daily_df) > 1:
        try:
            daily_return = float(daily_df.iloc[-1]["Daily_Return%"])
            spy_return = float(daily_df.iloc[-1]["SPY_Return%"])
            alpha = float(daily_df.iloc[-1]["Alpha"])
        except:
            pass

    holdings_detail = []
    for symbol, info in holdings.items():
        avg_price = info.get('avg_price', 0)
        current_price = current_prices.get(symbol, avg_price)
        return_pct = (current_price - avg_price) / avg_price * 100 if avg_price > 0 else 0
        holdings_detail.append({
            'symbol': symbol,
            'shares': info.get('shares', 0),
            'profit_loss_pct': return_pct
        })

    daily_data = {
        'date': today,
        'daily_return_pct': daily_return,
        'spy_return_pct': spy_return,
        'alpha': alpha
    }
    portfolio_value = {
        'total': total_value,
        'cash': cash,
        'stocks': stocks_value,
        'holdings_detail': holdings_detail
    }
    ref_signal = get_hybrid_signal()
    if ref_signal:
        is_hold = ref_signal.get('market_filter', False) or not ref_signal.get('picks')
        ref_signal['signal'] = "HOLD" if is_hold else "BUY"
        ref_signal['market_trend'] = "DOWN" if is_hold else "UP"
        if 'date' not in ref_signal:
            ref_signal['date'] = today
    send_daily_summary(daily_data, portfolio_value, signal=ref_signal)

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
            send_hybrid_signal(signal, INITIAL_CAPITAL, WEIGHT_MOMENTUM, WEIGHT_AI)
    except Exception as e:
        print(f"❌ 에러: {e}")