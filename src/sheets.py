# ============================================
# src/sheets.py
# Google Sheets Connection Module (v2)
# 7 Sheets: Portfolio, Trades, Signals,
#           Daily_Value, Monthly_Value, Yearly_Value, Performance
# ============================================

import gspread
from google.oauth2.service_account import Credentials
from datetime import datetime
import pandas as pd
import numpy as np
import os
import json

# ============================================
# Settings
# ============================================

SPREADSHEET_NAME = "Stock_Paper_Trading"
SERVICE_ACCOUNT_FILE = "service_account.json"

# Sheet Names
SHEET_HOLDINGS = "Holdings"
SHEET_TRADES = "Trades"
SHEET_SIGNALS = "Signals"
SHEET_DAILY = "Daily_Value"
SHEET_MONTHLY = "Monthly_Value"
SHEET_YEARLY = "Yearly_Value"
SHEET_PERFORMANCE = "Performance"

SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive"
]

# Headers
HEADERS = {
    SHEET_HOLDINGS: ["Symbol", "Shares", "Avg_Price", "Sector", "Buy_Date"],
    SHEET_TRADES: ["Date", "Symbol", "Action", "Shares", "Price", "Amount", "Commission", "Return%", "Realized_PnL", "Sector", "Memo"],
    SHEET_SIGNALS: ["Timestamp", "Analysis_Date", "Signal", "Picks", "Scores", "Allocations", "Market_Momentum", "SPY_Price", "Market_Trend"],
    SHEET_DAILY: ["Date", "Total_Value", "Cash", "Stocks_Value", "Daily_Return%", "SPY_Price", "SPY_Return%", "Alpha"],
    SHEET_MONTHLY: ["Year_Month", "Start_Value", "End_Value", "Monthly_Return%", "SPY_Return%", "Alpha", "Trades", "Commission", "Realized_PnL"],
    SHEET_YEARLY: ["Year", "Start_Value", "End_Value", "Yearly_Return%", "SPY_Return%", "Alpha", "Total_Trades", "Win_Rate", "Total_Commission", "Total_Realized_PnL", "Est_Tax"],
    SHEET_PERFORMANCE: ["Metric", "Value"]
}

# 수수료율 (0.1%)
from config import BUY_COMMISSION
COMMISSION_RATE = BUY_COMMISSION

# 해외주식 양도소득세 (22% - 기본공제 250만원)
TAX_RATE = 0.22
TAX_EXEMPTION = 2500000  # 원


# ============================================
# Utility: numpy/pandas → Python 타입 변환
# ============================================

def to_python(val):
    """
    numpy/pandas 타입을 Python 기본 타입으로 변환
    Google Sheets API는 JSON 직렬화가 필요하므로 numpy 타입 사용 불가
    """
    if val is None:
        return ""
    if isinstance(val, (np.integer, np.int64, np.int32)):
        return int(val)
    if isinstance(val, (np.floating, np.float64, np.float32)):
        return float(val)
    if isinstance(val, np.ndarray):
        return val.tolist()
    if hasattr(val, 'item'):
        return val.item()
    return val


# ============================================
# SheetsManager Class
# ============================================

class SheetsManager:
    
    def __init__(self, spreadsheet_name=SPREADSHEET_NAME):
        self.spreadsheet_name = spreadsheet_name
        self.gc = None
        self.spreadsheet = None
        self._connect()
    
    
    # ============================================
    # Connection
    # ============================================
    
    def _connect(self):
        # Method 1: Service Account File
        if os.path.exists(SERVICE_ACCOUNT_FILE):
            creds = Credentials.from_service_account_file(SERVICE_ACCOUNT_FILE, scopes=SCOPES)
            self.gc = gspread.authorize(creds)
            print("Connected via service account file")
        
        # Method 2: Environment Variable (GitHub Actions)
        elif os.environ.get("GOOGLE_CREDENTIALS"):
            creds_json = os.environ.get("GOOGLE_CREDENTIALS")
            creds_dict = json.loads(creds_json)
            creds = Credentials.from_service_account_info(creds_dict, scopes=SCOPES)
            self.gc = gspread.authorize(creds)
            print("Connected via environment variable")
        
        # Method 3: Colab Auth
        else:
            try:
                from google.colab import auth
                from google.auth import default
                auth.authenticate_user()
                creds, _ = default()
                self.gc = gspread.authorize(creds)
                print("Connected via Colab auth")
            except:
                raise Exception("No valid credentials found!")
        
        # Open Spreadsheet
        try:
            self.spreadsheet = self.gc.open(self.spreadsheet_name)
            print(f"Opened: {self.spreadsheet_name}")
        except gspread.SpreadsheetNotFound:
            print("Spreadsheet not found. Creating...")
            self._create_spreadsheet()
    
    
    def _create_spreadsheet(self):
        self.spreadsheet = self.gc.create(self.spreadsheet_name)
        
        # Create all sheets
        for sheet_name in HEADERS.keys():
            self.spreadsheet.add_worksheet(title=sheet_name, rows=5000, cols=20)
        
        # Delete default Sheet1
        try:
            self.spreadsheet.del_worksheet(self.spreadsheet.sheet1)
        except:
            pass
        
        # Initialize headers
        self._init_all_headers()
        
        print(f"Created: {self.spreadsheet_name}")
        print(f"URL: {self.spreadsheet.url}")
    
    
    def _init_all_headers(self):
        for sheet_name, headers in HEADERS.items():
            try:
                ws = self.spreadsheet.worksheet(sheet_name)
                ws.update("A1", [headers])
            except:
                pass
    
    
    def _get_worksheet(self, sheet_name):
        try:
            return self.spreadsheet.worksheet(sheet_name)
        except gspread.WorksheetNotFound:
            ws = self.spreadsheet.add_worksheet(title=sheet_name, rows=5000, cols=20)
            if sheet_name in HEADERS:
                ws.update("A1", [HEADERS[sheet_name]])
            return ws
    
    
    def init_sheets(self):
        """Initialize all sheets with headers (run once)"""
        print("Initializing sheets...")
        for sheet_name, headers in HEADERS.items():
            ws = self._get_worksheet(sheet_name)
            existing = ws.get_all_values()
            if len(existing) == 0:
                ws.update("A1", [headers])
                print(f"  {sheet_name}: headers added")
            else:
                print(f"  {sheet_name}: already exists")
        print("Done!")
    
    
    # ============================================
    # Holdings (수동 입력 시트)
    # ============================================
    
    def load_holdings(self):
        """
        Holdings 시트에서 보유 종목 로드
        (수동으로 입력한 데이터)
        
        Returns:
            DataFrame: Symbol, Shares, Avg_Price, Sector, Buy_Date
        """
        ws = self._get_worksheet(SHEET_HOLDINGS)
        data = ws.get_all_values()
        
        if len(data) <= 1:
            return pd.DataFrame(columns=HEADERS[SHEET_HOLDINGS])
        
        df = pd.DataFrame(data[1:], columns=data[0])
        
        # 빈 행 제거
        df = df[df["Symbol"].str.strip() != ""]
        
        print(f"Holdings loaded ({len(df)} stocks)")
        return df
    
    
    def save_holding(self, holding_data):
        """
        단일 보유 종목 추가
        
        Args:
            holding_data: {symbol, shares, avg_price, sector, buy_date}
        """
        ws = self._get_worksheet(SHEET_HOLDINGS)
        
        # Check header
        existing = ws.get_all_values()
        if len(existing) == 0:
            ws.append_row(HEADERS[SHEET_HOLDINGS])
        
        ws.append_row([
            str(holding_data.get("symbol", "")),
            to_python(holding_data.get("shares", 0)),
            to_python(holding_data.get("avg_price", 0)),
            str(holding_data.get("sector", "")),
            str(holding_data.get("buy_date", ""))
        ])
        
        print(f"Holding saved: {holding_data.get('symbol', '')}")
    
    
    def remove_holding(self, symbol):
        """
        보유 종목 제거 (매도 시)
        
        Args:
            symbol: 종목 코드
        """
        ws = self._get_worksheet(SHEET_HOLDINGS)
        data = ws.get_all_values()
        
        if len(data) <= 1:
            return
        
        # 해당 종목 찾기
        for i, row in enumerate(data):
            if row[0] == symbol:
                ws.delete_rows(i + 1)  # 1-indexed
                print(f"Holding removed: {symbol}")
                return
        
        print(f"Holding not found: {symbol}")
    
    
    def update_holding(self, symbol, shares=None, avg_price=None):
        """
        보유 종목 수량/가격 업데이트
        
        Args:
            symbol: 종목 코드
            shares: 새 수량 (None이면 변경 안 함)
            avg_price: 새 평균 단가 (None이면 변경 안 함)
        """
        ws = self._get_worksheet(SHEET_HOLDINGS)
        data = ws.get_all_values()
        
        if len(data) <= 1:
            return
        
        # 해당 종목 찾기
        for i, row in enumerate(data):
            if row[0] == symbol:
                if shares is not None:
                    ws.update_cell(i + 1, 2, shares)  # B열
                if avg_price is not None:
                    ws.update_cell(i + 1, 3, avg_price)  # C열
                print(f"Holding updated: {symbol}")
                return
        
        print(f"Holding not found: {symbol}")
    
    
    def sync_holdings_from_trades(self, initial_capital=None):
        """
        Trades 시트 기반으로 Holdings 시트 자동 동기화 + Cash 계산
        
        로직:
        - BUY: 보유 추가, Cash 차감
        - SELL/STOP_LOSS: 보유 차감, Cash 증가
        
        Args:
            initial_capital: 초기 자본금 (None이면 config에서 가져옴)
        
        Returns:
            dict: {holdings: {...}, cash: float}
        """
        # 초기 자본금
        if initial_capital is None:
            try:
                from config import INITIAL_CAPITAL
                initial_capital = INITIAL_CAPITAL
            except:
                initial_capital = 3000
        
        trades_df = self.load_trades()
        
        if trades_df.empty:
            print("No trades to sync")
            return {"holdings": {}, "cash": initial_capital}
        
        # 보유 종목 및 현금 계산
        holdings = {}
        cash = initial_capital
        
        for _, row in trades_df.iterrows():
            symbol = row.get("Symbol", "")
            action = row.get("Action", "").upper()
            
            try:
                shares = int(float(row.get("Shares", 0)))
                price = float(row.get("Price", 0))
            except:
                continue
            
            if not symbol or shares <= 0:
                continue
            
            # 금액 계산
            amount = shares * price
            commission = amount * COMMISSION_RATE
            
            sector = row.get("Sector", "")
            date = row.get("Date", "")
            
            if action == "BUY":
                # 현금 차감
                cash -= (amount + commission)
                
                if symbol in holdings:
                    # 추가 매수: 평균단가 계산
                    old = holdings[symbol]
                    old_shares = old["shares"]
                    old_avg = old["avg_price"]
                    new_shares = old_shares + shares
                    new_avg = (old_avg * old_shares + price * shares) / new_shares
                    holdings[symbol] = {
                        "shares": new_shares,
                        "avg_price": round(new_avg, 2),
                        "sector": sector or old.get("sector", ""),
                        "buy_date": old.get("buy_date", date)
                    }
                else:
                    # 신규 매수
                    holdings[symbol] = {
                        "shares": shares,
                        "avg_price": round(price, 2),
                        "sector": sector,
                        "buy_date": date
                    }
            
            elif action in ["SELL", "STOP_LOSS"]:
                # 현금 증가
                cash += (amount - commission)
                
                if symbol in holdings:
                    holdings[symbol]["shares"] -= shares
                    if holdings[symbol]["shares"] <= 0:
                        del holdings[symbol]
        
        # Holdings 시트 업데이트
        ws = self._get_worksheet(SHEET_HOLDINGS)
        ws.clear()
        ws.append_row(HEADERS[SHEET_HOLDINGS])
        
        for symbol, info in holdings.items():
            ws.append_row([
                symbol,
                to_python(info["shares"]),
                to_python(info["avg_price"]),
                str(info.get("sector", "")),
                str(info.get("buy_date", ""))
            ])
        
        cash = round(cash, 2)
        print(f"Holdings synced ({len(holdings)} stocks, Cash: ${cash:,.2f})")
        
        return {"holdings": holdings, "cash": cash}


    # ============================================
    # Trades
    # ============================================
    
    def save_trades(self, trades):
        """
        Append trades to sheet (수수료 자동 계산)
        
        Args:
            trades: list of {date, symbol, action, shares, price, amount, return_pct, realized_pnl, sector, memo}
        """
        if not trades:
            return
        
        ws = self._get_worksheet(SHEET_TRADES)
        
        # Check header
        existing = ws.get_all_values()
        if len(existing) == 0:
            ws.append_row(HEADERS[SHEET_TRADES])
        
        # Append trades
        for t in trades:
            amount = to_python(t.get("amount", 0))
            commission = round(amount * COMMISSION_RATE, 2)  # 0.1% 수수료
            
            ws.append_row([
                str(t.get("date", "")),
                str(t.get("symbol", "")),
                str(t.get("action", "")),
                to_python(t.get("shares", 0)),
                to_python(t.get("price", 0)),
                amount,
                commission,
                to_python(t.get("return_pct", 0)),
                to_python(t.get("realized_pnl", 0)),
                str(t.get("sector", "")),
                str(t.get("memo", ""))
            ])
        
        print(f"Trades saved ({len(trades)} rows)")
    
    
    def load_trades(self):
        """Load all trades"""
        ws = self._get_worksheet(SHEET_TRADES)
        data = ws.get_all_values()
        
        if len(data) <= 1:
            return pd.DataFrame(columns=HEADERS[SHEET_TRADES])
        
        df = pd.DataFrame(data[1:], columns=data[0])
        print(f"Trades loaded ({len(df)} rows)")
        return df
    
    
    # ============================================
    # Signals
    # ============================================
    
    def save_signal(self, signal):
        """
        Append signal to sheet
        
        Args:
            signal: {date, signal, picks, scores, allocations, market_momentum, spy_price, market_trend}
        """
        ws = self._get_worksheet(SHEET_SIGNALS)
        
        # Check header
        existing = ws.get_all_values()
        if len(existing) == 0:
            ws.append_row(HEADERS[SHEET_SIGNALS])
        
        # Date format
        if hasattr(signal.get("date"), "strftime"):
            date_str = signal["date"].strftime("%Y-%m-%d")
        else:
            date_str = str(signal.get("date", ""))
        
        # scores 변환
        scores = signal.get("scores", [])
        scores_str = ", ".join([f"{to_python(s):.4f}" for s in scores])
        
        # allocations 변환
        allocations = signal.get("allocations", [])
        alloc_str = ", ".join([f"{to_python(a)*100:.0f}%" for a in allocations])
        
        ws.append_row([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            date_str,
            str(signal.get("signal", "")),
            ", ".join(signal.get("picks", [])),
            scores_str,
            alloc_str,
            to_python(signal.get("market_momentum", 0)),
            to_python(signal.get("spy_price", 0)),
            str(signal.get("market_trend", ""))
        ])
        
        print(f"Signal saved ({signal.get('signal', '')})")
    
    
    def load_signals(self):
        """Load all signals"""
        ws = self._get_worksheet(SHEET_SIGNALS)
        data = ws.get_all_values()
        
        if len(data) <= 1:
            return pd.DataFrame(columns=HEADERS[SHEET_SIGNALS])
        
        df = pd.DataFrame(data[1:], columns=data[0])
        print(f"Signals loaded ({len(df)} rows)")
        return df
    
    
    # ============================================
    # Daily Value
    # ============================================
    
    def save_daily_value(self, daily_data):
        """
        Append daily value to sheet
        
        Args:
            daily_data: {date, total_value, cash, stocks_value, daily_return_pct, spy_price, spy_return_pct, alpha}
        """
        ws = self._get_worksheet(SHEET_DAILY)
        
        # Check header
        existing = ws.get_all_values()
        if len(existing) == 0:
            ws.append_row(HEADERS[SHEET_DAILY])
        
        ws.append_row([
            str(daily_data.get("date", "")),
            to_python(daily_data.get("total_value", 0)),
            to_python(daily_data.get("cash", 0)),
            to_python(daily_data.get("stocks_value", 0)),
            to_python(daily_data.get("daily_return_pct", 0)),
            to_python(daily_data.get("spy_price", 0)),
            to_python(daily_data.get("spy_return_pct", 0)),
            to_python(daily_data.get("alpha", 0))
        ])
        
        print(f"Daily value saved ({daily_data.get('date', '')})")
    
    
    def load_daily_values(self):
        """Load all daily values"""
        ws = self._get_worksheet(SHEET_DAILY)
        data = ws.get_all_values()
        
        if len(data) <= 1:
            return pd.DataFrame(columns=HEADERS[SHEET_DAILY])
        
        df = pd.DataFrame(data[1:], columns=data[0])
        print(f"Daily values loaded ({len(df)} rows)")
        return df
    
    
    # ============================================
    # Monthly Value
    # ============================================
    
    def save_monthly_value(self, monthly_data):
        """
        Append monthly value to sheet
        
        Args:
            monthly_data: {year_month, start_value, end_value, monthly_return_pct, 
                          spy_return_pct, alpha, trades, commission, realized_pnl}
        """
        ws = self._get_worksheet(SHEET_MONTHLY)
        
        # Check header
        existing = ws.get_all_values()
        if len(existing) == 0:
            ws.append_row(HEADERS[SHEET_MONTHLY])
        
        ws.append_row([
            str(monthly_data.get("year_month", "")),
            to_python(monthly_data.get("start_value", 0)),
            to_python(monthly_data.get("end_value", 0)),
            to_python(monthly_data.get("monthly_return_pct", 0)),
            to_python(monthly_data.get("spy_return_pct", 0)),
            to_python(monthly_data.get("alpha", 0)),
            to_python(monthly_data.get("trades", 0)),
            to_python(monthly_data.get("commission", 0)),
            to_python(monthly_data.get("realized_pnl", 0))
        ])
        
        print(f"Monthly value saved ({monthly_data.get('year_month', '')})")
    
    
    def update_monthly_summary(self):
        """
        월간 리포트 자동 생성 (Daily_Value + Trades 기반)
        """
        daily_df = self.load_daily_values()
        trades_df = self.load_trades()
        
        if daily_df.empty:
            print("No daily data for monthly summary")
            return
        
        # 날짜 파싱
        daily_df["Date"] = pd.to_datetime(daily_df["Date"])
        daily_df["Year_Month"] = daily_df["Date"].dt.strftime("%Y-%m")
        
        if not trades_df.empty:
            trades_df["Date"] = pd.to_datetime(trades_df["Date"])
            trades_df["Year_Month"] = trades_df["Date"].dt.strftime("%Y-%m")
        
        # 기존 월간 데이터
        existing_months = []
        monthly_df = self.load_monthly_values()
        if not monthly_df.empty:
            existing_months = monthly_df["Year_Month"].tolist()
        
        # 월별 집계
        saved_count = 0
        for ym in daily_df["Year_Month"].unique():
            if ym in existing_months:
                continue
            
            month_daily = daily_df[daily_df["Year_Month"] == ym].copy()
            month_daily = month_daily.sort_values("Date")
            
            if len(month_daily) < 2:
                continue
            
            # 수익률 계산
            start_value = float(month_daily.iloc[0]["Total_Value"])
            end_value = float(month_daily.iloc[-1]["Total_Value"])
            monthly_return = (end_value - start_value) / start_value * 100 if start_value > 0 else 0
            
            # SPY 수익률
            start_spy = float(month_daily.iloc[0]["SPY_Price"])
            end_spy = float(month_daily.iloc[-1]["SPY_Price"])
            spy_return = (end_spy - start_spy) / start_spy * 100 if start_spy > 0 else 0
            
            # 거래 집계
            trades_count = 0
            commission_sum = 0
            realized_pnl_sum = 0
            
            if not trades_df.empty:
                month_trades = trades_df[trades_df["Year_Month"] == ym]
                trades_count = len(month_trades)
                try:
                    commission_sum = month_trades["Commission"].astype(float).sum()
                    realized_pnl_sum = month_trades["Realized_PnL"].astype(float).sum()
                except:
                    pass
            
            monthly_data = {
                "year_month": ym,
                "start_value": round(start_value, 2),
                "end_value": round(end_value, 2),
                "monthly_return_pct": round(monthly_return, 2),
                "spy_return_pct": round(spy_return, 2),
                "alpha": round(monthly_return - spy_return, 2),
                "trades": trades_count,
                "commission": round(commission_sum, 2),
                "realized_pnl": round(realized_pnl_sum, 2)
            }
            
            self.save_monthly_value(monthly_data)
            saved_count += 1
        
        if saved_count > 0:
            print(f"Monthly summary updated ({saved_count} months)")
        else:
            print("No new monthly data to save")
    
    
    def load_monthly_values(self):
        """Load all monthly values"""
        ws = self._get_worksheet(SHEET_MONTHLY)
        data = ws.get_all_values()
        
        if len(data) <= 1:
            return pd.DataFrame(columns=HEADERS[SHEET_MONTHLY])
        
        df = pd.DataFrame(data[1:], columns=data[0])
        print(f"Monthly values loaded ({len(df)} rows)")
        return df
    
    
    # ============================================
    # Yearly Value
    # ============================================
    
    def save_yearly_value(self, yearly_data):
        """
        Append yearly value to sheet
        
        Args:
            yearly_data: {year, start_value, end_value, yearly_return_pct, spy_return_pct, 
                         alpha, total_trades, win_rate, total_commission, total_realized_pnl, est_tax}
        """
        ws = self._get_worksheet(SHEET_YEARLY)
        
        # Check header
        existing = ws.get_all_values()
        if len(existing) == 0:
            ws.append_row(HEADERS[SHEET_YEARLY])
        
        ws.append_row([
            to_python(yearly_data.get("year", "")),
            to_python(yearly_data.get("start_value", 0)),
            to_python(yearly_data.get("end_value", 0)),
            to_python(yearly_data.get("yearly_return_pct", 0)),
            to_python(yearly_data.get("spy_return_pct", 0)),
            to_python(yearly_data.get("alpha", 0)),
            to_python(yearly_data.get("total_trades", 0)),
            to_python(yearly_data.get("win_rate", 0)),
            to_python(yearly_data.get("total_commission", 0)),
            to_python(yearly_data.get("total_realized_pnl", 0)),
            to_python(yearly_data.get("est_tax", 0))
        ])
        
        print(f"Yearly value saved ({yearly_data.get('year', '')})")
    
    
    def update_yearly_summary(self, exchange_rate=1400):
        """
        연간 리포트 자동 생성 (세금 계산 포함)
        
        Args:
            exchange_rate: 원/달러 환율 (세금 계산용)
        """
        daily_df = self.load_daily_values()
        trades_df = self.load_trades()
        
        if daily_df.empty:
            print("No daily data for yearly summary")
            return
        
        # 날짜 파싱
        daily_df["Date"] = pd.to_datetime(daily_df["Date"])
        daily_df["Year"] = daily_df["Date"].dt.year
        
        if not trades_df.empty:
            trades_df["Date"] = pd.to_datetime(trades_df["Date"])
            trades_df["Year"] = trades_df["Date"].dt.year
        
        # 기존 연간 데이터
        existing_years = []
        yearly_df = self.load_yearly_values()
        if not yearly_df.empty:
            existing_years = yearly_df["Year"].astype(str).tolist()
        
        # 연도별 집계
        saved_count = 0
        for year in daily_df["Year"].unique():
            if str(year) in existing_years:
                continue
            
            year_daily = daily_df[daily_df["Year"] == year].copy()
            year_daily = year_daily.sort_values("Date")
            
            if len(year_daily) < 2:
                continue
            
            # 수익률 계산
            start_value = float(year_daily.iloc[0]["Total_Value"])
            end_value = float(year_daily.iloc[-1]["Total_Value"])
            yearly_return = (end_value - start_value) / start_value * 100 if start_value > 0 else 0
            
            # SPY 수익률
            start_spy = float(year_daily.iloc[0]["SPY_Price"])
            end_spy = float(year_daily.iloc[-1]["SPY_Price"])
            spy_return = (end_spy - start_spy) / start_spy * 100 if start_spy > 0 else 0
            
            # 거래 집계
            total_trades = 0
            wins = 0
            total_commission = 0
            total_realized_pnl = 0
            
            if not trades_df.empty:
                year_trades = trades_df[trades_df["Year"] == year]
                total_trades = len(year_trades)
                
                try:
                    returns = year_trades["Return%"].astype(float)
                    wins = (returns > 0).sum()
                    total_commission = year_trades["Commission"].astype(float).sum()
                    total_realized_pnl = year_trades["Realized_PnL"].astype(float).sum()
                except:
                    pass
            
            win_rate = (wins / total_trades * 100) if total_trades > 0 else 0
            
            # 세금 계산 (해외주식 양도소득세)
            # 실현이익(원화) - 250만원 공제 후 22%
            realized_pnl_krw = total_realized_pnl * exchange_rate
            taxable_amount = max(0, realized_pnl_krw - TAX_EXEMPTION)
            est_tax = round(taxable_amount * TAX_RATE)
            
            yearly_data = {
                "year": year,
                "start_value": round(start_value, 2),
                "end_value": round(end_value, 2),
                "yearly_return_pct": round(yearly_return, 2),
                "spy_return_pct": round(spy_return, 2),
                "alpha": round(yearly_return - spy_return, 2),
                "total_trades": total_trades,
                "win_rate": round(win_rate, 1),
                "total_commission": round(total_commission, 2),
                "total_realized_pnl": round(total_realized_pnl, 2),
                "est_tax": est_tax
            }
            
            self.save_yearly_value(yearly_data)
            saved_count += 1
        
        if saved_count > 0:
            print(f"Yearly summary updated ({saved_count} years)")
        else:
            print("No new yearly data to save")
    
    
    def load_yearly_values(self):
        """Load all yearly values"""
        ws = self._get_worksheet(SHEET_YEARLY)
        data = ws.get_all_values()
        
        if len(data) <= 1:
            return pd.DataFrame(columns=HEADERS[SHEET_YEARLY])
        
        df = pd.DataFrame(data[1:], columns=data[0])
        print(f"Yearly values loaded ({len(df)} rows)")
        return df
    
    
    # ============================================
    # Performance
    # ============================================
    
    def save_performance(self, metrics):
        """
        Save performance metrics (overwrite)
        
        Args:
            metrics: {
                initial_capital, current_value, total_return_pct, cagr,
                spy_return_pct, alpha, mdd, sharpe_ratio, win_rate,
                total_trades, start_date, days,
                total_commission, total_realized_pnl, est_tax
            }
        """
        ws = self._get_worksheet(SHEET_PERFORMANCE)
        ws.clear()
        
        data = [
            HEADERS[SHEET_PERFORMANCE],
            ["Initial_Capital", f"${metrics.get('initial_capital', 0):,.2f}"],
            ["Current_Value", f"${metrics.get('current_value', 0):,.2f}"],
            ["Total_Return%", f"{metrics.get('total_return_pct', 0):+.2f}%"],
            ["CAGR", f"{metrics.get('cagr', 0):+.2f}%"],
            ["SPY_Return%", f"{metrics.get('spy_return_pct', 0):+.2f}%"],
            ["Alpha", f"{metrics.get('alpha', 0):+.2f}%"],
            ["MDD", f"{metrics.get('mdd', 0):.2f}%"],
            ["Sharpe_Ratio", f"{metrics.get('sharpe_ratio', 0):.2f}"],
            ["Win_Rate", f"{metrics.get('win_rate', 0):.1f}%"],
            ["Total_Trades", metrics.get("total_trades", 0)],
            ["Start_Date", metrics.get("start_date", "")],
            ["Days", metrics.get("days", 0)],
            ["", ""],  # 빈 줄
            ["--- 비용/세금 ---", ""],
            ["Total_Commission", f"${metrics.get('total_commission', 0):,.2f}"],
            ["Total_Realized_PnL", f"${metrics.get('total_realized_pnl', 0):,.2f}"],
            ["Est_Tax (KRW)", f"₩{metrics.get('est_tax', 0):,.0f}"],
            ["", ""],
            ["Last_Updated", datetime.now().strftime("%Y-%m-%d %H:%M:%S")]
        ]
        
        ws.update(f"A1:B{len(data)}", data)
        print("Performance saved")
    
    
    def load_performance(self):
        """Load performance metrics"""
        ws = self._get_worksheet(SHEET_PERFORMANCE)
        data = ws.get_all_values()
        
        if len(data) <= 1:
            return {}
        
        metrics = {}
        for row in data[1:]:
            if len(row) >= 2:
                metrics[row[0]] = row[1]
        
        print("Performance loaded")
        return metrics
    
    
    # ============================================
    # Utilities
    # ============================================
    
    def clear_sheet(self, sheet_name):
        """Clear specific sheet (keep header)"""
        ws = self._get_worksheet(sheet_name)
        ws.clear()
        if sheet_name in HEADERS:
            ws.update("A1", [HEADERS[sheet_name]])
        print(f"{sheet_name} cleared")
    
    
    def clear_all(self):
        """Clear all sheets"""
        confirm = input("Clear all sheets? (y/N): ")
        if confirm.lower() != "y":
            print("Cancelled")
            return
        
        for sheet_name in HEADERS.keys():
            self.clear_sheet(sheet_name)
        print("All sheets cleared")
    
    
    def get_url(self):
        """Get spreadsheet URL"""
        return self.spreadsheet.url
    
    
    def get_summary(self):
        """Print summary"""
        print("=" * 60)
        print("Google Sheets Summary")
        print("=" * 60)
        print(f"URL: {self.spreadsheet.url}")
        print()
        
        # Holdings
        holdings_df = self.load_holdings()
        print(f"Holdings: {len(holdings_df)} stocks")
        
        # Trades
        trades_df = self.load_trades()
        print(f"Trades: {len(trades_df)} records")
        
        # Signals
        signals_df = self.load_signals()
        print(f"Signals: {len(signals_df)} records")
        
        # Daily
        daily_df = self.load_daily_values()
        print(f"Daily: {len(daily_df)} records")
        
        # Monthly
        monthly_df = self.load_monthly_values()
        print(f"Monthly: {len(monthly_df)} records")
        
        # Yearly
        yearly_df = self.load_yearly_values()
        print(f"Yearly: {len(yearly_df)} records")
        
        print("=" * 60)


# ============================================
# Test
# ============================================

if __name__ == "__main__":
    print("SheetsManager Test")
    print("=" * 60)
    sheets = SheetsManager()
    sheets.init_sheets()
    print(f"\nURL: {sheets.get_url()}")
    sheets.get_summary()