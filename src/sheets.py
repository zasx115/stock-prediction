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
    SHEET_TRADES: ["Date", "Symbol", "Action", "Shares", "Price", "Amount", "Return%", "Sector", "Memo"],
    SHEET_SIGNALS: ["Timestamp", "Analysis_Date", "Signal", "Picks", "Scores", "Allocations", "Market_Momentum", "SPY_Price", "Market_Trend"],
    SHEET_DAILY: ["Date", "Total_Value", "Cash", "Stocks_Value", "Daily_Return%", "SPY_Price", "SPY_Return%", "Alpha"],
    SHEET_MONTHLY: ["Year_Month", "Start_Value", "End_Value", "Monthly_Return%", "SPY_Return%", "Alpha", "Best_Stock", "Worst_Stock"],
    SHEET_YEARLY: ["Year", "Start_Value", "End_Value", "Yearly_Return%", "SPY_Return%", "Alpha", "Total_Trades", "Win_Rate"],
    SHEET_PERFORMANCE: ["Metric", "Value"]
}


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
            holding_data.get("symbol", ""),
            holding_data.get("shares", 0),
            holding_data.get("avg_price", 0),
            holding_data.get("sector", ""),
            holding_data.get("buy_date", "")
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
    
    
    # ============================================
    # Trades
    # ============================================
    
    def save_trades(self, trades):
        """
        Append trades to sheet
        
        Args:
            trades: list of {date, symbol, action, shares, price, amount, return_pct, sector, memo}
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
            ws.append_row([
                t.get("date", ""),
                t.get("symbol", ""),
                t.get("action", ""),
                t.get("shares", 0),
                t.get("price", 0),
                t.get("amount", 0),
                t.get("return_pct", 0),
                t.get("sector", ""),
                t.get("memo", "")
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
        
        ws.append_row([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            date_str,
            signal.get("signal", ""),
            ", ".join(signal.get("picks", [])),
            ", ".join([f"{s:.4f}" for s in signal.get("scores", [])]),
            ", ".join([f"{a:.0%}" for a in signal.get("allocations", [])]),
            signal.get("market_momentum", ""),
            signal.get("spy_price", ""),
            signal.get("market_trend", "")
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
            daily_data.get("date", ""),
            daily_data.get("total_value", 0),
            daily_data.get("cash", 0),
            daily_data.get("stocks_value", 0),
            daily_data.get("daily_return_pct", 0),
            daily_data.get("spy_price", 0),
            daily_data.get("spy_return_pct", 0),
            daily_data.get("alpha", 0)
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
            monthly_data: {year_month, start_value, end_value, monthly_return_pct, spy_return_pct, alpha, best_stock, worst_stock}
        """
        ws = self._get_worksheet(SHEET_MONTHLY)
        
        # Check header
        existing = ws.get_all_values()
        if len(existing) == 0:
            ws.append_row(HEADERS[SHEET_MONTHLY])
        
        ws.append_row([
            monthly_data.get("year_month", ""),
            monthly_data.get("start_value", 0),
            monthly_data.get("end_value", 0),
            monthly_data.get("monthly_return_pct", 0),
            monthly_data.get("spy_return_pct", 0),
            monthly_data.get("alpha", 0),
            monthly_data.get("best_stock", ""),
            monthly_data.get("worst_stock", "")
        ])
        
        print(f"Monthly value saved ({monthly_data.get('year_month', '')})")
    
    
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
            yearly_data: {year, start_value, end_value, yearly_return_pct, spy_return_pct, alpha, total_trades, win_rate}
        """
        ws = self._get_worksheet(SHEET_YEARLY)
        
        # Check header
        existing = ws.get_all_values()
        if len(existing) == 0:
            ws.append_row(HEADERS[SHEET_YEARLY])
        
        ws.append_row([
            yearly_data.get("year", ""),
            yearly_data.get("start_value", 0),
            yearly_data.get("end_value", 0),
            yearly_data.get("yearly_return_pct", 0),
            yearly_data.get("spy_return_pct", 0),
            yearly_data.get("alpha", 0),
            yearly_data.get("total_trades", 0),
            yearly_data.get("win_rate", 0)
        ])
        
        print(f"Yearly value saved ({yearly_data.get('year', '')})")
    
    
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
                total_trades, start_date, days
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