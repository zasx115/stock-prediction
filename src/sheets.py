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
SHEET_PORTFOLIO = "Portfolio"
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
    SHEET_PORTFOLIO: ["Symbol", "Shares", "Avg_Price", "Sector", "Buy_Date", "Current_Price", "Return%", "Value"],
    SHEET_TRADES: ["Date", "Symbol", "Action", "Shares", "Price", "Amount", "Commission", "Slippage", "Return%", "Sector", "Score", "Memo"],
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
    # Portfolio
    # ============================================
    
    def save_portfolio(self, portfolio, current_prices=None):
        """
        Save portfolio to sheet
        
        Args:
            portfolio: {"cash": float, "holdings": {symbol: {"shares": int, "avg_price": float, "sector": str, "buy_date": str}}}
            current_prices: {symbol: float} - optional current prices
        """
        ws = self._get_worksheet(SHEET_PORTFOLIO)
        ws.clear()
        
        # Header
        data = [HEADERS[SHEET_PORTFOLIO]]
        
        # Meta info row
        data.append(["_CASH", portfolio.get("cash", 0), "", "", portfolio.get("created_at", ""), "", "", ""])
        
        # Holdings
        for symbol, info in portfolio.get("holdings", {}).items():
            shares = info.get("shares", 0)
            avg_price = info.get("avg_price", 0)
            sector = info.get("sector", "")
            buy_date = info.get("buy_date", "")
            
            # Current price
            current_price = 0
            if current_prices and symbol in current_prices:
                current_price = current_prices[symbol]
            
            # Calculate return and value
            return_pct = 0
            value = 0
            if avg_price > 0 and current_price > 0:
                return_pct = round((current_price - avg_price) / avg_price * 100, 2)
                value = round(shares * current_price, 2)
            
            data.append([symbol, shares, avg_price, sector, buy_date, current_price, return_pct, value])
        
        ws.update(f"A1:H{len(data)}", data)
        print(f"Portfolio saved ({len(portfolio.get('holdings', {}))} holdings)")
    
    
    def load_portfolio(self, initial_capital=2000):
        """Load portfolio from sheet"""
        ws = self._get_worksheet(SHEET_PORTFOLIO)
        data = ws.get_all_values()
        
        portfolio = {
            "cash": initial_capital,
            "holdings": {},
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        if len(data) <= 1:
            self.save_portfolio(portfolio)
            return portfolio
        
        # Parse data (skip header)
        for row in data[1:]:
            if len(row) < 3:
                continue
            
            symbol = row[0]
            
            # Meta row (_CASH)
            if symbol == "_CASH":
                try:
                    portfolio["cash"] = float(row[1])
                    portfolio["created_at"] = row[4] if len(row) > 4 else ""
                except:
                    pass
                continue
            
            # Holdings
            if symbol and symbol != "Symbol":
                try:
                    portfolio["holdings"][symbol] = {
                        "shares": int(float(row[1])),
                        "avg_price": float(row[2]),
                        "sector": row[3] if len(row) > 3 else "",
                        "buy_date": row[4] if len(row) > 4 else ""
                    }
                except:
                    pass
        
        print(f"Portfolio loaded ({len(portfolio['holdings'])} holdings)")
        return portfolio
    
    
    # ============================================
    # Trades
    # ============================================
    
    def save_trades(self, trades):
        """
        Append trades to sheet
        
        Args:
            trades: list of {date, symbol, action, shares, price, amount, commission, slippage, return_pct, sector, score, memo}
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
                t.get("commission", 0),
                t.get("slippage", 0),
                t.get("return_pct", 0),
                t.get("sector", ""),
                t.get("score", ""),
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
        
        # Portfolio
        portfolio = self.load_portfolio()
        print(f"Portfolio: ${portfolio['cash']:,.2f} cash, {len(portfolio['holdings'])} holdings")
        
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