# ============================================

# src/sheets.py

# Google Sheets Connection Module

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

SHEET_PORTFOLIO = "Portfolio"
SHEET_TRADES = "Trades"
SHEET_SIGNALS = "Signals"

SCOPES = [
"https://www.googleapis.com/auth/spreadsheets",
"https://www.googleapis.com/auth/drive"
]

# ============================================

# SheetsManager Class

# ============================================

class SheetsManager:

def __init__(self, spreadsheet_name=SPREADSHEET_NAME):
    self.spreadsheet_name = spreadsheet_name
    self.gc = None
    self.spreadsheet = None
    self._connect()


def _connect(self):
    # Method 1: Service Account File
    if os.path.exists(SERVICE_ACCOUNT_FILE):
        creds = Credentials.from_service_account_file(
            SERVICE_ACCOUNT_FILE,
            scopes=SCOPES
        )
        self.gc = gspread.authorize(creds)
        print("Connected via service account file")
    
    # Method 2: Environment Variable (GitHub Actions)
    elif os.environ.get("GOOGLE_CREDENTIALS"):
        creds_json = os.environ.get("GOOGLE_CREDENTIALS")
        creds_dict = json.loads(creds_json)
        creds = Credentials.from_service_account_info(
            creds_dict,
            scopes=SCOPES
        )
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
        print(f"Opened spreadsheet: {self.spreadsheet_name}")
    except gspread.SpreadsheetNotFound:
        print("Spreadsheet not found. Creating new one...")
        self._create_spreadsheet()


def _create_spreadsheet(self):
    self.spreadsheet = self.gc.create(self.spreadsheet_name)
    
    self.spreadsheet.add_worksheet(title=SHEET_PORTFOLIO, rows=100, cols=10)
    self.spreadsheet.add_worksheet(title=SHEET_TRADES, rows=5000, cols=15)
    self.spreadsheet.add_worksheet(title=SHEET_SIGNALS, rows=1000, cols=10)
    
    try:
        default_sheet = self.spreadsheet.sheet1
        self.spreadsheet.del_worksheet(default_sheet)
    except:
        pass
    
    self._init_headers()
    
    print(f"Created spreadsheet: {self.spreadsheet_name}")
    print(f"URL: {self.spreadsheet.url}")


def _init_headers(self):
    # Trades header
    trades_ws = self.spreadsheet.worksheet(SHEET_TRADES)
    trades_ws.update("A1:I1", [[
        "Date", "Symbol", "Action", "Shares",
        "Price", "Amount", "Commission", "Slippage", "Return%"
    ]])
    
    # Signals header
    signals_ws = self.spreadsheet.worksheet(SHEET_SIGNALS)
    signals_ws.update("A1:G1", [[
        "Timestamp", "Analysis_Date", "Signal",
        "Picks", "Scores", "Allocations", "Market_Momentum"
    ]])


def _get_worksheet(self, sheet_name):
    try:
        return self.spreadsheet.worksheet(sheet_name)
    except gspread.WorksheetNotFound:
        return self.spreadsheet.add_worksheet(
            title=sheet_name, rows=1000, cols=15
        )


# ============================================
# Portfolio Save/Load
# ============================================

def save_portfolio(self, portfolio):
    ws = self._get_worksheet(SHEET_PORTFOLIO)
    ws.clear()
    
    data = [
        ["=== Portfolio ===", ""],
        ["cash", portfolio.get("cash", 0)],
        ["created_at", portfolio.get("created_at", "")],
        ["last_updated", datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
        ["", ""],
        ["Symbol", "Shares", "Avg_Price"]
    ]
    
    for symbol, info in portfolio.get("holdings", {}).items():
        data.append([symbol, info["shares"], info["avg_price"]])
    
    ws.update(f"A1:C{len(data)}", data)
    print(f"Portfolio saved ({len(portfolio.get('holdings', {}))} holdings)")


def load_portfolio(self, initial_capital=2000):
    ws = self._get_worksheet(SHEET_PORTFOLIO)
    data = ws.get_all_values()
    
    if len(data) <= 1:
        portfolio = {
            "cash": initial_capital,
            "holdings": {},
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        self.save_portfolio(portfolio)
        return portfolio
    
    portfolio = {
        "cash": initial_capital,
        "holdings": {},
        "created_at": "",
        "last_updated": ""
    }
    
    for row in data[1:5]:
        if len(row) >= 2:
            key = row[0].lower()
            if key == "cash":
                try:
                    portfolio["cash"] = float(row[1])
                except:
                    pass
            elif key == "created_at":
                portfolio["created_at"] = row[1]
            elif key == "last_updated":
                portfolio["last_updated"] = row[1]
    
    for row in data[6:]:
        if len(row) >= 3 and row[0] and row[0] != "Symbol":
            try:
                portfolio["holdings"][row[0]] = {
                    "shares": int(float(row[1])),
                    "avg_price": float(row[2])
                }
            except:
                pass
    
    print(f"Portfolio loaded ({len(portfolio['holdings'])} holdings)")
    return portfolio


# ============================================
# Trades Save/Load
# ============================================

def save_trades(self, trades):
    if not trades:
        return
    
    ws = self._get_worksheet(SHEET_TRADES)
    
    existing = ws.get_all_values()
    if len(existing) == 0:
        ws.append_row([
            "Date", "Symbol", "Action", "Shares",
            "Price", "Amount", "Commission", "Slippage", "Return%"
        ])
    
    for trade in trades:
        ws.append_row([
            trade.get("date", ""),
            trade.get("symbol", ""),
            trade.get("action", ""),
            trade.get("shares", 0),
            trade.get("price", 0),
            trade.get("amount", 0),
            trade.get("commission", 0),
            trade.get("slippage", 0),
            trade.get("return_pct", 0)
        ])
    
    print(f"Trades saved ({len(trades)} rows)")


def load_trades(self):
    ws = self._get_worksheet(SHEET_TRADES)
    data = ws.get_all_values()
    
    if len(data) <= 1:
        return pd.DataFrame()
    
    df = pd.DataFrame(data[1:], columns=data[0])
    print(f"Trades loaded ({len(df)} rows)")
    return df


# ============================================
# Signals Save/Load
# ============================================

def save_signal(self, signal):
    ws = self._get_worksheet(SHEET_SIGNALS)
    
    existing = ws.get_all_values()
    if len(existing) == 0:
        ws.append_row([
            "Timestamp", "Analysis_Date", "Signal",
            "Picks", "Scores", "Allocations", "Market_Momentum"
        ])
    
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
        signal.get("market_momentum", "")
    ])
    
    print(f"Signal saved ({signal.get('signal', '')})")


def load_signals(self):
    ws = self._get_worksheet(SHEET_SIGNALS)
    data = ws.get_all_values()
    
    if len(data) <= 1:
        return pd.DataFrame()
    
    df = pd.DataFrame(data[1:], columns=data[0])
    print(f"Signals loaded ({len(df)} rows)")
    return df


# ============================================
# Utilities
# ============================================

def clear_all(self):
    confirm = input("Clear all sheets? (y/N): ")
    if confirm.lower() != "y":
        print("Cancelled")
        return
    
    for sheet_name in [SHEET_PORTFOLIO, SHEET_TRADES, SHEET_SIGNALS]:
        ws = self._get_worksheet(sheet_name)
        ws.clear()
    
    self._init_headers()
    print("All sheets cleared")


def get_url(self):
    return self.spreadsheet.url


def get_summary(self):
    print("=" * 60)
    print("Google Sheets Summary")
    print("=" * 60)
    print(f"Spreadsheet: {self.spreadsheet_name}")
    print(f"URL: {self.spreadsheet.url}")
    print()
    
    portfolio = self.load_portfolio()
    print(f"Portfolio:")
    print(f"  Cash: ${portfolio['cash']:,.2f}")
    print(f"  Holdings: {len(portfolio['holdings'])} stocks")
    
    trades_df = self.load_trades()
    print(f"Trades: {len(trades_df)} records")
    
    signals_df = self.load_signals()
    print(f"Signals: {len(signals_df)} records")
    
    print("=" * 60)

# ============================================

# Test

# ============================================

if **name** == “**main**”:
print(“SheetsManager Test”)
print(”=” * 60)


sheets = SheetsManager()
print(f"URL: {sheets.get_url()}")
sheets.get_summary()