# ============================================

# 파일명: src/sheets.py

# 설명: Google Sheets 연동 모듈

# 

# 기능:

# - 서비스 계정으로 Google Sheets 인증

# - 포트폴리오 저장/로드

# - 거래 내역 저장/로드

# - 신호 기록 저장/로드

# 

# 사용법:

# from src.sheets import SheetsManager

# sheets = SheetsManager()

# sheets.save_portfolio(portfolio)

# ============================================

import gspread
from google.oauth2.service_account import Credentials
from datetime import datetime
import pandas as pd
import os
import json

# ============================================

# 설정

# ============================================

# Google Sheets 설정

SPREADSHEET_NAME = “Stock_Paper_Trading”

# 서비스 계정 키 경로 (GitHub Secrets에서 가져옴)

SERVICE_ACCOUNT_FILE = “service_account.json”

# 시트 이름

SHEET_PORTFOLIO = “Portfolio”
SHEET_TRADES = “Trades”
SHEET_SIGNALS = “Signals”

# API 범위

SCOPES = [
“https://www.googleapis.com/auth/spreadsheets”,
“https://www.googleapis.com/auth/drive”
]

# ============================================

# SheetsManager 클래스

# ============================================

class SheetsManager:
“””
Google Sheets 연동 관리자

```
사용 예시:
    sheets = SheetsManager()
    
    # 포트폴리오
    sheets.save_portfolio(portfolio)
    portfolio = sheets.load_portfolio()
    
    # 거래 기록
    sheets.save_trades(trades)
    trades_df = sheets.load_trades()
    
    # 신호 기록
    sheets.save_signal(signal)
    signals_df = sheets.load_signals()
"""

def __init__(self, spreadsheet_name=SPREADSHEET_NAME):
    """
    초기화 및 Google Sheets 연결
    
    Args:
        spreadsheet_name: 스프레드시트 이름
    """
    self.spreadsheet_name = spreadsheet_name
    self.gc = None
    self.spreadsheet = None
    
    # 연결
    self._connect()


# ============================================
# 연결
# ============================================

def _connect(self):
    """Google Sheets 연결"""
    
    # 방법 1: 서비스 계정 파일
    if os.path.exists(SERVICE_ACCOUNT_FILE):
        creds = Credentials.from_service_account_file(
            SERVICE_ACCOUNT_FILE,
            scopes=SCOPES
        )
        self.gc = gspread.authorize(creds)
        print(f"Connected via service account file")
    
    # 방법 2: 환경 변수 (GitHub Actions)
    elif os.environ.get("GOOGLE_CREDENTIALS"):
        creds_json = os.environ.get("GOOGLE_CREDENTIALS")
        creds_dict = json.loads(creds_json)
        creds = Credentials.from_service_account_info(
            creds_dict,
            scopes=SCOPES
        )
        self.gc = gspread.authorize(creds)
        print(f"Connected via environment variable")
    
    # 방법 3: Colab 인증
    else:
        try:
            from google.colab import auth
            from google.auth import default
            auth.authenticate_user()
            creds, _ = default()
            self.gc = gspread.authorize(creds)
            print(f"Connected via Colab auth")
        except:
            raise Exception("No valid credentials found!")
    
    # 스프레드시트 열기
    try:
        self.spreadsheet = self.gc.open(self.spreadsheet_name)
        print(f"Opened spreadsheet: {self.spreadsheet_name}")
    except gspread.SpreadsheetNotFound:
        print(f"Spreadsheet not found. Creating new one...")
        self._create_spreadsheet()


def _create_spreadsheet(self):
    """새 스프레드시트 생성"""
    self.spreadsheet = self.gc.create(self.spreadsheet_name)
    
    # 시트 생성
    self.spreadsheet.add_worksheet(title=SHEET_PORTFOLIO, rows=100, cols=10)
    self.spreadsheet.add_worksheet(title=SHEET_TRADES, rows=5000, cols=15)
    self.spreadsheet.add_worksheet(title=SHEET_SIGNALS, rows=1000, cols=10)
    
    # 기본 Sheet1 삭제
    try:
        default_sheet = self.spreadsheet.sheet1
        self.spreadsheet.del_worksheet(default_sheet)
    except:
        pass
    
    # 헤더 설정
    self._init_headers()
    
    print(f"Created spreadsheet: {self.spreadsheet_name}")
    print(f"URL: {self.spreadsheet.url}")


def _init_headers(self):
    """시트 헤더 초기화"""
    
    # Trades 헤더
    trades_ws = self.spreadsheet.worksheet(SHEET_TRADES)
    trades_ws.update("A1:I1", [[
        "Date", "Symbol", "Action", "Shares", 
        "Price", "Amount", "Commission", "Slippage", "Return%"
    ]])
    
    # Signals 헤더
    signals_ws = self.spreadsheet.worksheet(SHEET_SIGNALS)
    signals_ws.update("A1:G1", [[
        "Timestamp", "Analysis_Date", "Signal", 
        "Picks", "Scores", "Allocations", "Market_Momentum"
    ]])


def _get_worksheet(self, sheet_name):
    """시트 가져오기 (없으면 생성)"""
    try:
        return self.spreadsheet.worksheet(sheet_name)
    except gspread.WorksheetNotFound:
        return self.spreadsheet.add_worksheet(
            title=sheet_name, rows=1000, cols=15
        )


# ============================================
# 포트폴리오 저장/로드
# ============================================

def save_portfolio(self, portfolio):
    """
    포트폴리오 저장
    
    Args:
        portfolio: {
            "cash": float,
            "holdings": {symbol: {"shares": int, "avg_price": float}},
            "created_at": str,
            "last_updated": str
        }
    """
    ws = self._get_worksheet(SHEET_PORTFOLIO)
    ws.clear()
    
    # 메타 정보
    data = [
        ["=== Portfolio ===", ""],
        ["cash", portfolio.get("cash", 0)],
        ["created_at", portfolio.get("created_at", "")],
        ["last_updated", datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
        ["", ""],
        ["Symbol", "Shares", "Avg_Price"]
    ]
    
    # 보유 종목
    for symbol, info in portfolio.get("holdings", {}).items():
        data.append([symbol, info["shares"], info["avg_price"]])
    
    # 저장
    ws.update(f"A1:C{len(data)}", data)
    print(f"Portfolio saved ({len(portfolio.get('holdings', {}))} holdings)")


def load_portfolio(self, initial_capital=2000):
    """
    포트폴리오 로드
    
    Args:
        initial_capital: 초기 자본금 (포트폴리오 없을 때)
    
    Returns:
        dict: 포트폴리오
    """
    ws = self._get_worksheet(SHEET_PORTFOLIO)
    data = ws.get_all_values()
    
    # 빈 시트면 초기 포트폴리오 반환
    if len(data) <= 1:
        portfolio = {
            "cash": initial_capital,
            "holdings": {},
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        self.save_portfolio(portfolio)
        return portfolio
    
    # 파싱
    portfolio = {
        "cash": initial_capital,
        "holdings": {},
        "created_at": "",
        "last_updated": ""
    }
    
    # 메타 정보 (2~4행)
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
    
    # 보유 종목 (7행부터)
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
# 거래 내역 저장/로드
# ============================================

def save_trades(self, trades):
    """
    거래 내역 저장 (추가)
    
    Args:
        trades: list of dict
            [{
                "date": str,
                "symbol": str,
                "action": str (BUY/SELL/ADD/REDUCE/STOP_LOSS),
                "shares": int,
                "price": float,
                "amount": float,
                "commission": float,
                "slippage": float,
                "return_pct": float
            }, ...]
    """
    if not trades:
        return
    
    ws = self._get_worksheet(SHEET_TRADES)
    
    # 헤더 확인
    existing = ws.get_all_values()
    if len(existing) == 0:
        ws.append_row([
            "Date", "Symbol", "Action", "Shares",
            "Price", "Amount", "Commission", "Slippage", "Return%"
        ])
    
    # 거래 추가
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
    """
    거래 내역 로드
    
    Returns:
        DataFrame: 거래 내역
    """
    ws = self._get_worksheet(SHEET_TRADES)
    data = ws.get_all_values()
    
    if len(data) <= 1:
        return pd.DataFrame()
    
    df = pd.DataFrame(data[1:], columns=data[0])
    print(f"Trades loaded ({len(df)} rows)")
    return df


# ============================================
# 신호 기록 저장/로드
# ============================================

def save_signal(self, signal):
    """
    신호 저장 (추가)
    
    Args:
        signal: {
            "date": datetime or str,
            "signal": str (BUY/HOLD/ERROR),
            "picks": list,
            "scores": list,
            "allocations": list,
            "market_momentum": float (optional)
        }
    """
    ws = self._get_worksheet(SHEET_SIGNALS)
    
    # 헤더 확인
    existing = ws.get_all_values()
    if len(existing) == 0:
        ws.append_row([
            "Timestamp", "Analysis_Date", "Signal",
            "Picks", "Scores", "Allocations", "Market_Momentum"
        ])
    
    # 날짜 변환
    if hasattr(signal.get("date"), "strftime"):
        date_str = signal["date"].strftime("%Y-%m-%d")
    else:
        date_str = str(signal.get("date", ""))
    
    # 신호 추가
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
    """
    신호 기록 로드
    
    Returns:
        DataFrame: 신호 기록
    """
    ws = self._get_worksheet(SHEET_SIGNALS)
    data = ws.get_all_values()
    
    if len(data) <= 1:
        return pd.DataFrame()
    
    df = pd.DataFrame(data[1:], columns=data[0])
    print(f"Signals loaded ({len(df)} rows)")
    return df


# ============================================
# 유틸리티
# ============================================

def clear_all(self):
    """모든 시트 초기화"""
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
    """스프레드시트 URL 반환"""
    return self.spreadsheet.url


def get_summary(self):
    """
    전체 요약 출력
    """
    print("=" * 60)
    print("Google Sheets Summary")
    print("=" * 60)
    print(f"Spreadsheet: {self.spreadsheet_name}")
    print(f"URL: {self.spreadsheet.url}")
    print()
    
    # 포트폴리오
    portfolio = self.load_portfolio()
    print(f"Portfolio:")
    print(f"  Cash: ${portfolio['cash']:,.2f}")
    print(f"  Holdings: {len(portfolio['holdings'])} stocks")
    
    # 거래
    trades_df = self.load_trades()
    print(f"Trades: {len(trades_df)} records")
    
    # 신호
    signals_df = self.load_signals()
    print(f"Signals: {len(signals_df)} records")
    
    print("=" * 60)
```

# ============================================

# 테스트

# ============================================

if **name** == “**main**”:
print(“SheetsManager Test”)
print(”=” * 60)

```
# 연결 테스트
sheets = SheetsManager()

# URL 출력
print(f"\nSpreadsheet URL: {sheets.get_url()}")

# 요약
sheets.get_summary()
```