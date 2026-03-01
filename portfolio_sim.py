import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
import pandas as pd
import numpy as np
import os
from datetime import datetime

# --- Configuration & Guardrails ---
# Expanded list of 100 tickers (Representative mix across sectors + survivorship bias proxies)
# Including Mega caps, Mid caps, and known 2024 laggards/removals
TICKERS_100 = [
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA', 'NVDA', 'BRK-B', 'V', 'JPM',
    'UNH', 'MA', 'HD', 'PG', 'AVGO', 'ABBV', 'LLY', 'COST', 'MRK', 'ADBE',
    'CRM', 'PEP', 'CVX', 'WMT', 'KO', 'TMO', 'BAC', 'ACN', 'MCD', 'CSCO',
    'LIN', 'ABT', 'ORCL', 'INTU', 'AMD', 'PFE', 'WFC', 'DIS', 'PM', 'TXN',
    'COP', 'VZ', 'CAT', 'HON', 'AMAT', 'QCOM', 'LOW', 'RTX', 'UNP', 'IBM',
    'AXP', 'INTC', 'SBUX', 'GS', 'UPS', 'DE', 'GILD', 'PLD', 'MS', 'GE',
    'LMT', 'ISRG', 'MDLZ', 'SYK', 'T', 'BLK', 'BA', 'TJX', 'ADI', 'ADP',
    'CVS', 'MMC', 'AMGN', 'CB', 'VRTX', 'REGN', 'MDT', 'CI', 'BSX', 'PANW',
    'SNPS', 'ZTS', 'PYPL', 'FISV', 'MU', 'LRCX', 'ETN', 'ITW', 'SCHW', 'EL',
    'HUM', 'NKE', 'MO', 'TGT', 'BDX', 'DUK', 'USB', 'AAL', 'WBA', 'PARA'
]

START_DATE = "2023-01-01"
END_DATE = datetime.now().strftime("%Y-%m-%d")

SMA_WINDOW = 150
SLIPPAGE_BPS = 0.0005  # 5 basis points
TAX_RATE = 0.25        # 25% Capital Gains tax
WHIPSAW_BUFFER = 0.005 # 0.5% price buffer
INITIAL_CAPITAL = 100000
MONTHS_IN_SIM = 14 # Approx since Jan 2024 start of execution

DATA_DIR = "data"

def fetch_data(tickers):
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
    
    print(f"🛰️ Fetching data for {len(tickers)} tickers...")
    # Using thread-safe downloader for speed with 100 tickers
    data = yf.download(tickers, start=START_DATE, end=END_DATE, group_by='ticker')
    for ticker in tickers:
        try:
            df = data[ticker]
            if not df.empty:
                df.to_csv(f"{DATA_DIR}/{ticker}.csv")
        except:
            continue

def calculate_indicators(ticker):
    path = f"{DATA_DIR}/{ticker}.csv"
    if not os.path.exists(path): return None
    df = pd.read_csv(path, parse_dates=True, index_col=0, header=[0, 1])
    if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
    df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
    df['SMA150'] = df['Close'].rolling(window=SMA_WINDOW).mean()
    df['Signal'] = 0
    df.loc[df['Close'] > df['SMA150'] * (1 + WHIPSAW_BUFFER), 'Signal'] = 1
    df.loc[df['Close'] < df['SMA150'] * (1 - WHIPSAW_BUFFER), 'Signal'] = -1
    df['Position'] = df['Signal'].replace(0, np.nan).ffill().fillna(0).replace(-1, 0)
    return df

def run_portfolio_sim():
    fetch_data(TICKERS_100)
    
    portfolio_value = INITIAL_CAPITAL
    allocation_per_ticker = INITIAL_CAPITAL / len(TICKERS_100)
    
    all_returns = []
    
    for ticker in TICKERS_100:
        df = calculate_indicators(ticker)
        if df is None: continue
        
        df = df.dropna(subset=['SMA150'])
        # Filter for the actual strategy start (Jan 1, 2024 onwards)
        df = df[df.index >= '2024-01-01']
        if df.empty: continue
        
        df['Market_Returns'] = df['Close'].pct_change()
        df['Strategy_Returns'] = df['Market_Returns'] * df['Position'].shift(1)
        
        # Friction
        trades = df['Position'].diff().fillna(0).abs()
        df['Strategy_Returns'] -= (trades * SLIPPAGE_BPS)
        
        # Cumulative Strategy Return for this ticker
        ticker_cum_return = (1 + df['Strategy_Returns'].fillna(0)).prod()
        all_returns.append(ticker_cum_return)
        
    # Aggregate Portfolio Result
    avg_gross_return = np.mean(all_returns) - 1
    gross_profit = INITIAL_CAPITAL * avg_gross_return
    tax_drag = max(0, gross_profit * TAX_RATE)
    net_profit = gross_profit - tax_drag
    monthly_income = net_profit / MONTHS_IN_SIM
    
    print("\n--- Portfolio Aggregation (100 Tickers) ---")
    print(f"Initial Capital: ${INITIAL_CAPITAL:,}")
    print(f"Gross Profit:    ${gross_profit:,.2f}")
    print(f"Tax Drag (25%):  -${tax_drag:,.2f}")
    print(f"Net Profit:      ${net_profit:,.2f}")
    print(f"Estimated Monthly Income: ${monthly_income:,.2f}")
    
    if monthly_income >= 1000:
        print("✅ Goal Met: This strategy currently supports a $1k/mo stream.")
    else:
        print(f"❌ Goal Shortfall: Need ${1000 - monthly_income:,.2f} more per month.")

if __name__ == "__main__":
    run_portfolio_sim()
