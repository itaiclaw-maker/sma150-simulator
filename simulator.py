import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
import pandas as pd
import numpy as np
import os
from datetime import datetime

# --- Configuration & Guardrails ---
TICKERS = [
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA', 'NVDA', # Winners
    'WBA', 'AAL', 'INTC', 'DIS', 'PYPL', 'PARA', 'LUV'       # Underperformers
]

START_DATE = "2023-01-01"
END_DATE = datetime.now().strftime("%Y-%m-%d")

SMA_WINDOW = 150
SLIPPAGE_BPS = 0.0005  # 5 basis points
TAX_RATE = 0.25        # 25% Capital Gains tax
WHIPSAW_BUFFER = 0.005 # 0.5% price buffer

DATA_DIR = "data"

def fetch_data(tickers):
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
    
    print(f"🛰️ Fetching data for {len(tickers)} tickers...")
    for ticker in tickers:
        path = f"{DATA_DIR}/{ticker}.csv"
        df = yf.download(ticker, start=START_DATE, end=END_DATE)
        if not df.empty:
            df.to_csv(path)

def calculate_indicators(ticker):
    path = f"{DATA_DIR}/{ticker}.csv"
    if not os.path.exists(path):
        return None
    
    df = pd.read_csv(path, parse_dates=True, index_col=0, header=[0, 1])
    
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    
    df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
    df['SMA150'] = df['Close'].rolling(window=SMA_WINDOW).mean()
    
    df['Signal'] = 0
    df.loc[df['Close'] > df['SMA150'] * (1 + WHIPSAW_BUFFER), 'Signal'] = 1
    df.loc[df['Close'] < df['SMA150'] * (1 - WHIPSAW_BUFFER), 'Signal'] = -1
    
    df['Position'] = df['Signal'].replace(0, np.nan).ffill().fillna(0).replace(-1, 0)
    
    df.to_csv(path)
    return df

def run_simulation():
    fetch_data(TICKERS)
    results = []
    all_equity_curves = pd.DataFrame()

    for ticker in TICKERS:
        print(f"🛰️ Processing {ticker}...")
        df = calculate_indicators(ticker)
        if df is not None:
            df['Market_Returns'] = df['Close'].pct_change()
            df['Strategy_Returns'] = df['Market_Returns'] * df['Position'].shift(1)
            
            # Slippage
            trades = df['Position'].diff().fillna(0).abs()
            df['Strategy_Returns'] -= (trades * SLIPPAGE_BPS)
            
            # Equity Curve
            # Ensure we only calculate from the first valid SMA150 point
            df = df.dropna(subset=['SMA150'])
            if df.empty: continue
            
            df['Equity_Curve'] = (1 + df['Strategy_Returns'].fillna(0)).cumprod()
            all_equity_curves[ticker] = df['Equity_Curve']
            
            total_return = df['Equity_Curve'].iloc[-1] - 1
            tax_drag = total_return * TAX_RATE if total_return > 0 else 0
            net_return = total_return - tax_drag

            results.append({
                'Ticker': ticker, 
                'Gross ROI': total_return, 
                'Tax Drag': tax_drag, 
                'Net ROI': net_return
            })
    
    summary = pd.DataFrame(results)
    print("\n--- Strategy Summary (Phase 2: Tax Adjusted) ---")
    print(summary)
    
    # Plotting
    plt.figure(figsize=(12, 8))
    sns.set_style("darkgrid")
    for column in all_equity_curves.columns:
        plt.plot(all_equity_curves.index, all_equity_curves[column], label=column, alpha=0.7)
    
    plt.title("SMA150 Crossover Strategy - Equity Curves (Normalized to 1.0)")
    plt.xlabel("Date")
    plt.ylabel("Portfolio Value")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig("equity_curves.png")
    print("\n📊 Equity curves saved to equity_curves.png")

if __name__ == "__main__":
    run_simulation()
