import pandas as pd
import yfinance as yf
import numpy as np
import os
from datetime import datetime, timedelta

# --- CONFIGURATION ---
UNIVERSE_START_DATE = "2024-01-01"
BACKTEST_YEARS = 2
SMA_WINDOW = 150
NUM_TICKERS = 50
DATA_DIR = "projects/sma150-simulator/data"
SLIPPAGE_BPS = 5 # 0.05%
TAX_RATE = 0.25
WHIPSAW_BUFFER_PCT = 0.005 # 0.5%

# Tickers removed from S&P 500 in 2024/2025 (Partial list for survivorship bias modeling)
# In a full production version, we would fetch the point-in-time constituents.
REMOVED_TICKERS = {
    "WBA": "2025-08-28",
    "AAL": "2024-09-23",
    "ILMN": "2024-06-24",
    "ETSY": "2024-09-23",
    "ZION": "2024-03-18",
    "WHR": "2024-03-18",
    "PXD": "2024-05-08",
    "XRAY": "2024-04-03",
    "VFC": "2024-04-03",
}

def get_sp500_tickers():
    """Fetches current S&P 500 tickers from Wikipedia."""
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    table = pd.read_html(url)
    df = table[0]
    return df['Symbol'].tolist()

def prepare_universe():
    """Combines current tickers with removed ones to mitigate survivorship bias."""
    current_tickers = get_sp500_tickers()
    # Add removed tickers
    full_universe = list(set(current_tickers + list(REMOVED_TICKERS.keys())))
    # For this simulation, we select a random subset
    np.random.seed(42) # For reproducibility
    selected_tickers = np.random.choice(full_universe, NUM_TICKERS, replace=False).tolist()
    return selected_tickers

def fetch_data(tickers):
    """Downloads historical data with a warmup period for SMA."""
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        
    end_date = datetime.now().strftime('%Y-%m-%d')
    # Warmup period: roughly 220 calendar days for 150 trading days
    start_date = (datetime.strptime(UNIVERSE_START_DATE, '%Y-%m-%d') - timedelta(days=220)).strftime('%Y-%m-%d')
    
    print(f"Fetching data for {len(tickers)} tickers from {start_date} to {end_date}...")
    
    data = yf.download(tickers, start=start_date, end=end_date, group_by='ticker')
    return data

def process_sma(data, tickers):
    """Calculates 150 SMA and stores results."""
    results = {}
    for ticker in tickers:
        try:
            df = data[ticker].copy()
            if df.empty: continue
            
            # Calculate 150 SMA
            df['SMA150'] = df['Adj Close'].rolling(window=SMA_WINDOW).mean()
            
            # Identify removal date if applicable
            if ticker in REMOVED_TICKERS:
                removal_date = pd.to_datetime(REMOVED_TICKERS[ticker])
                df = df[df.index <= removal_date]
                
            results[ticker] = df
            df.to_csv(f"{DATA_DIR}/{ticker}_processed.csv")
        except Exception as e:
            print(f"Error processing {ticker}: {e}")
            
    print(f"Processing complete. Data saved to {DATA_DIR}")
    return results

if __name__ == "__main__":
    tickers = prepare_universe()
    data = fetch_data(tickers)
    process_sma(data, tickers)
