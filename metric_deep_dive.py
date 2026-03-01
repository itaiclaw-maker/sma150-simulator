import pandas as pd
import numpy as np
import os

# --- Configuration ---
DATA_DIR = "data"
TAX_RATE = 0.25
SLIPPAGE = 0.0005
WHIPSAW_BUFFER = 0.005
START_DATE_STR = '2024-01-01'

def run_multi_metric_analysis():
    all_metrics = []
    
    files = [f for f in os.listdir(DATA_DIR) if f.endswith('.csv')]
    print(f"🛰️ Running multi-metric analysis on {len(files)} tickers...")
    
    for file in files:
        path = os.path.join(DATA_DIR, file)
        try:
            df = pd.read_csv(path, parse_dates=True, index_col=0, header=[0, 1])
            if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
            df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
            
            # Strategy Logic
            df['SMA150'] = df['Close'].rolling(window=150).mean()
            df = df.dropna(subset=['SMA150'])
            df = df[df.index >= START_DATE_STR]
            if len(df) < 50: continue
            
            df['Signal'] = 0
            df.loc[df['Close'] > df['SMA150'] * (1 + WHIPSAW_BUFFER), 'Signal'] = 1
            df.loc[df['Close'] < df['SMA150'] * (1 - WHIPSAW_BUFFER), 'Signal'] = -1
            df['Position'] = df['Signal'].replace(0, np.nan).ffill().fillna(0).replace(-1, 0)
            
            # Returns & Friction
            df['Market_Returns'] = df['Close'].pct_change()
            df['Strategy_Returns'] = (df['Market_Returns'] * df['Position'].shift(1)).fillna(0)
            trades_series = df['Position'].diff().fillna(0).abs()
            num_trades = trades_series.sum()
            df['Strategy_Returns'] -= (trades_series * SLIPPAGE)
            
            # Equity & Drawdown
            df['Equity'] = (1 + df['Strategy_Returns']).cumprod()
            df['Peak'] = df['Equity'].expanding().max()
            df['Drawdown'] = (df['Equity'] - df['Peak']) / df['Peak']
            
            # Metrics
            total_return = df['Equity'].iloc[-1] - 1
            net_return_pct = total_return * (1 - TAX_RATE) if total_return > 0 else total_return
            max_dd_pct = df['Drawdown'].min() * 100
            
            all_metrics.append({
                'Ticker': file.replace('.csv', ''),
                'NetProfitPct': net_return_pct * 100,
                'NumTrades': num_trades,
                'MaxDD': max_dd_pct
            })
        except:
            continue

    m_df = pd.DataFrame(all_metrics)
    
    stats = {
        "Metric": ["Net Profit After Tax (%)", "Number of Trades", "Max Drawdown (%)"],
        "Mean": [m_df['NetProfitPct'].mean(), m_df['NumTrades'].mean(), m_df['MaxDD'].mean()],
        "Std Dev": [m_df['NetProfitPct'].std(), m_df['NumTrades'].std(), m_df['MaxDD'].std()]
    }
    
    summary_df = pd.DataFrame(stats)
    print("\n--- Current Strategy Performance Metrics (Mean & Std) ---")
    print(summary_df.to_string(index=False))
    
    print("\n🛰️ Optimization Pathways to improve these scores:")
    print("1. To reduce MaxDD Std: Implement a Stop-Loss (e.g., exit if price drops 10% below entry regardless of SMA).")
    print("2. To reduce Num Trades: Increase Whipsaw Buffer to 1.0% or 1.5%.")
    print("3. To increase Net Profit: Add a Volatility Filter (ATR) to avoid trading stocks in high-volatility sideways regimes.")

if __name__ == "__main__":
    run_multi_metric_analysis()
