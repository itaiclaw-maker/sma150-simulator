import pandas as pd
import numpy as np
import os

# --- Configuration ---
DATA_DIR = "data"
TAX_RATE = 0.25
SLIPPAGE = 0.0005
WHIPSAW_BUFFER = 0.010 # Increased to 1.0% to reduce trades
START_DATE_STR = '2024-01-01'
STOP_LOSS_PCT = 0.08 # 8% Hard stop loss below entry price

def run_optimized_analysis():
    all_metrics = []
    
    files = [f for f in os.listdir(DATA_DIR) if f.endswith('.csv')]
    
    for file in files:
        path = os.path.join(DATA_DIR, file)
        try:
            df = pd.read_csv(path, parse_dates=True, index_col=0, header=[0, 1])
            if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
            df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
            
            df['SMA150'] = df['Close'].rolling(window=150).mean()
            df = df.dropna(subset=['SMA150'])
            df = df[df.index >= START_DATE_STR]
            if len(df) < 50: continue
            
            # --- Trading Logic with Stop Loss ---
            pos = 0
            entry_price = 0
            signals = []
            
            for i in range(len(df)):
                price = df['Close'].iloc[i]
                sma = df['SMA150'].iloc[i]
                
                # Check for Entry
                if pos == 0:
                    if price > sma * (1 + WHIPSAW_BUFFER):
                        pos = 1
                        entry_price = price
                    signals.append(pos)
                # Check for Exit (SMA cross or Stop Loss)
                else:
                    stop_price = entry_price * (1 - STOP_LOSS_PCT)
                    if price < sma * (1 - WHIPSAW_BUFFER) or price < stop_price:
                        pos = 0
                        entry_price = 0
                    signals.append(pos)
            
            df['Position'] = signals
            df['Market_Returns'] = df['Close'].pct_change()
            df['Strategy_Returns'] = (df['Market_Returns'] * df['Position'].shift(1)).fillna(0)
            
            trades_series = df['Position'].diff().fillna(0).abs()
            df['Strategy_Returns'] -= (trades_series * SLIPPAGE)
            
            df['Equity'] = (1 + df['Strategy_Returns']).cumprod()
            df['Peak'] = df['Equity'].expanding().max()
            df['Drawdown'] = (df['Equity'] - df['Peak']) / df['Peak']
            
            total_return = df['Equity'].iloc[-1] - 1
            net_return_pct = total_return * (1 - TAX_RATE) if total_return > 0 else total_return
            
            all_metrics.append({
                'NetProfitPct': net_return_pct * 100,
                'NumTrades': trades_series.sum(),
                'MaxDD': df['Drawdown'].min() * 100
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
    print("\n--- Optimized Strategy Metrics (1% Buffer + 8% Stop Loss) ---")
    print(summary_df.to_string(index=False))

if __name__ == "__main__":
    run_optimized_analysis()
