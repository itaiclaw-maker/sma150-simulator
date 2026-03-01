import pandas as pd
import numpy as np
import os

# --- Configuration ---
DATA_DIR = "data"
TAX_RATE = 0.25
SLIPPAGE = 0.0005
WHIPSAW_BUFFER = 0.010 
STOP_LOSS_PCT = 0.08
START_DATE_STR = '2024-01-01'
INITIAL_CAPITAL = 110000

def run_vol_weighted_analysis():
    all_ticker_data = []
    
    files = [f for f in os.listdir(DATA_DIR) if f.endswith('.csv')]
    print(f"🛰️ Running Volatility-Weighted Analysis on {len(files)} tickers...")
    
    for file in files:
        path = os.path.join(DATA_DIR, file)
        try:
            df = pd.read_csv(path, parse_dates=True, index_col=0, header=[0, 1])
            if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
            df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
            
            # --- VOLATILITY CALCULATION (Pre-Sim: 2023) ---
            pre_sim = df[(df.index >= '2023-01-01') & (df.index < '2024-01-01')]
            if len(pre_sim) < 200: continue
            
            # Daily Return Volatility
            vol = pre_sim['Close'].pct_change().std()
            if vol == 0 or np.isnan(vol): continue
            
            # --- STRATEGY LOGIC ---
            df['SMA150'] = df['Close'].rolling(window=150).mean()
            df = df.dropna(subset=['SMA150'])
            df = df[df.index >= START_DATE_STR]
            if len(df) < 50: continue
            
            pos = 0
            entry_price = 0
            signals = []
            
            for i in range(len(df)):
                price = df['Close'].iloc[i]
                sma = df['SMA150'].iloc[i]
                if pos == 0:
                    if price > sma * (1 + WHIPSAW_BUFFER):
                        pos = 1
                        entry_price = price
                else:
                    if price < sma * (1 - WHIPSAW_BUFFER) or price < (entry_price * (1 - STOP_LOSS_PCT)):
                        pos = 0
                        entry_price = 0
                signals.append(pos)
            
            df['Position'] = signals
            df['Market_Returns'] = df['Close'].pct_change().fillna(0)
            df['Strategy_Returns'] = (df['Market_Returns'] * df['Position'].shift(1)).fillna(0)
            
            # Slippage
            trades = df['Position'].diff().fillna(0).abs()
            df['Strategy_Returns'] -= (trades * SLIPPAGE)
            
            all_ticker_data.append({
                'Ticker': file.replace('.csv', ''),
                'Vol': vol,
                'Returns': df['Strategy_Returns'].values,
                'Equity': (1 + df['Strategy_Returns']).cumprod(),
                'Trades': trades.sum()
            })
        except:
            continue

    # --- INVERSE VOLATILITY WEIGHTING ---
    # Weight = (1/Vol) / Sum(1/Vol)
    total_inv_vol = sum(1/d['Vol'] for d in all_ticker_data)
    for d in all_ticker_data:
        d['Weight'] = (1/d['Vol']) / total_inv_vol
    
    # --- PORTFOLIO METRICS ---
    # We simulate 1000 portfolios by sampling 100 tickers based on weights
    num_sims = 1000
    sim_results = []
    
    for _ in range(num_sims):
        # Sample tickers based on their Inverse-Vol weights
        # (This simulates a portfolio where size is proportional to stability)
        indices = np.random.choice(len(all_ticker_data), size=100, replace=True, p=[d['Weight'] for d in all_ticker_data])
        sampled = [all_ticker_data[i] for i in indices]
        
        min_len = min(len(s['Returns']) for s in sampled)
        # Average return of this specific weighted portfolio
        portfolio_returns = np.mean([s['Returns'][:min_len] for s in sampled], axis=0)
        
        total_ret = np.prod(1 + portfolio_returns) - 1
        net_ret = total_ret * (1 - TAX_RATE) if total_ret > 0 else total_ret
        
        # Drawdown for this sample
        equity = np.cumprod(1 + portfolio_returns)
        peak = np.maximum.accumulate(equity)
        dd = (equity - peak) / peak
        
        sim_results.append({
            'NetProfitPct': net_ret * 100,
            'NumTrades': np.mean([s['Trades'] for s in sampled]),
            'MaxDD': dd.min() * 100
        })

    m_df = pd.DataFrame(sim_results)
    
    stats = {
        "Metric": ["Net Profit After Tax (%)", "Number of Trades", "Max Drawdown (%)"],
        "Mean": [m_df['NetProfitPct'].mean(), m_df['NumTrades'].mean(), m_df['MaxDD'].mean()],
        "Std Dev": [m_df['NetProfitPct'].std(), m_df['NumTrades'].std(), m_df['MaxDD'].std()]
    }
    
    print("\n--- Volatility-Weighted Strategy Metrics (Inverse Vol) ---")
    print(pd.DataFrame(stats).to_string(index=False))

if __name__ == "__main__":
    run_vol_weighted_analysis()
