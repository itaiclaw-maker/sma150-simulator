import pandas as pd
import numpy as np
import os

# --- Configuration ---
DATA_DIR = "data"
MONTHS_IN_SIM = 14

def run_risk_analysis():
    all_metrics = []
    
    files = [f for f in os.listdir(DATA_DIR) if f.endswith('.csv')]
    
    for file in files:
        path = os.path.join(DATA_DIR, file)
        try:
            df = pd.read_csv(path, parse_dates=True, index_col=0, header=[0, 1])
            if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
            df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
            
            # Trend Filter: Positive 2023 performance
            pre_sim = df[(df.index >= '2023-01-01') & (df.index < '2024-01-01')]
            if len(pre_sim) < 200 or (pre_sim['Close'].iloc[-1] / pre_sim['Close'].iloc[0]) <= 1:
                continue
            
            # Strategy Logic
            df['SMA150'] = df['Close'].rolling(window=150).mean()
            df = df.dropna(subset=['SMA150'])
            df = df[df.index >= '2024-01-01']
            if len(df) < 20: continue
            
            df['Signal'] = 0
            df.loc[df['Close'] > df['SMA150'] * 1.005, 'Signal'] = 1
            df.loc[df['Close'] < df['SMA150'] * 0.995, 'Signal'] = -1
            df['Position'] = df['Signal'].replace(0, np.nan).ffill().fillna(0).replace(-1, 0)
            
            df['Market_Returns'] = df['Close'].pct_change()
            df['Strategy_Returns'] = (df['Market_Returns'] * df['Position'].shift(1)).fillna(0)
            # Slippage
            trades = df['Position'].diff().fillna(0).abs()
            df['Strategy_Returns'] -= (trades * 0.0005)
            
            # --- RISK METRICS ---
            df['Equity'] = (1 + df['Strategy_Returns']).cumprod()
            df['Peak'] = df['Equity'].expanding().max()
            df['Drawdown'] = (df['Equity'] - df['Peak']) / df['Peak']
            
            total_return = df['Equity'].iloc[-1] - 1
            max_dd = df['Drawdown'].min()
            volatility = df['Strategy_Returns'].std() * np.sqrt(252) # Annualized Vol
            
            all_metrics.append({
                'Ticker': file.replace('.csv', ''),
                'Return': total_return,
                'MaxDD': max_dd,
                'Vol': volatility
            })
        except:
            continue

    metrics_df = pd.DataFrame(all_metrics)
    
    # Portfolio Aggregate Metrics
    avg_return = metrics_df['Return'].mean()
    net_return = avg_return * 0.75 # Tax adjustment
    avg_max_dd = metrics_df['MaxDD'].mean()
    worst_max_dd = metrics_df['MaxDD'].min()
    portfolio_vol = metrics_df['Vol'].mean() / np.sqrt(len(metrics_df)) # Rough portfolio vol diversification

    print("\n--- Strategy Risk/Reward Blueprint ---")
    print(f"Annualized Net Profit:  {((1 + net_return)**(12/MONTHS_IN_SIM) - 1)*100:.2f}%")
    print(f"Average Ticker Max DD:  {avg_max_dd*100:.2f}%")
    print(f"Portfolio Volatility:   {portfolio_vol*100:.2f}% (Diversified)")
    print(f"Worst Ticker DD:        {worst_max_dd*100:.2f}%")
    print(f"Profit-to-Risk Ratio:   {abs(net_return/avg_max_dd):.2f} (Target > 1.0)")

if __name__ == "__main__":
    run_risk_analysis()
