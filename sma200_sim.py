import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns

# --- Configuration ---
DATA_DIR = "data"
INITIAL_CAPITAL = 100000
TAX_RATE = 0.25
NUM_SIMULATIONS = 1000
MONTHS_IN_SIM = 14
GOAL_MONTHLY = 1000
SMA_WINDOW = 200  # <--- Changed from 150 to 200

def run_sma200_monte_carlo():
    all_ticker_returns = []
    skipped_tickers = 0
    
    files = [f for f in os.listdir(DATA_DIR) if f.endswith('.csv')]
    print(f"🛰️ Running SMA{SMA_WINDOW} simulation with TREND FILTER...")
    
    for file in files:
        path = os.path.join(DATA_DIR, file)
        try:
            df = pd.read_csv(path, parse_dates=True, index_col=0, header=[0, 1])
            if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
            
            df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
            
            # --- TREND FILTER (2023 Performance) ---
            pre_sim_data = df[(df.index >= '2023-01-01') & (df.index < '2024-01-01')]
            if len(pre_sim_data) < 200: 
                skipped_tickers += 1
                continue
                
            one_year_return = (pre_sim_data['Close'].iloc[-1] / pre_sim_data['Close'].iloc[0]) - 1
            if one_year_return <= 0:
                skipped_tickers += 1
                continue
            
            # --- SMA200 Strategy Logic ---
            df[f'SMA{SMA_WINDOW}'] = df['Close'].rolling(window=SMA_WINDOW).mean()
            df = df.dropna(subset=[f'SMA{SMA_WINDOW}'])
            df = df[df.index >= '2024-01-01']
            
            if len(df) < 20: continue
            
            df['Signal'] = 0
            df.loc[df['Close'] > df[f'SMA{SMA_WINDOW}'] * 1.005, 'Signal'] = 1
            df.loc[df['Close'] < df[f'SMA{SMA_WINDOW}'] * 0.995, 'Signal'] = -1
            df['Position'] = df['Signal'].replace(0, np.nan).ffill().fillna(0).replace(-1, 0)
            
            df['Market_Returns'] = df['Close'].pct_change()
            df['Strategy_Returns'] = (df['Market_Returns'] * df['Position'].shift(1)).fillna(0)
            
            # Slippage (5 bps)
            trades = df['Position'].diff().fillna(0).abs()
            df['Strategy_Returns'] -= (trades * 0.0005)
            
            all_ticker_returns.append(df['Strategy_Returns'].values)
        except Exception:
            continue

    print(f"✅ Filtered: Kept {len(all_ticker_returns)} tickers, Pruned {skipped_tickers} laggards.")

    # Run Monte Carlo
    simulation_results = []
    for _ in range(NUM_SIMULATIONS):
        indices = np.random.choice(len(all_ticker_returns), size=100, replace=True)
        sampled_returns = [all_ticker_returns[i] for i in indices]
        min_len = min(len(r) for r in sampled_returns)
        sampled_returns = [r[:min_len] for r in sampled_returns]
        
        portfolio_daily_returns = np.mean(sampled_returns, axis=0)
        total_gross_return = np.prod(1 + portfolio_daily_returns) - 1
        gross_profit = INITIAL_CAPITAL * total_gross_return
        tax_drag = max(0, gross_profit * TAX_RATE)
        net_profit = gross_profit - tax_drag
        monthly_income = net_profit / MONTHS_IN_SIM
        
        simulation_results.append(monthly_income)

    sim_array = np.array(simulation_results)
    prob_of_success = np.sum(sim_array >= GOAL_MONTHLY) / NUM_SIMULATIONS * 100
    p10 = np.percentile(sim_array, 10)
    p50 = np.percentile(sim_array, 50)
    
    plt.figure(figsize=(10, 6))
    sns.histplot(sim_array, kde=True, color='royalblue')
    plt.axvline(GOAL_MONTHLY, color='red', linestyle='--', label=f'Goal (${GOAL_MONTHLY})')
    plt.axvline(p10, color='orange', linestyle=':', label=f'P10: ${p10:,.0f} (90% Prob)')
    plt.title(f"SMA{SMA_WINDOW} Monte Carlo: Probability after Selective Pruning\nSuccess Probability: {prob_of_success:.1f}%")
    plt.xlabel("Monthly Income ($)")
    plt.legend()
    plt.savefig(f"sma{SMA_WINDOW}_monte_carlo.png")
    
    print(f"\n--- SMA{SMA_WINDOW} Results ---")
    print(f"Success Probability: {prob_of_success:.1f}%")
    print(f"90% Confidence Floor (P10): ${p10:,.2f}")
    print(f"Median Expected (P50):      ${p50:,.2f}")

if __name__ == "__main__":
    run_sma200_monte_carlo()
