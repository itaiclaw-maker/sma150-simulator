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

def run_monte_carlo():
    # 1. Load all processed strategy returns from existing data
    all_ticker_returns = []
    
    files = [f for f in os.listdir(DATA_DIR) if f.endswith('.csv')]
    print(f"🛰️ Loading strategy returns from {len(files)} tickers...")
    
    for file in files:
        ticker = file.replace('.csv', '')
        path = os.path.join(DATA_DIR, file)
        try:
            # We need the strategy returns we calculated earlier
            # Re-calculating briefly from the CSVs for precision
            df = pd.read_csv(path, parse_dates=True, index_col=0, header=[0, 1])
            if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
            
            df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
            df['SMA150'] = df['Close'].rolling(window=150).mean()
            df = df.dropna(subset=['SMA150'])
            df = df[df.index >= '2024-01-01']
            
            if len(df) < 20: continue
            
            # Re-generate signal and returns
            df['Signal'] = 0
            df.loc[df['Close'] > df['SMA150'] * 1.005, 'Signal'] = 1
            df.loc[df['Close'] < df['SMA150'] * 0.995, 'Signal'] = -1
            df['Position'] = df['Signal'].replace(0, np.nan).ffill().fillna(0).replace(-1, 0)
            
            df['Market_Returns'] = df['Close'].pct_change()
            df['Strategy_Returns'] = (df['Market_Returns'] * df['Position'].shift(1)).fillna(0)
            
            # Apply slippage per trade
            trades = df['Position'].diff().fillna(0).abs()
            df['Strategy_Returns'] -= (trades * 0.0005)
            
            all_ticker_returns.append(df['Strategy_Returns'].values)
        except Exception as e:
            continue

    if not all_ticker_returns:
        print("❌ No valid return data found.")
        return

    # 2. Run Monte Carlo
    # We will simulate 'NUM_SIMULATIONS' portfolios by randomly sampling returns from our ticker pool
    # This simulates "What if I chose a different set of 100 stocks from the market?"
    
    simulation_results = []
    
    print(f"🎲 Running {NUM_SIMULATIONS} Monte Carlo simulations...")
    for _ in range(NUM_SIMULATIONS):
        # Randomly sample 100 return streams with replacement (or size of our pool)
        indices = np.random.choice(len(all_ticker_returns), size=100, replace=True)
        sampled_returns = [all_ticker_returns[i] for i in indices]
        
        # Align lengths (min length to keep it fair)
        min_len = min(len(r) for r in sampled_returns)
        sampled_returns = [r[:min_len] for r in sampled_returns]
        
        # Average return across the 100 sampled tickers
        portfolio_daily_returns = np.mean(sampled_returns, axis=0)
        
        # Calculate Total Net Profit
        total_gross_return = np.prod(1 + portfolio_daily_returns) - 1
        gross_profit = INITIAL_CAPITAL * total_gross_return
        tax_drag = max(0, gross_profit * TAX_RATE)
        net_profit = gross_profit - tax_drag
        monthly_income = net_profit / MONTHS_IN_SIM
        
        simulation_results.append(monthly_income)

    # 3. Analyze & Visualize
    sim_array = np.array(simulation_results)
    prob_of_success = np.sum(sim_array >= GOAL_MONTHLY) / NUM_SIMULATIONS * 100
    p10 = np.percentile(sim_array, 10) # 90% chance it's better than this (Worst Case)
    p50 = np.percentile(sim_array, 50) # Median
    p90 = np.percentile(sim_array, 90) # Best Case
    
    plt.figure(figsize=(10, 6))
    sns.histplot(sim_array, kde=True, color='skyblue')
    plt.axvline(GOAL_MONTHLY, color='red', linestyle='--', label=f'Goal (${GOAL_MONTHLY})')
    plt.axvline(p10, color='orange', linestyle=':', label=f'P10: ${p10:,.0f} (90% Prob)')
    plt.title(f"Monte Carlo: Monthly Income Probability\nSuccess Probability: {prob_of_success:.1f}%")
    plt.xlabel("Monthly Income ($)")
    plt.ylabel("Frequency")
    plt.legend()
    plt.savefig("monte_carlo_results.png")
    
    print("\n--- Monte Carlo Results ---")
    print(f"Probability of hitting ${GOAL_MONTHLY}/mo: {prob_of_success:.1f}%")
    print(f"90% Confidence Floor (P10): ${p10:,.2f}")
    print(f"Median Expected (P50):     ${p50:,.2f}")
    print(f"High Performance (P90):    ${p90:,.2f}")
    print("\n📊 Results saved to monte_carlo_results.png")

if __name__ == "__main__":
    run_monte_carlo()
