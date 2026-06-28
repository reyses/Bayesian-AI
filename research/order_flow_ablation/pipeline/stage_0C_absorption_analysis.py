import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns

def run_absorption_analysis():
    print("Loading datasets...")
    baseline_df = pd.read_parquet("DATA/ATLAS/baseline_features_416D.parquet")
    delta_df = pd.read_parquet("DATA/ATLAS/order_flow_delta_5s.parquet")

    if baseline_df.index.tz is None and delta_df.index.tz is not None:
        baseline_df.index = baseline_df.index.tz_localize('UTC')
    elif baseline_df.index.tz is not None and delta_df.index.tz is None:
        delta_df.index = delta_df.index.tz_localize('UTC')

    merged = baseline_df.join(delta_df[['delta', 'volume', 'open']], how='inner', rsuffix='_trade')

    print("Calculating Proxies...")
    merged['price_delta'] = merged['close'] - merged['open']
    merged['facsimile'] = np.sign(merged['price_delta']) * merged['volume']

    # Filter out NaNs and zero-volume bars to prevent division by zero
    merged.dropna(subset=['delta', 'facsimile', 'volume'], inplace=True)
    merged = merged[merged['volume'] > 0].copy()

    # Calculate Absorption (Divergence)
    # If the bar moves UP (facsimile > 0) but the delta is negative (delta < 0), that is absorption.
    merged['absorption_flag'] = np.sign(merged['delta']) != np.sign(merged['facsimile'])
    
    # Calculate Magnitude of Disagreement (delta relative to total volume)
    merged['delta_ratio'] = merged['delta'] / merged['volume']
    
    # Calculate future returns
    merged['fwd_ret_1m'] = merged['close'].shift(-12) - merged['close'] # 1m future (12 * 5s = 60s)
    merged['fwd_ret_5m'] = merged['close'].shift(-60) - merged['close'] # 5m future
    
    merged.dropna(subset=['fwd_ret_1m', 'fwd_ret_5m'], inplace=True)
    
    # Split into 4 Quadrants:
    # 1. Trend Confirm UP (Price UP, Delta UP)
    # 2. Absorption UP (Price UP, Delta DN) -> Passive Sellers absorbing aggressive buyers
    # 3. Trend Confirm DN (Price DN, Delta DN)
    # 4. Absorption DN (Price DN, Delta UP) -> Passive Buyers absorbing aggressive sellers
    
    merged['quadrant'] = 'Unknown'
    merged.loc[(merged['price_delta'] > 0) & (merged['delta'] > 0), 'quadrant'] = 'Confirm UP'
    merged.loc[(merged['price_delta'] > 0) & (merged['delta'] < 0), 'quadrant'] = 'Absorption UP'
    merged.loc[(merged['price_delta'] < 0) & (merged['delta'] < 0), 'quadrant'] = 'Confirm DN'
    merged.loc[(merged['price_delta'] < 0) & (merged['delta'] > 0), 'quadrant'] = 'Absorption DN'
    
    print("\nQuadrant Distribution:")
    print(merged['quadrant'].value_counts(normalize=True) * 100)
    
    print("\nFuture Returns by Quadrant (1m):")
    stats_1m = merged.groupby('quadrant')['fwd_ret_1m'].mean() * 10000 # in bps / ticks roughly
    print(stats_1m)

    print("\nFuture Returns by Quadrant (5m):")
    stats_5m = merged.groupby('quadrant')['fwd_ret_5m'].mean() * 10000
    print(stats_5m)

    # Plotting
    os.makedirs('research/order_flow_ablation/reports', exist_ok=True)
    
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='quadrant', y='fwd_ret_5m', data=merged, showfliers=False, order=['Confirm UP', 'Absorption UP', 'Confirm DN', 'Absorption DN'])
    plt.title('Future 5m Price Movement by Order Flow Quadrant')
    plt.ylabel('Forward 5m Return (Price)')
    plt.axhline(0, color='black', linestyle='--')
    plt.tight_layout()
    plt.savefig('research/order_flow_ablation/reports/quadrant_returns.png')
    
    plt.figure(figsize=(8, 6))
    counts = merged['quadrant'].value_counts()
    plt.pie(counts, labels=counts.index, autopct='%1.1f%%', colors=['#2ecc71', '#e74c3c', '#3498db', '#f1c40f'])
    plt.title('Frequency of Order Flow Regimes')
    plt.tight_layout()
    plt.savefig('research/order_flow_ablation/reports/quadrant_frequency.png')

    with open('research/order_flow_ablation/reports/absorption_analysis.md', 'w') as f:
        f.write("# Order Flow Absorption Analysis\n\n")
        f.write("Investigating the causal relationship between market trade volume (True Delta) and OHLCV volume with the open-close sign (Facsimile).\n\n")
        f.write("## Quadrant Frequency\n")
        f.write("```text\n")
        f.write(str(merged['quadrant'].value_counts(normalize=True) * 100) + "\n")
        f.write("```\n\n")
        f.write("## Forward Returns (5m)\n")
        f.write("```text\n")
        f.write(str(stats_5m) + "\n")
        f.write("```\n\n")
        f.write("![Quadrant Frequency](quadrant_frequency.png)\n\n")
        f.write("![Quadrant Returns](quadrant_returns.png)\n")

if __name__ == "__main__":
    run_absorption_analysis()
