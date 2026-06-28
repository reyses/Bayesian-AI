import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os

def run_wick_analysis():
    print("Loading datasets...")
    baseline_df = pd.read_parquet("DATA/ATLAS/baseline_features_416D.parquet")
    delta_df = pd.read_parquet("DATA/ATLAS/order_flow_delta_5s.parquet")

    if baseline_df.index.tz is None and delta_df.index.tz is not None:
        baseline_df.index = baseline_df.index.tz_localize('UTC')
    elif baseline_df.index.tz is not None and delta_df.index.tz is None:
        delta_df.index = delta_df.index.tz_localize('UTC')

    merged = baseline_df.join(delta_df[['delta', 'volume', 'open', 'high', 'low', 'close']], how='inner', rsuffix='_trade')

    print("Calculating Proxies and Wicks...")
    merged['price_delta'] = merged['close_trade'] - merged['open']
    merged['facsimile'] = np.sign(merged['price_delta']) * merged['volume']
    
    # Wick calculations
    merged['upper_wick'] = merged['high'] - np.maximum(merged['open'], merged['close_trade'])
    merged['lower_wick'] = np.minimum(merged['open'], merged['close_trade']) - merged['low']
    merged['body_size'] = np.abs(merged['price_delta'])
    merged['total_range'] = merged['high'] - merged['low']
    
    epsilon = 1e-8
    merged['upper_wick_ratio'] = merged['upper_wick'] / (merged['total_range'] + epsilon)
    merged['lower_wick_ratio'] = merged['lower_wick'] / (merged['total_range'] + epsilon)

    # Unified Wick Signals based on True Delta direction
    # Opposing wick: Upper wick if Delta > 0, Lower wick if Delta < 0
    merged['opposing_wick_ratio'] = np.where(
        merged['delta'] > 0, 
        merged['upper_wick_ratio'], 
        merged['lower_wick_ratio']
    )
    
    # Confirming wick: Lower wick if Delta > 0, Upper wick if Delta < 0
    merged['confirming_wick_ratio'] = np.where(
        merged['delta'] > 0,
        merged['lower_wick_ratio'],
        merged['upper_wick_ratio']
    )

    merged.dropna(subset=['delta', 'facsimile', 'volume', 'upper_wick'], inplace=True)
    merged = merged[merged['volume'] > 0].copy()
    
    # Quadrant logic
    merged['quadrant'] = 'Unknown'
    merged.loc[(merged['price_delta'] > 0) & (merged['delta'] > 0), 'quadrant'] = 'Confirm UP'
    merged.loc[(merged['price_delta'] > 0) & (merged['delta'] < 0), 'quadrant'] = 'Absorption UP'
    merged.loc[(merged['price_delta'] < 0) & (merged['delta'] < 0), 'quadrant'] = 'Confirm DN'
    merged.loc[(merged['price_delta'] < 0) & (merged['delta'] > 0), 'quadrant'] = 'Absorption DN'

    # Correlations
    cols_to_correlate = [
        'delta', 'facsimile', 'price_delta', 'volume', 
        'opposing_wick_ratio', 'confirming_wick_ratio', 'body_size'
    ]
    
    corr_matrix = merged[cols_to_correlate].corr()
    
    print("\n--- Correlation Matrix ---")
    print(corr_matrix[['delta', 'facsimile', 'price_delta']].to_string())
    
    print("\n--- Wick Stats by Quadrant ---")
    wick_stats = merged.groupby('quadrant')[['opposing_wick_ratio', 'confirming_wick_ratio', 'body_size']].mean()
    print(wick_stats)

    os.makedirs('research/order_flow_ablation/reports', exist_ok=True)
    
    # Save correlation heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", vmin=-1, vmax=1)
    plt.title('Correlation Matrix: True Delta, Facsimile, and Unified Wicks')
    plt.tight_layout()
    plt.savefig('research/order_flow_ablation/reports/wick_correlation_heatmap.png')
    
    # Save quadrant wick bar chart
    fig, ax = plt.subplots(figsize=(8, 6))
    wick_stats[['opposing_wick_ratio', 'confirming_wick_ratio']].plot(kind='bar', ax=ax, color=['#e74c3c', '#2ecc71'])
    ax.set_title('Unified Wick Ratios by Quadrant')
    ax.set_ylabel('Wick / Total Range')
    ax.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig('research/order_flow_ablation/reports/wick_quadrant_analysis.png')

if __name__ == "__main__":
    run_wick_analysis()
