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
    
    # Normalize wicks by total range to get 'wick ratio'
    # Adding epsilon to prevent division by zero
    epsilon = 1e-8
    merged['upper_wick_ratio'] = merged['upper_wick'] / (merged['total_range'] + epsilon)
    merged['lower_wick_ratio'] = merged['lower_wick'] / (merged['total_range'] + epsilon)

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
        'upper_wick', 'lower_wick', 'body_size', 'total_range',
        'upper_wick_ratio', 'lower_wick_ratio'
    ]
    
    corr_matrix = merged[cols_to_correlate].corr()
    
    print("\n--- Correlation Matrix ---")
    print(corr_matrix[['delta', 'facsimile', 'price_delta']].to_string())
    
    print("\n--- Wick Stats by Quadrant ---")
    wick_stats = merged.groupby('quadrant')[['upper_wick', 'lower_wick', 'upper_wick_ratio', 'lower_wick_ratio', 'body_size']].mean()
    print(wick_stats)

    os.makedirs('research/order_flow_ablation/reports', exist_ok=True)
    
    # Save correlation heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", vmin=-1, vmax=1)
    plt.title('Correlation Matrix: True Delta, Facsimile, and Wick Sizes')
    plt.tight_layout()
    plt.savefig('research/order_flow_ablation/reports/wick_correlation_heatmap.png')
    
    # Save quadrant wick bar chart
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    wick_stats[['upper_wick', 'lower_wick', 'body_size']].plot(kind='bar', ax=axes[0])
    axes[0].set_title('Absolute Wick & Body Size by Quadrant')
    axes[0].set_ylabel('Points')
    axes[0].tick_params(axis='x', rotation=45)
    
    wick_stats[['upper_wick_ratio', 'lower_wick_ratio']].plot(kind='bar', ax=axes[1])
    axes[1].set_title('Wick Ratio (Wick / Total Range) by Quadrant')
    axes[1].set_ylabel('Ratio')
    axes[1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig('research/order_flow_ablation/reports/wick_quadrant_analysis.png')

if __name__ == "__main__":
    run_wick_analysis()
