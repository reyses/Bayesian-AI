import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

def run_mtf_analysis():
    print("Loading 5s Order Flow Delta dataset...")
    df = pd.read_parquet("DATA/ATLAS/order_flow_delta_5s.parquet")
    
    if df.index.tz is None:
        df.index = df.index.tz_localize('UTC')

    # We will resample to these timeframes
    timeframes = ['5s', '15s', '1min', '5min', '15min', '1h']
    
    results = []

    print("Running Multi-Timeframe Correlation Analysis...")
    for tf in timeframes:
        if tf == '5S':
            resampled = df.copy()
        else:
            resampled = df.resample(tf).agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum',
                'delta': 'sum'
            }).dropna()

        # Calculate features
        resampled['price_delta'] = resampled['close'] - resampled['open']
        
        resampled['wick_from_close'] = np.where(
            resampled['price_delta'] > 0,
            resampled['high'] - resampled['close'],
            resampled['close'] - resampled['low']
        )
        
        epsilon = 1e-8
        resampled['total_range'] = resampled['high'] - resampled['low']
        resampled['body_vol'] = (resampled['price_delta'] / (resampled['total_range'] + epsilon)) * resampled['volume']
        resampled['wick_close_vol'] = (resampled['wick_from_close'] / (resampled['total_range'] + epsilon)) * resampled['volume']

        cols = ['delta', 'body_vol', 'wick_close_vol', 'volume']
        df_clean = resampled.dropna(subset=cols).copy()
        df_clean = df_clean[df_clean['volume'] > 0]

        if len(df_clean) > 100:
            corr_matrix = df_clean[cols].corr()
            body_corr = corr_matrix.loc['body_vol', 'delta']
            wick_corr = corr_matrix.loc['wick_close_vol', 'delta']
            
            results.append({
                'Timeframe': tf,
                'N_Bars': len(df_clean),
                'Body Volume (Aggressive Flow Proxy)': body_corr,
                'Wick from Close Volume (Absorption Proxy)': wick_corr
            })

    results_df = pd.DataFrame(results).set_index('Timeframe')
    
    print("\n--- Multi-Timeframe Correlation with True Delta ---")
    print(results_df.to_string())

    # Save to report
    os.makedirs('research/order_flow_ablation/reports', exist_ok=True)
    
    plt.figure(figsize=(10, 6))
    results_df[['Body Volume (Aggressive Flow Proxy)', 'Wick from Close Volume (Absorption Proxy)']].plot(kind='bar', color=['#3498db', '#e74c3c'])
    plt.axhline(0, color='black', linewidth=1)
    plt.title('Multi-Timeframe Correlation with True Delta')
    plt.ylabel('Pearson Correlation (r)')
    plt.xticks(rotation=45)
    plt.legend(title='Engineered Feature')
    plt.tight_layout()
    plt.savefig('research/order_flow_ablation/reports/mtf_correlation_chart.png')
    
    print("\nChart saved to mtf_correlation_chart.png")

if __name__ == "__main__":
    run_mtf_analysis()
