import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

def create_interaction_matrix():
    print("Loading datasets...")
    baseline_df = pd.read_parquet("DATA/ATLAS/baseline_features_416D.parquet")
    delta_df = pd.read_parquet("DATA/ATLAS/order_flow_delta_5s.parquet")

    if baseline_df.index.tz is None and delta_df.index.tz is not None:
        baseline_df.index = baseline_df.index.tz_localize('UTC')
    elif baseline_df.index.tz is not None and delta_df.index.tz is None:
        delta_df.index = delta_df.index.tz_localize('UTC')

    merged = baseline_df.join(delta_df[['delta', 'volume', 'open', 'high', 'low', 'close']], how='inner', rsuffix='_trade')

    print("Calculating Proxies and Wicks...")
    merged['price_delta'] = merged['close'] - merged['open']
    merged['facsimile'] = np.sign(merged['price_delta']) * merged['volume']
    
    # Wick calculations
    merged['upper_wick'] = merged['high'] - np.maximum(merged['open'], merged['close'])
    merged['lower_wick'] = np.minimum(merged['open'], merged['close']) - merged['low']
    merged['total_range'] = merged['high'] - merged['low']
    
    epsilon = 1e-8
    merged['upper_wick_ratio'] = merged['upper_wick'] / (merged['total_range'] + epsilon)
    merged['lower_wick_ratio'] = merged['lower_wick'] / (merged['total_range'] + epsilon)

    cols_to_plot = ['delta', 'facsimile', 'price_delta', 'upper_wick_ratio', 'lower_wick_ratio']
    df_clean = merged.dropna(subset=cols_to_plot).copy()
    
    # Filter out extreme volume/delta outliers for a cleaner scatter plot (99th percentile)
    for col in ['delta', 'facsimile', 'price_delta']:
        p_low = df_clean[col].quantile(0.01)
        p_high = df_clean[col].quantile(0.99)
        df_clean = df_clean[(df_clean[col] >= p_low) & (df_clean[col] <= p_high)]

    print(f"Data shape after cleaning: {df_clean.shape}")
    
    # Sample down to 20,000 points to prevent huge file sizes and slow rendering
    if len(df_clean) > 20000:
        df_sample = df_clean.sample(20000, random_state=42)
    else:
        df_sample = df_clean

    print("Generating Interaction Matrix (PairPlot)...")
    # Rename columns for prettier labels in the plot
    df_sample = df_sample.rename(columns={
        'delta': 'True Delta',
        'facsimile': 'Facsimile',
        'price_delta': 'Price Delta',
        'upper_wick_ratio': 'Upper Wick (Ratio)',
        'lower_wick_ratio': 'Lower Wick (Ratio)'
    })
    
    # We use a pairplot to show the scatter interaction matrix
    sns.set_theme(style="whitegrid")
    
    # Using 'hist' kind for dense data to clearly show the 2D distribution instead of an opaque blob
    g = sns.PairGrid(df_sample[['True Delta', 'Facsimile', 'Price Delta', 'Upper Wick (Ratio)', 'Lower Wick (Ratio)']])
    g.map_diag(sns.histplot, kde=True, color='#2c3e50')
    g.map_offdiag(sns.histplot, bins=30, pthresh=.05, cmap="mako")
    
    g.fig.suptitle('Continuous Interaction Matrix Plot (Distributions)', y=1.02, fontsize=16)
    
    os.makedirs('research/order_flow_ablation/reports', exist_ok=True)
    
    plt.savefig('research/order_flow_ablation/reports/interaction_matrix_plot.png', bbox_inches='tight', dpi=150)
    print("Interaction Matrix generated successfully.")

if __name__ == "__main__":
    create_interaction_matrix()
