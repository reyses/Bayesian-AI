import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

def create_main_effects_plot():
    print("Loading datasets...")
    baseline_df = pd.read_parquet("DATA/ATLAS/baseline_features_416D.parquet")
    delta_df = pd.read_parquet("DATA/ATLAS/order_flow_delta_5s.parquet")

    if baseline_df.index.tz is None and delta_df.index.tz is not None:
        baseline_df.index = baseline_df.index.tz_localize('UTC')
    elif baseline_df.index.tz is not None and delta_df.index.tz is None:
        delta_df.index = delta_df.index.tz_localize('UTC')

    merged = baseline_df.join(delta_df[['delta', 'volume', 'open']], how='inner', rsuffix='_trade')

    print("Calculating...")
    merged['price_delta'] = merged['close'] - merged['open']
    
    # Calculate signs
    merged['True_Delta_Sign'] = np.sign(merged['delta'])
    merged['Price_Move_Sign'] = np.sign(merged['price_delta'])
    
    # Forward 5m return scaled (same as stage_0C)
    merged['fwd_ret_5m'] = (merged['close'].shift(-60) - merged['close']) * 10000
    
    # Drop NaNs and zeros
    df_clean = merged.dropna(subset=['True_Delta_Sign', 'Price_Move_Sign', 'fwd_ret_5m']).copy()
    df_clean = df_clean[(df_clean['True_Delta_Sign'] != 0) & (df_clean['Price_Move_Sign'] != 0) & (df_clean['volume'] > 0)]
    
    # Filter out extreme outliers that might corrupt the mean (e.g. shifting across weekend gaps)
    # 99.9th percentile to remove gap artifacts
    p_low = df_clean['fwd_ret_5m'].quantile(0.001)
    p_high = df_clean['fwd_ret_5m'].quantile(0.999)
    df_clean = df_clean[(df_clean['fwd_ret_5m'] >= p_low) & (df_clean['fwd_ret_5m'] <= p_high)]
    
    # Map to readable labels
    df_clean['True_Delta'] = df_clean['True_Delta_Sign'].map({1.0: 'Buying (+)', -1.0: 'Selling (-)'})
    df_clean['Price_Move'] = df_clean['Price_Move_Sign'].map({1.0: 'UP (+)', -1.0: 'DOWN (-)'})
    
    # Overall mean
    overall_mean = df_clean['fwd_ret_5m'].mean()
    
    # Means for Main Effects
    mean_by_delta = df_clean.groupby('True_Delta')['fwd_ret_5m'].mean().reindex(['Selling (-)', 'Buying (+)'])
    mean_by_price = df_clean.groupby('Price_Move')['fwd_ret_5m'].mean().reindex(['DOWN (-)', 'UP (+)'])
    
    os.makedirs('research/order_flow_ablation/reports', exist_ok=True)
    
    # 1. Main Effects Plot (Minitab style)
    fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharey=True)
    fig.suptitle('Main Effects Plot for Forward 5m Return\nData Means', fontsize=14, y=1.05)
    
    # Plot True Delta Effect
    axes[0].plot(mean_by_delta.index, mean_by_delta.values, marker='o', color='#1f77b4', linestyle='-', markersize=8)
    axes[0].axhline(overall_mean, color='gray', linestyle='--', linewidth=1)
    axes[0].set_title('True Delta (Aggressive Flow)')
    axes[0].set_ylabel('Mean Forward Return (Scaled)')
    axes[0].grid(True, alpha=0.3)
    
    # Plot Price Move Effect
    axes[1].plot(mean_by_price.index, mean_by_price.values, marker='o', color='#1f77b4', linestyle='-', markersize=8)
    axes[1].axhline(overall_mean, color='gray', linestyle='--', linewidth=1)
    axes[1].set_title('Price Move (Close - Open)')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('research/order_flow_ablation/reports/main_effects_plot.png', bbox_inches='tight')
    
    # 2. Interaction Plot (Crucial for showing absorption)
    plt.figure(figsize=(8, 6))
    sns.pointplot(data=df_clean, x='Price_Move', y='fwd_ret_5m', hue='True_Delta', 
                  order=['DOWN (-)', 'UP (+)'], hue_order=['Selling (-)', 'Buying (+)'],
                  markers=['o', 's'], linestyles=['-', '--'])
    
    plt.title('Interaction Plot for Forward 5m Return\n(True Delta x Price Move)')
    plt.axhline(overall_mean, color='gray', linestyle='--', linewidth=1)
    plt.ylabel('Mean Forward Return (Scaled)')
    plt.xlabel('Price Move (Close - Open)')
    plt.legend(title='True Delta')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('research/order_flow_ablation/reports/interaction_plot.png', bbox_inches='tight')
    
    print("Plots generated successfully.")

if __name__ == "__main__":
    create_main_effects_plot()
