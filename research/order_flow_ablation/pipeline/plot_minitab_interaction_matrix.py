import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

def create_minitab_interaction_matrix():
    print("Loading datasets...")
    baseline_df = pd.read_parquet("DATA/ATLAS/baseline_features_416D.parquet")
    delta_df = pd.read_parquet("DATA/ATLAS/order_flow_delta_5s.parquet")

    if baseline_df.index.tz is None and delta_df.index.tz is not None:
        baseline_df.index = baseline_df.index.tz_localize('UTC')
    elif baseline_df.index.tz is not None and delta_df.index.tz is None:
        delta_df.index = delta_df.index.tz_localize('UTC')

    merged = baseline_df.join(delta_df[['delta', 'volume', 'open', 'high', 'low']], how='inner', rsuffix='_trade')

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

    # Unified Wick Signals based on True Delta direction
    merged['opposing_wick_ratio'] = np.where(
        merged['delta'] > 0, 
        merged['upper_wick_ratio'], 
        merged['lower_wick_ratio']
    )
    
    merged['confirming_wick_ratio'] = np.where(
        merged['delta'] > 0,
        merged['lower_wick_ratio'],
        merged['upper_wick_ratio']
    )

    merged['fwd_ret_5m'] = (merged['close'].shift(-60) - merged['close']) * 10000

    cols_to_keep = ['delta', 'facsimile', 'opposing_wick_ratio', 'confirming_wick_ratio', 'fwd_ret_5m', 'volume']
    df_clean = merged.dropna(subset=cols_to_keep).copy()
    
    # Filter 0 volume
    df_clean = df_clean[df_clean['volume'] > 0]
    
    # Filter out extreme outliers in target to not corrupt means
    p_low = df_clean['fwd_ret_5m'].quantile(0.001)
    p_high = df_clean['fwd_ret_5m'].quantile(0.999)
    df_clean = df_clean[(df_clean['fwd_ret_5m'] >= p_low) & (df_clean['fwd_ret_5m'] <= p_high)]

    print("Discretizing factors into Tertiles (Low, Mid, High)...")
    factors = {
        'delta': 'True Delta',
        'facsimile': 'Facsimile',
        'opposing_wick_ratio': 'Opposing Wick',
        'confirming_wick_ratio': 'Confirming Wick'
    }
    
    for col, nice_name in factors.items():
        # qcut for equal sized bins
        df_clean[nice_name] = pd.qcut(df_clean[col], q=3, labels=['Low', 'Mid', 'High'], duplicates='drop')
        
    factor_names = list(factors.values())
    k = len(factor_names)
    
    print("Generating Minitab-style Interaction Plot Matrix...")
    fig, axes = plt.subplots(k, k, figsize=(14, 14), sharey=True)
    fig.suptitle('Interaction Plot for Means\nData Means (Forward 5m Return)', fontsize=18, y=0.98)
    
    # Colors and markers to mimic Minitab
    colors = ['#1f77b4', '#d62728', '#2ca02c']
    markers = ['o', 's', 'D']
    linestyles = ['-', '--', '-.']
    levels = ['Low', 'Mid', 'High']
    
    for i, row_factor in enumerate(factor_names):
        for j, col_factor in enumerate(factor_names):
            ax = axes[i, j]
            
            # Minitab Interaction Plot Matrix is usually empty on the diagonal
            # We will follow the upper or lower triangle layout or fill it all (except diagonal)
            if i == j:
                # Diagonal - just put the label
                ax.text(0.5, 0.5, row_factor, ha='center', va='center', fontsize=14, fontweight='bold')
                ax.set_xticks([])
                ax.set_yticks([])
                # Remove border
                for spine in ax.spines.values():
                    spine.set_visible(False)
                # Keep background clean
                ax.set_facecolor('whitesmoke')
            else:
                # Off-diagonal: Interaction Plot
                # x-axis is col_factor, hue is row_factor
                means = df_clean.groupby([col_factor, row_factor])['fwd_ret_5m'].mean().unstack()
                
                # Plot each line for the row_factor (hue)
                for hue_idx, hue_val in enumerate(levels):
                    if hue_val in means.columns:
                        y_vals = means[hue_val]
                        x_vals = range(len(y_vals))
                        ax.plot(x_vals, y_vals, marker=markers[hue_idx], color=colors[hue_idx], 
                                linestyle=linestyles[hue_idx], markersize=6, label=f"{row_factor}: {hue_val}")
                
                ax.set_xticks(range(len(levels)))
                
                # Only show x tick labels on the bottom row (or top)
                if i == k - 1 or (i == k-2 and j == k-1): # Bottom edge handling
                    ax.set_xticklabels(levels, rotation=45)
                else:
                    ax.set_xticklabels([])
                    
                ax.grid(True, alpha=0.3)
                
                # Setup legend for the rightmost plots
                if j == k - 1 and i != j:
                    ax.legend(title=row_factor, loc='center left', bbox_to_anchor=(1, 0.5), fontsize=9)
    
    plt.tight_layout(rect=[0, 0, 0.9, 0.96])
    
    os.makedirs('research/order_flow_ablation/reports', exist_ok=True)
    out_path = 'research/order_flow_ablation/reports/minitab_interaction_matrix.png'
    plt.savefig(out_path, bbox_inches='tight', dpi=150, facecolor='whitesmoke')
    print(f"Plot saved to {out_path}")

if __name__ == "__main__":
    create_minitab_interaction_matrix()
