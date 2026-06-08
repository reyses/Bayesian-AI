import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from core_v2.features import load_features

def main():
    day = '2025_02_05'
    atlas_root = "C:/Users/reyse/OneDrive/Desktop/Bayesian-AI/DATA/ATLAS"
    features_root = os.path.join(atlas_root, 'FEATURES_5s_v2')
    
    print(f"[INFO] Loading features for {day}...")
    df = load_features([day], root=features_root)
    ohlcv = pd.read_parquet(os.path.join(atlas_root, '5s', f'{day}.parquet'))
    
    min_len = min(len(df), len(ohlcv))
    df = df.iloc[:min_len].copy()
    ohlcv = ohlcv.iloc[:min_len].copy()
    
    # We want to measure the forward 30-bar return (since SEED_BARS=30)
    ohlcv['fwd_return'] = ohlcv['close'].shift(-30) - ohlcv['close']
    
    # We will pick two heavily interacting features found in our Pareto/Heatmap:
    feat_x = 'L3_5s_z_low_9'
    feat_color = 'L3_5s_reversion_prob_9'
    
    print(f"[INFO] Creating Minitab Interaction Plot for {feat_x} and {feat_color}")
    
    df['fwd_return'] = ohlcv['fwd_return'].values
    
    # Drop NaNs
    valid = df[[feat_x, feat_color, 'fwd_return']].dropna()
    
    # Discretize into Low, Mid, High (Tertiles)
    try:
        valid[feat_x + '_bin'] = pd.qcut(valid[feat_x], q=3, labels=['Low', 'Mid', 'High'], duplicates='drop')
        valid[feat_color + '_bin'] = pd.qcut(valid[feat_color], q=3, labels=['Low', 'Mid', 'High'], duplicates='drop')
    except ValueError:
        print("[WARNING] Could not split into exact tertiles due to duplicates. Using rank-based.")
        valid[feat_x + '_bin'] = pd.qcut(valid[feat_x].rank(method='first'), q=3, labels=['Low', 'Mid', 'High'])
        valid[feat_color + '_bin'] = pd.qcut(valid[feat_color].rank(method='first'), q=3, labels=['Low', 'Mid', 'High'])
        
    # Aggregate means
    agg = valid.groupby([feat_x + '_bin', feat_color + '_bin'])['fwd_return'].mean().reset_index()
    
    # Plot Minitab style interaction plot
    plt.figure(figsize=(10, 7))
    sns.pointplot(
        data=agg, 
        x=feat_x + '_bin', 
        y='fwd_return', 
        hue=feat_color + '_bin',
        palette=["#1f77b4", "#ff7f0e", "#2ca02c"],
        markers=["o", "s", "D"],
        linestyles=["-", "--", "-."]
    )
    
    plt.title(f"Minitab Interaction Plot\nMean Forward 30-Bar Return by {feat_x} & {feat_color}", fontsize=14)
    plt.xlabel(feat_x, fontsize=12)
    plt.ylabel("Mean Forward Return (ticks)", fontsize=12)
    plt.axhline(0, color='black', linewidth=1, linestyle='--')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('artifacts/minitab_interaction_plot.png', dpi=150)
    print("[INFO] Saved interaction plot to artifacts/minitab_interaction_plot.png")

if __name__ == "__main__":
    main()
