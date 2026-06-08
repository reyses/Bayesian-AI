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
    
    ohlcv['fwd_return'] = ohlcv['close'].shift(-30) - ohlcv['close']
    
    feat_x = 'L3_5s_z_low_9'
    feat_color = 'L3_5s_reversion_prob_9'
    
    df['fwd_return'] = ohlcv['fwd_return'].values
    valid = df[[feat_x, feat_color, 'fwd_return']].dropna()
    
    # We discretize the color feature into 3 bins just to have 3 distinct lines
    try:
        valid[feat_color + '_bin'] = pd.qcut(valid[feat_color], q=3, labels=['Low', 'Mid', 'High'], duplicates='drop')
    except ValueError:
        valid[feat_color + '_bin'] = pd.qcut(valid[feat_color].rank(method='first'), q=3, labels=['Low', 'Mid', 'High'])
        
    print(f"[INFO] Fitting Quadratic Curves (order=2)...")
    
    g = sns.lmplot(
        data=valid, 
        x=feat_x, 
        y='fwd_return', 
        hue=feat_color + '_bin',
        order=2, # THIS MAKES IT A QUADRATIC CURVE
        scatter_kws={'alpha': 0.05, 's': 5}, 
        line_kws={'linewidth': 3},
        palette=["#1f77b4", "#ff7f0e", "#2ca02c"],
        height=7,
        aspect=1.4,
        legend_out=False
    )
    
    plt.title(f"Continuous Quadratic Interaction Plot\n{feat_x} vs Forward Return (Contingent on {feat_color})", fontsize=14)
    plt.xlabel(feat_x, fontsize=12)
    plt.ylabel("Forward 30-Bar Return (ticks)", fontsize=12)
    plt.axhline(0, color='black', linewidth=1, linestyle='--')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('artifacts/curved_interaction_plot.png', dpi=150)
    print("[INFO] Saved interaction plot to artifacts/curved_interaction_plot.png")

if __name__ == "__main__":
    main()
