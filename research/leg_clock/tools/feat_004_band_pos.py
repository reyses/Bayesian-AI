"""Feature 004: Band Position
Calculate the normalized position of price relative to the 1h OLS channels, 
aligned to trade direction intent.
"""
import glob
import json
import os
import sys
from datetime import datetime, timezone
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

warnings.filterwarnings('ignore', category=RuntimeWarning)

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, '..', '..', '..'))

# Import the rolling OLS bands
sys.path.append(os.path.join(_REPO, 'research', 'level_hold', 'tools'))
from level_hold_study import rolling_ols_bands

def load_atlas(day, tf='1m'):
    path = os.path.join(_REPO, 'DATA', 'ATLAS', tf, f"{day}.parquet")
    if not os.path.exists(path):
        return None
    return pd.read_parquet(path)

def main():
    files = sorted(glob.glob(os.path.join(_REPO, 'DATA', 'ai_cusp_picks', '*_multi.json')))
    features = []
    
    for f in files:
        day_str = os.path.basename(f).replace('ai_picks_', '').replace('_multi.json', '').replace('-', '_')
        df = load_atlas(day_str, '1m')
        if df is None: continue
        
        ts = df['timestamp'].values
        close = df['close'].values
        
        up, lo, _ = rolling_ols_bands(close, 60)
        
        band_width = up - lo
        # normalized position (0 = lower band, 1 = upper band)
        band_pos = (close - lo) / np.maximum(band_width, 1e-6)
        
        try:
            d = json.load(open(f))
        except Exception:
            continue
            
        trades = d.get('trades', [])
        hours = np.array([datetime.fromtimestamp(t['entry_ts'], tz=timezone.utc).hour for t in trades])
        df_hours = np.array([datetime.fromtimestamp(t, tz=timezone.utc).hour for t in ts])
        
        for t, h in zip(trades, hours):
            entry_ts = t['entry_ts']
            is_long = str(t.get('direction', '')).upper().startswith('L')
            
            idx = np.searchsorted(ts, entry_ts, side='right') - 1
            if idx < 60: continue
            
            # Align position: 
            # If long, 0 means lower band (reversal support). 
            # If short, 1 means lower band, so 1 - pos makes 0 the upper band (reversal resistance).
            aligned_pos = band_pos[idx] if is_long else (1.0 - band_pos[idx])
            
            features.append({
                'is_label': 1,
                'aligned_pos': aligned_pos
            })
            
            null_candidates = np.where((df_hours == h) & (ts <= entry_ts) & (np.arange(len(ts)) >= 60))[0]
            null_candidates = [c for c in null_candidates if abs(c - idx) > 5]
            
            if len(null_candidates) > 0:
                null_idx = np.random.choice(null_candidates)
                null_pos = band_pos[null_idx] if is_long else (1.0 - band_pos[null_idx])
                
                features.append({
                    'is_label': 0,
                    'aligned_pos': null_pos
                })

    df_feat = pd.DataFrame(features)
    
    print("=== Feature 004: OLS Band Position ===")
    print(f"Total samples: {len(df_feat)} (50% labels, 50% nulls)")
    
    for col in ['aligned_pos']:
        labels = df_feat[df_feat['is_label'] == 1][col].values
        nulls = df_feat[df_feat['is_label'] == 0][col].values
        
        mean_diff = np.mean(labels) - np.mean(nulls)
        
        pool = np.concatenate([labels, nulls])
        n_l = len(labels)
        diffs = []
        for _ in range(1000):
            np.random.shuffle(pool)
            diffs.append(np.mean(pool[:n_l]) - np.mean(pool[n_l:]))
        p_val = np.mean(np.abs(diffs) >= np.abs(mean_diff))
        
        auc = roc_auc_score(df_feat['is_label'], df_feat[col])
        if auc < 0.5: auc = 1.0 - auc
        gap = auc - 0.5
        
        signal = "NOISE"
        if gap >= 0.10: signal = "REAL"
        elif gap >= 0.05: signal = "CONDITIONAL"
            
        print(f"\nFeature: {col}")
        print(f"Label Mean: {np.mean(labels):.3f} | Null Mean: {np.mean(nulls):.3f}")
        print(f"Mean Diff: {mean_diff:.3f} (p={p_val:.4f})")
        print(f"AUC: {auc:.3f} | Gap: {gap:.3f} -> {signal}")

if __name__ == '__main__':
    main()
