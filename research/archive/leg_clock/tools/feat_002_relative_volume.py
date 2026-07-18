"""Feature 002: Relative Volume Rate
Calculate volume as a rate (contracts/sec) relative to short/long-window 
acceleration, and test distinctness vs a time-matched NULL.
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

def load_atlas(day, tf='1m'):
    path = os.path.join(_REPO, 'DATA', 'ATLAS', tf, f"{day}.parquet")
    if not os.path.exists(path):
        return None
    return pd.read_parquet(path)

def calc_volume_features(df):
    """Calculates causal volume features on the dataframe."""
    # Base rate: contracts per second (1m bars = 60s)
    vol_rate = df['volume'] / 60.0
    
    # Normal volume (rolling 60 min mean)
    vol_normal = vol_rate.rolling(window=60, min_periods=10).mean()
    
    # Short and long windows
    vol_short = vol_rate.rolling(window=5, min_periods=1).mean()
    vol_long = vol_rate.rolling(window=15, min_periods=3).mean()
    
    # Accel ratios (handle division by zero and NaNs)
    accel_short = vol_short / vol_normal.replace(0, 1)
    accel_long = vol_long / vol_normal.replace(0, 1)
    
    return vol_rate.values, accel_short.values, accel_long.values

def main():
    files = sorted(glob.glob(os.path.join(_REPO, 'DATA', 'ai_cusp_picks', '*_multi.json')))
    features = []
    
    for f in files:
        day_str = os.path.basename(f).replace('ai_picks_', '').replace('_multi.json', '').replace('-', '_')
        df = load_atlas(day_str, '1m')
        if df is None: continue
        
        ts = df['timestamp'].values
        vol_rate, accel_short, accel_long = calc_volume_features(df)
        
        try:
            d = json.load(open(f))
        except Exception:
            continue
            
        trades = d.get('trades', [])
        hours = np.array([datetime.fromtimestamp(t['entry_ts'], tz=timezone.utc).hour for t in trades])
        df_hours = np.array([datetime.fromtimestamp(t, tz=timezone.utc).hour for t in ts])
        
        for t, h in zip(trades, hours):
            entry_ts = t['entry_ts']
            
            idx = np.searchsorted(ts, entry_ts, side='right') - 1
            if idx < 0 or np.isnan(accel_short[idx]): continue
            
            features.append({
                'is_label': 1,
                'vol_rate': vol_rate[idx],
                'accel_short': accel_short[idx],
                'accel_long': accel_long[idx]
            })
            
            null_candidates = np.where((df_hours == h) & (ts <= entry_ts) & ~np.isnan(accel_short))[0]
            null_candidates = [c for c in null_candidates if abs(c - idx) > 5]
            
            if len(null_candidates) > 0:
                null_idx = np.random.choice(null_candidates)
                features.append({
                    'is_label': 0,
                    'vol_rate': vol_rate[null_idx],
                    'accel_short': accel_short[null_idx],
                    'accel_long': accel_long[null_idx]
                })

    df_feat = pd.DataFrame(features)
    df_feat = df_feat.fillna(0.0) # Safety net
    
    print("=== Feature 002: Relative Volume Rate ===")
    print(f"Total samples: {len(df_feat)} (50% labels, 50% nulls)")
    
    for col in ['vol_rate', 'accel_short', 'accel_long']:
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
