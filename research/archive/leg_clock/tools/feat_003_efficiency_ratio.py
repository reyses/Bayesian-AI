"""Feature 003: Efficiency Ratio
Calculate Efficiency Ratio (net move / path length) over short, mid, and long 
windows to distinguish oscillation (low ER) from trend (high ER).
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

def calc_er(close, window):
    """Calculates Kaufman's Efficiency Ratio."""
    n = len(close)
    er = np.zeros(n, dtype=float)
    if n <= window: return er
    
    # Path length
    diffs = np.abs(np.diff(close))
    path_len = np.zeros(n)
    path_len[1:] = diffs
    rolling_path = pd.Series(path_len).rolling(window=window, min_periods=window).sum().values
    
    # Net move
    net_move = np.zeros(n)
    net_move[window:] = np.abs(close[window:] - close[:-window])
    
    # Avoid division by zero
    valid = (rolling_path > 0)
    er[valid] = net_move[valid] / rolling_path[valid]
    
    return er

def main():
    files = sorted(glob.glob(os.path.join(_REPO, 'DATA', 'ai_cusp_picks', '*_multi.json')))
    features = []
    
    for f in files:
        day_str = os.path.basename(f).replace('ai_picks_', '').replace('_multi.json', '').replace('-', '_')
        df = load_atlas(day_str, '1m')
        if df is None: continue
        
        ts = df['timestamp'].values
        close = df['close'].values
        
        er_5m = calc_er(close, 5)
        er_15m = calc_er(close, 15)
        er_60m = calc_er(close, 60)
        
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
            if idx < 60: continue # Need at least 60 bars for er_60m
            
            features.append({
                'is_label': 1,
                'er_5m': er_5m[idx],
                'er_15m': er_15m[idx],
                'er_60m': er_60m[idx]
            })
            
            null_candidates = np.where((df_hours == h) & (ts <= entry_ts) & (np.arange(len(ts)) >= 60))[0]
            null_candidates = [c for c in null_candidates if abs(c - idx) > 5]
            
            if len(null_candidates) > 0:
                null_idx = np.random.choice(null_candidates)
                features.append({
                    'is_label': 0,
                    'er_5m': er_5m[null_idx],
                    'er_15m': er_15m[null_idx],
                    'er_60m': er_60m[null_idx]
                })

    df_feat = pd.DataFrame(features)
    
    print("=== Feature 003: Efficiency Ratio ===")
    print(f"Total samples: {len(df_feat)} (50% labels, 50% nulls)")
    
    for col in ['er_5m', 'er_15m', 'er_60m']:
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
