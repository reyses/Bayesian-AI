"""Feature 007: During-Trade Dynamics
Evaluates the Realized Velocity, MAE, and MFE of golden labels vs a random NULL 
trade simulated for the exact same duration.
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

def main():
    files = sorted(glob.glob(os.path.join(_REPO, 'DATA', 'ai_cusp_picks', '*_multi.json')))
    features = []
    
    for f in files:
        day_str = os.path.basename(f).replace('ai_picks_', '').replace('_multi.json', '').replace('-', '_')
        df = load_atlas(day_str, '1m')
        if df is None: continue
        
        ts = df['timestamp'].values
        hi = df['high'].values
        lo = df['low'].values
        
        try:
            d = json.load(open(f))
        except Exception:
            continue
            
        trades = d.get('trades', [])
        hours = np.array([datetime.fromtimestamp(t['entry_ts'], tz=timezone.utc).hour for t in trades])
        df_hours = np.array([datetime.fromtimestamp(t, tz=timezone.utc).hour for t in ts])
        
        for t, h in zip(trades, hours):
            entry_ts = t['entry_ts']
            exit_ts = t.get('exit_ts', entry_ts)
            
            is_long = str(t.get('direction', '')).upper().startswith('L')
            dur = exit_ts - entry_ts
            if dur <= 0: continue
            
            idx = np.searchsorted(ts, entry_ts, side='right') - 1
            if idx < 0: continue
            end_idx = np.searchsorted(ts, exit_ts, side='right') - 1
            if end_idx < idx: end_idx = idx
            
            dur_mins = max(dur / 60.0, 1.0)
            entry_px = (hi[idx] + lo[idx]) / 2.0
            
            window_hi = np.max(hi[idx:end_idx+1])
            window_lo = np.min(lo[idx:end_idx+1])
            
            if is_long:
                l_mfe = window_hi - entry_px
                l_mae = entry_px - window_lo
            else:
                l_mfe = entry_px - window_lo
                l_mae = window_hi - entry_px
                
            l_mfe = max(0, l_mfe) / 0.25
            l_mae = max(0, l_mae) / 0.25
            
            l_vel = l_mfe / dur_mins
            
            features.append({
                'is_label': 1,
                'mae': l_mae,
                'mfe': l_mfe,
                'velocity': l_vel,
                'duration': dur_mins
            })
            
            # Simulate random trade for same duration
            null_candidates = np.where((df_hours == h) & (ts <= entry_ts) & (np.arange(len(ts)) < len(ts) - (end_idx - idx) - 1))[0]
            
            if len(null_candidates) > 0:
                null_idx = np.random.choice(null_candidates)
                null_end_idx = min(null_idx + (end_idx - idx), len(ts) - 1)
                
                if null_end_idx >= null_idx:
                    n_entry_px = (hi[null_idx] + lo[null_idx]) / 2.0
                    
                    n_window_hi = np.max(hi[null_idx:null_end_idx+1])
                    n_window_lo = np.min(lo[null_idx:null_end_idx+1])
                    
                    if is_long:
                        n_mfe = n_window_hi - n_entry_px
                        n_mae = n_entry_px - n_window_lo
                    else:
                        n_mfe = n_entry_px - n_window_lo
                        n_mae = n_window_hi - n_entry_px
                        
                    n_mfe = max(0, n_mfe) / 0.25
                    n_mae = max(0, n_mae) / 0.25
                    
                    n_vel = n_mfe / dur_mins
                    
                    features.append({
                        'is_label': 0,
                        'mae': n_mae,
                        'mfe': n_mfe,
                        'velocity': n_vel,
                        'duration': dur_mins
                    })

    if not features:
        print("Total samples: 0 (50% labels, 50% nulls)")
        return
        
    df_feat = pd.DataFrame(features)
    
    print("=== Feature 007: During-Trade Dynamics ===")
    print(f"Total samples: {len(df_feat)} (50% labels, 50% nulls)")
    
    for col in ['mae', 'mfe', 'velocity', 'duration']:
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
        
        # For MAE, smaller is better (so label < null is the strong class).
        # We handle AUC inversion below:
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
