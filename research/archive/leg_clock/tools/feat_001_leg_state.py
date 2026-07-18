"""Feature 001: Leg State (1m ZigZag)
Calculate the causal state of the current 1m leg (direction, extent, velocity)
and test distinctness vs a time-matched NULL.
"""
import glob
import json
import os
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, '..', '..', '..'))
TICK = 0.25

def load_atlas(day, tf='1m'):
    path = os.path.join(_REPO, 'DATA', 'ATLAS', tf, f"{day}.parquet")
    if not os.path.exists(path):
        return None
    return pd.read_parquet(path)

def stream_zigzag_state(close, timestamps, thr_ticks=40):
    """Causally tracks the zigzag state. Returns arrays of leg_dir, leg_extent, leg_duration."""
    thr = thr_ticks * TICK
    n = len(close)
    
    # State arrays
    out_dir = np.zeros(n, dtype=int)
    out_ext = np.zeros(n, dtype=float)
    out_dur = np.zeros(n, dtype=float)
    
    if n == 0: return out_dir, out_ext, out_dur
    
    hi_i = lo_i = 0
    direction = 0
    ext_i = 0
    
    for i in range(1, n):
        c = close[i]
        
        # Initialization phase
        if direction == 0:
            if c > close[hi_i]: hi_i = i
            if c < close[lo_i]: lo_i = i
            if close[hi_i] - c >= thr:
                direction, ext_i = -1, (lo_i if lo_i > hi_i else i)
            elif c - close[lo_i] >= thr:
                direction, ext_i = 1, (hi_i if hi_i > lo_i else i)
        
        # Tracking phase
        else:
            if direction > 0:
                if c > close[ext_i]:
                    ext_i = i
                elif close[ext_i] - c >= thr:
                    direction, ext_i = -1, i
            else:
                if c < close[ext_i]:
                    ext_i = i
                elif c - close[ext_i] >= thr:
                    direction, ext_i = 1, i
        
        if direction != 0:
            out_dir[i] = direction
            out_ext[i] = abs(c - close[ext_i]) / TICK
            dur_mins = (timestamps[i] - timestamps[ext_i]) / 60.0
            out_dur[i] = max(1.0, dur_mins)
            
    return out_dir, out_ext, out_dur

def main():
    files = sorted(glob.glob(os.path.join(_REPO, 'DATA', 'ai_cusp_picks', '*_multi.json')))
    
    features = []
    
    for f in files:
        day_str = os.path.basename(f).replace('ai_picks_', '').replace('_multi.json', '').replace('-', '_')
        df = load_atlas(day_str, '1m')
        if df is None: continue
        
        ts = df['timestamp'].values
        close = df['close'].values
        
        # Causal leg state
        leg_dir, leg_ext, leg_dur = stream_zigzag_state(close, ts, thr_ticks=40)
        leg_vel = leg_ext / np.maximum(leg_dur, 1.0)
        
        try:
            d = json.load(open(f))
        except Exception:
            continue
            
        trades = d.get('trades', [])
        # Get hours for null matching
        hours = np.array([datetime.fromtimestamp(t['entry_ts'], tz=timezone.utc).hour for t in trades])
        
        df_hours = np.array([datetime.fromtimestamp(t, tz=timezone.utc).hour for t in ts])
        
        for t, h in zip(trades, hours):
            entry_ts = t['entry_ts']
            is_long = str(t.get('direction', '')).upper().startswith('L')
            
            # Find exact bar index <= entry_ts
            idx = np.searchsorted(ts, entry_ts, side='right') - 1
            if idx < 0: continue
            
            # Extract features for label
            features.append({
                'is_label': 1,
                'is_long_trade': 1 if is_long else 0,
                'leg_dir': leg_dir[idx],
                'leg_ext': leg_ext[idx],
                'leg_vel': leg_vel[idx]
            })
            
            # Construct matched null
            null_candidates = np.where((df_hours == h) & (ts <= entry_ts))[0]
            # Exclude the actual label index and immediate surrounding
            null_candidates = [c for c in null_candidates if abs(c - idx) > 5]
            
            if len(null_candidates) > 0:
                null_idx = np.random.choice(null_candidates)
                features.append({
                    'is_label': 0,
                    'is_long_trade': 1 if is_long else 0, # nulls are matched to trade direction intent
                    'leg_dir': leg_dir[null_idx],
                    'leg_ext': leg_ext[null_idx],
                    'leg_vel': leg_vel[null_idx]
                })

    df_feat = pd.DataFrame(features)
    
    # Analyze distinctness
    print("=== Feature 001: 1m Leg State (40-tick zigzag) ===")
    print(f"Total samples: {len(df_feat)} (50% labels, 50% nulls)")
    
    # Alignment: -1 means leg was going DOWN when we went LONG (pullback).
    df_feat['leg_alignment'] = df_feat['leg_dir'] * np.where(df_feat['is_long_trade'], 1, -1)
    
    for col in ['leg_alignment', 'leg_ext', 'leg_vel']:
        labels = df_feat[df_feat['is_label'] == 1][col].values
        nulls = df_feat[df_feat['is_label'] == 0][col].values
        
        mean_diff = np.mean(labels) - np.mean(nulls)
        
        # bootstrap p-value
        pool = np.concatenate([labels, nulls])
        n_l = len(labels)
        diffs = []
        for _ in range(1000):
            np.random.shuffle(pool)
            diffs.append(np.mean(pool[:n_l]) - np.mean(pool[n_l:]))
        p_val = np.mean(np.abs(diffs) >= np.abs(mean_diff))
        
        # AUC (needs scaling/flipping so >0.5)
        auc = roc_auc_score(df_feat['is_label'], df_feat[col])
        if auc < 0.5: auc = 1.0 - auc
        gap = auc - 0.5
        
        signal = "NOISE"
        if gap >= 0.10: signal = "REAL"
        elif gap >= 0.05: signal = "CONDITIONAL"
            
        print(f"\nFeature: {col}")
        print(f"Label Mean: {np.mean(labels):.2f} | Null Mean: {np.mean(nulls):.2f}")
        print(f"Mean Diff: {mean_diff:.2f} (p={p_val:.4f})")
        print(f"AUC: {auc:.3f} | Gap: {gap:.3f} -> {signal}")

if __name__ == '__main__':
    main()
