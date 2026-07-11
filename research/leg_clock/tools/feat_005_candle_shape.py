"""Feature 005: Candle Shape
Calculate body/wick fractions of the last fully closed 1m candle prior to entry,
testing for aligned rejection wicks and body structure.
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
        op = df['open'].values
        hi = df['high'].values
        lo = df['low'].values
        cl = df['close'].values
        
        c_range = np.maximum(hi - lo, 1e-6)
        body = np.abs(cl - op)
        high_wick = hi - np.maximum(cl, op)
        low_wick = np.minimum(cl, op) - lo
        
        body_ratio = body / c_range
        high_wick_ratio = high_wick / c_range
        low_wick_ratio = low_wick / c_range
        
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
            
            # Get the exact bar starting before entry_ts
            idx = np.searchsorted(ts, entry_ts, side='right') - 1
            
            # To be strictly causal using fully-closed bars, we look at idx - 1
            prev_idx = idx - 1
            if prev_idx < 0: continue
            
            # Aligned rejection: low wick for longs, high wick for shorts
            aligned_rejection = low_wick_ratio[prev_idx] if is_long else high_wick_ratio[prev_idx]
            
            features.append({
                'is_label': 1,
                'aligned_rejection': aligned_rejection,
                'body_ratio': body_ratio[prev_idx]
            })
            
            null_candidates = np.where((df_hours == h) & (ts <= entry_ts))[0]
            # Exclude immediate surroundings
            null_candidates = [c for c in null_candidates if abs(c - prev_idx) > 5 and c > 0]
            
            if len(null_candidates) > 0:
                null_idx = np.random.choice(null_candidates)
                
                null_rejection = low_wick_ratio[null_idx] if is_long else high_wick_ratio[null_idx]
                
                features.append({
                    'is_label': 0,
                    'aligned_rejection': null_rejection,
                    'body_ratio': body_ratio[null_idx]
                })

    df_feat = pd.DataFrame(features)
    
    print("=== Feature 005: Candle Shape (Prior 1m Bar) ===")
    print(f"Total samples: {len(df_feat)} (50% labels, 50% nulls)")
    
    for col in ['aligned_rejection', 'body_ratio']:
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
