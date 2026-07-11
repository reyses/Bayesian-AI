"""Phase 3: SFE Feature Exhaustion
Generates the entire Statistical Field Engine (L1, L2, L3) feature matrix for 
every label and matched null, then tests every single SFE column for 
signal-magnitude distinctness (AUC Gap).
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
pd.options.mode.chained_assignment = None

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, '..', '..', '..'))

sys.path.append(os.path.join(_REPO))
sys.path.append(os.path.join(_REPO, 'core_v2'))
from statistical_field_engine import StatisticalFieldEngine

def load_atlas(day, tf='1m'):
    path = os.path.join(_REPO, 'DATA', 'ATLAS', tf, f"{day}.parquet")
    if not os.path.exists(path):
        return None
    return pd.read_parquet(path)

def main():
    files = sorted(glob.glob(os.path.join(_REPO, 'DATA', 'ai_cusp_picks', '*_multi.json')))
    features = []
    
    sfe = StatisticalFieldEngine()
    
    # Track missing files or parsing issues
    valid_files = 0
    
    for f in files:
        day_str = os.path.basename(f).replace('ai_picks_', '').replace('_multi.json', '').replace('-', '_')
        df = load_atlas(day_str, '1m')
        if df is None: continue
        
        try:
            d = json.load(open(f))
        except Exception:
            continue
            
        trades = d.get('trades', [])
        if not trades: continue
        
        # SFE Computation
        df_L0 = sfe.compute_L0(df)
        df_L1 = sfe.compute_L1(df, '1m')
        df_L2 = sfe.compute_L2(df, '1m', N=30)
        df_L3 = sfe.compute_L3(df, '1m', N=30)
        
        # Concat all features
        df_sfe = pd.concat([df_L0, df_L1, df_L2, df_L3], axis=1)
        
        ts = df['timestamp'].values
        hours = np.array([datetime.fromtimestamp(t['entry_ts'], tz=timezone.utc).hour for t in trades])
        df_hours = np.array([datetime.fromtimestamp(t, tz=timezone.utc).hour for t in ts])
        
        for t, h in zip(trades, hours):
            entry_ts = t['entry_ts']
            is_long = str(t.get('direction', '')).upper().startswith('L')
            
            idx = np.searchsorted(ts, entry_ts, side='right') - 1
            if idx < 60: continue
            
            # Label Row
            row_data = df_sfe.iloc[idx].to_dict()
            row_data['is_label'] = 1
            row_data['is_long'] = int(is_long)
            features.append(row_data)
            
            # Matched Null
            null_candidates = np.where((df_hours == h) & (ts <= entry_ts) & (np.arange(len(ts)) >= 60))[0]
            null_candidates = [c for c in null_candidates if abs(c - idx) > 5]
            
            if len(null_candidates) > 0:
                null_idx = np.random.choice(null_candidates)
                null_row = df_sfe.iloc[null_idx].to_dict()
                null_row['is_label'] = 0
                null_row['is_long'] = int(is_long)
                features.append(null_row)
        
        valid_files += 1

    print(f"Processed {valid_files} daily datasets.")
    
    if not features:
        print("No features extracted.")
        return
        
    df_feat = pd.DataFrame(features)
    
    print("=== Feature 006: SFE Exhaustion Run ===")
    print(f"Total samples: {len(df_feat)} (50% labels, 50% nulls)")
    
    # Exclude non-feature columns
    skip_cols = ['is_label', 'is_long', 'timestamp']
    feat_cols = [c for c in df_feat.columns if c not in skip_cols and pd.api.types.is_numeric_dtype(df_feat[c])]
    
    results = []
    
    for col in feat_cols:
        # We must align direction-sensitive features (like velocity) to trade intent.
        # But for an exhaustive run without knowing which features are symmetric,
        # we compute AUC on the raw feature, and ALSO compute AUC on the "aligned" feature.
        # Aligned: feat * (1 if long else -1).
        
        raw_vals = df_feat[col].values
        aligned_vals = raw_vals * np.where(df_feat['is_long'] == 1, 1.0, -1.0)
        
        # Test 1: Raw AUC (good for absolute metrics like volatility, duration, range)
        valid = ~np.isnan(raw_vals) & ~np.isinf(raw_vals)
        if valid.sum() < 100: continue
        
        try:
            auc_raw = roc_auc_score(df_feat['is_label'][valid], raw_vals[valid])
            if auc_raw < 0.5: auc_raw = 1.0 - auc_raw
            
            auc_align = roc_auc_score(df_feat['is_label'][valid], aligned_vals[valid])
            if auc_align < 0.5: auc_align = 1.0 - auc_align
        except ValueError:
            continue
            
        best_auc = max(auc_raw, auc_align)
        gap = best_auc - 0.5
        is_aligned = auc_align > auc_raw
        
        # Calculate means on the best representation
        eval_vals = aligned_vals if is_aligned else raw_vals
        labels = eval_vals[(df_feat['is_label'] == 1) & valid]
        nulls = eval_vals[(df_feat['is_label'] == 0) & valid]
        
        mean_diff = np.mean(labels) - np.mean(nulls)
        
        results.append({
            'Feature': col,
            'Aligned': 'Yes' if is_aligned else 'No',
            'Gap': gap,
            'Label_Mean': np.mean(labels),
            'Null_Mean': np.mean(nulls),
            'Mean_Diff': mean_diff
        })

    # Sort and display
    df_res = pd.DataFrame(results).sort_values('Gap', ascending=False)
    
    print(f"\nTested {len(df_res)} features. Top 20 Strongest Causal Features:")
    print("-" * 80)
    for i, row in df_res.head(20).iterrows():
        sig = "REAL" if row['Gap'] >= 0.10 else ("CONDITIONAL" if row['Gap'] >= 0.05 else "NOISE")
        print(f"{row['Feature']:<25} | Gap: {row['Gap']:.3f} ({sig}) | Aligned: {row['Aligned']:<3} | Label: {row['Label_Mean']:>8.3f} | Null: {row['Null_Mean']:>8.3f}")
        
    print("\n" + "-" * 80)
    print("Full Exhaustion Complete.")

if __name__ == '__main__':
    main()
