import os
import glob
import json
import numpy as np
import pandas as pd
from itertools import combinations
import matplotlib.pyplot as plt
import seaborn as sns

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
REPORTS = os.path.join(ROOT, "research", "nt8_catalog", "reports")
HORIZONS = os.path.join(REPORTS, "fps_horizons.parquet")
LABELS_DIR = os.path.join(ROOT, "DATA", "ai_cusp_picks")

def load_horizons():
    df = pd.read_parquet(HORIZONS)
    print(f"Loaded {len(df)} horizons")
    # Fix ORB-02 timestamps
    orb_mask = df['doss'] == 'ORB-02'
    df.loc[orb_mask, 'entry_ts'] += 1800
    
    # Exclude absent index spaces
    df = df[~df['doss'].isin(['SEASON-12', 'RENKO-24'])]
    print(f"Post-filter horizons: {len(df)}")
    return df

def load_labels():
    files = glob.glob(os.path.join(LABELS_DIR, "ai_picks_*_multi.json"))
    trades = []
    for f in files:
        with open(f, 'r') as fp:
            data = json.load(fp)
            if 'trades' in data:
                trades.extend(data['trades'])
    df_labels = pd.DataFrame(trades)
    print(f"Loaded {len(df_labels)} labeled trades")
    return df_labels

def main():
    df_horizons = load_horizons()
    df_labels = load_labels()
    
    # 1. Bucket entries into 5-minute bins
    # entry_ts is unix seconds. 5 mins = 300 seconds
    df_horizons['bin_5m'] = (df_horizons['entry_ts'] // 300) * 300
    
    # Create co-fire matrix
    # pivot to see which dossiers fired in which bin
    firing = df_horizons.groupby(['bin_5m', 'doss']).size().unstack(fill_value=0)
    firing = (firing > 0).astype(int)
    
    dossiers = firing.columns.tolist()
    n = len(dossiers)
    jaccard = np.zeros((n, n))
    co_count = np.zeros((n, n))
    
    for i in range(n):
        for j in range(n):
            if i == j:
                jaccard[i, j] = 1.0
                co_count[i, j] = firing.iloc[:, i].sum()
            else:
                inter = (firing.iloc[:, i] & firing.iloc[:, j]).sum()
                union = (firing.iloc[:, i] | firing.iloc[:, j]).sum()
                co_count[i, j] = inter
                jaccard[i, j] = inter / union if union > 0 else 0
                
    df_jaccard = pd.DataFrame(jaccard, index=dossiers, columns=dossiers)
    df_count = pd.DataFrame(co_count, index=dossiers, columns=dossiers)
    
    print("\n--- CO-FIRE JACCARD MATRIX ---")
    print(df_jaccard.round(3))
    
    print("\n--- HIGHLY CORRELATED PAIRS (Jaccard > 0.1) ---")
    for d1, d2 in combinations(dossiers, 2):
        if df_jaccard.loc[d1, d2] > 0.1:
            print(f"{d1} - {d2}: J={df_jaccard.loc[d1, d2]:.3f} (co-fires: {df_count.loc[d1, d2]})")
            
    # Confluence zones
    firing['doss_count'] = firing.sum(axis=1)
    for k in [2, 3, 4, 5]:
        zones = (firing['doss_count'] >= k).sum()
        print(f"Confluence bins (>= {k} dossiers): {zones} bins")
        
    # 2. Label overlay
    # For each labeled trade, distance to nearest catalog event
    # We will build a sorted array of catalog event timestamps per dossier
    catalog_ts = {d: np.sort(df_horizons[df_horizons['doss'] == d]['entry_ts'].values) for d in dossiers}
    # Also for all dossiers combined
    catalog_ts['ALL'] = np.sort(df_horizons['entry_ts'].values)
    
    label_ts = df_labels['entry_ts'].values
    
    def nearest_dist(target_ts, array_ts):
        if len(array_ts) == 0:
            return np.nan
        idx = np.searchsorted(array_ts, target_ts)
        # Check idx and idx-1
        dists = []
        if idx < len(array_ts):
            dists.append(np.abs(array_ts[idx] - target_ts))
        if idx > 0:
            dists.append(np.abs(array_ts[idx-1] - target_ts))
        return np.min(dists) / 60.0 # in minutes
        
    results = {}
    for group, ts_arr in catalog_ts.items():
        if len(ts_arr) == 0: continue
        dists = np.array([nearest_dist(ts, ts_arr) for ts in label_ts])
        # Expected distance (arithmetic baseline)
        # lambda = events / total time
        # active time roughly 6.5 hours per day (23400 sec) * 576 days
        total_time_minutes = 576 * 390
        rate = len(ts_arr) / total_time_minutes
        exp_dist = 1.0 / (2 * rate) if rate > 0 else np.nan
        results[group] = {
            'N_events': len(ts_arr),
            'median_dist_min': np.nanmedian(dists),
            'mean_dist_min': np.nanmean(dists),
            'baseline_dist_min': exp_dist,
            'match_within_5m': np.mean(dists <= 5.0),
            'match_within_15m': np.mean(dists <= 15.0)
        }
        
    df_res = pd.DataFrame(results).T
    print("\n--- LABEL OVERLAY DISTANCES (MINUTES) ---")
    print(df_res.round(2))

if __name__ == "__main__":
    main()
