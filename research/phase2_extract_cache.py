import os
import sys
import json
import time
import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from collections import defaultdict

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from core_v2.features import load_features

def main():
    print("[Extract] Starting Flat Tensor Cache Extraction...")
    
    with open('artifacts/stage2_year_segments.json', 'r') as f:
        segments = json.load(f)
        
    valid = [s for s in segments if s['status'] in ['PRISTINE', 'RECOVERED']]
    print(f"Total valid segments to extract: {len(valid)}")
    
    segments_by_day = defaultdict(list)
    for i, seg in enumerate(valid):
        segments_by_day[seg['day']].append((i, seg))
        
    atlas_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "DATA", "ATLAS"))
    features_root = os.path.join(atlas_root, 'FEATURES_5s_v2')
    P_total = 177
    
    total_L = 0
    for seg in valid:
        total_L += (seg['end_idx'] - seg['start_idx'])
        
    print(f"[Extract] Allocating Flat C-Arrays for {total_L} total segment ticks...")
    X_flat = torch.zeros((total_L, P_total), dtype=torch.float32)
    Y_flat = torch.zeros((total_L, 1), dtype=torch.float32)
    boundaries = torch.zeros((len(valid) + 1,), dtype=torch.int32)
    error_bands = torch.zeros((len(valid),), dtype=torch.float32)
    
    t0 = time.time()
    current_row = 0
    
    days_processed = 0
    for day, seg_list in segments_by_day.items():
        try:
            df = load_features([day], root=features_root)
            ohlcv = pd.read_parquet(os.path.join(atlas_root, '5s', f"{day}.parquet"))
        except Exception as e:
            print(f"Failed to load {day}: {e}")
            continue
            
        min_len = min(len(df), len(ohlcv))
        df = df.iloc[:min_len]
        ohlcv = ohlcv.iloc[:min_len]
        
        features_cols = [c for c in df.columns if c != 'timestamp' and 'price_mean' not in c and 'vwap' not in c]
        scaler = StandardScaler()
        X_global = scaler.fit_transform(df[features_cols].values)
        
        valid_idx = ~np.isnan(X_global).any(axis=1)
        X_global = X_global[valid_idx]
        close_prices = ohlcv['close'].values[valid_idx]
        
        for idx, seg in seg_list:
            s_idx = seg['start_idx']
            e_idx = seg['end_idx']
            length = e_idx - s_idx
            
            X_slice = X_global[s_idx:e_idx, :P_total]
            Y_slice = close_prices[s_idx:e_idx]
            Y_slice = Y_slice - Y_slice[0]
            
            X_flat[current_row:current_row+length] = torch.tensor(X_slice, dtype=torch.float32)
            Y_flat[current_row:current_row+length] = torch.tensor(Y_slice, dtype=torch.float32).unsqueeze(1)
            error_bands[idx] = seg['error_band_used']
            boundaries[idx] = current_row
            
            current_row += length
            
        days_processed += 1
        if days_processed % 10 == 0:
            print(f"  Processed {days_processed}/{len(segments_by_day)} days... ({time.time()-t0:.1f}s)")
            
    boundaries[-1] = current_row
    
    print(f"[Extract] Saving Flat Tensor Cache payload (Zero Object Overhead)...")
    payload = {
        'X_flat': X_flat,
        'Y_flat': Y_flat,
        'boundaries': boundaries,
        'error_bands': error_bands
    }
    torch.save(payload, 'artifacts/sweep_cache_flat.pt')
    print(f"Extraction finished in {time.time()-t0:.2f}s!")

if __name__ == '__main__':
    main()
