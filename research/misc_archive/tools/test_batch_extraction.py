import os
import sys
import json
import time
import numpy as np
import pandas as pd
import torch
from collections import defaultdict
from sklearn.preprocessing import StandardScaler

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from core_v2.features import load_features

# Import classify_tier from archive
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'archive', 'research', 'Regression segments')))
try:
    from tiering import classify_tier
except ImportError:
    def classify_tier(residuals, error_band, max_tier=8):
        max_res = np.max(residuals)
        ratio = max_res / error_band
        if ratio <= 1.0: return 1
        if ratio <= 1.5: return 2
        if ratio <= 2.0: return 3
        if ratio <= 2.5: return 4
        return 8

def map_local_poly_to_global(active_idx, local_poly_idx, P_total):
    p_local = len(active_idx)
    if local_poly_idx < p_local: return active_idx[local_poly_idx]
    quad_offset = local_poly_idx - p_local
    count = 0
    local_i, local_j = -1, -1
    for i in range(p_local):
        for j in range(i, p_local):
            if count == quad_offset:
                local_i, local_j = i, j
                break
            count += 1
        if local_i != -1: break
            
    f1, f2 = active_idx[local_i], active_idx[local_j]
    if f1 > f2: f1, f2 = f2, f1
    idx = f1 * P_total - (f1 * (f1 - 1)) // 2 + (f2 - f1)
    return P_total + idx

def poly_expand_global_gpu(X_t):
    device = X_t.device
    N, p = X_t.shape
    idx_i, idx_j = torch.triu_indices(p, p, offset=0, device=device)
    quad = X_t[:, idx_i] * X_t[:, idx_j]
    return torch.cat([X_t, quad], dim=1)

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[Batch Test] Using device: {device}")
    
    with open('artifacts/stage2_year_segments.json', 'r') as f:
        segments = json.load(f)
        
    valid = [s for s in segments if s['status'] in ['PRISTINE', 'RECOVERED']]
    P_total = 177
    total_expanded_features = P_total + (P_total * (P_total + 1)) // 2
    
    # 1. Build Dense Beta Matrix
    print(f"[Batch Test] Building Dense Beta Matrix...")
    t0 = time.time()
    B_dense = torch.zeros((total_expanded_features, len(valid)), dtype=torch.float32, device=device)
    
    for seg_idx, s in enumerate(valid):
        active_idx = s['active_grid_cells']
        fixed_terms = s['surviving_polynomial_indices']
        betas = s['beta_coefficients']
        
        if isinstance(active_idx, (int, float)): active_idx = [int(active_idx)]
        if isinstance(fixed_terms, (int, float)): fixed_terms = [int(fixed_terms)]
        if isinstance(betas, (int, float)): betas = [float(betas)]
        
        for local_poly_idx, beta_val in zip(fixed_terms, betas):
            try:
                global_idx = map_local_poly_to_global(active_idx, local_poly_idx, P_total)
                B_dense[global_idx, seg_idx] = beta_val
            except: pass
    print(f"  Matrix built in {time.time()-t0:.2f}s")
    
    # 2. Extract Data for first 10 segments
    target_segments = valid[:10]
    segments_by_day = defaultdict(list)
    for i, seg in enumerate(target_segments):
        segments_by_day[seg['day']].append((i, seg))
        
    print(f"\n[Extraction] Loading raw data for {len(target_segments)} segments across {len(segments_by_day)} days...")
    t1 = time.time()
    
    atlas_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "DATA", "ATLAS"))
    features_root = os.path.join(atlas_root, 'FEATURES_5s_v2')
    
    extracted_X = {}
    extracted_Y = {}
    
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
        
        for i, seg in seg_list:
            s_idx = seg['start_idx']
            e_idx = seg['end_idx']
            X_slice = X_global[s_idx:e_idx, :P_total]
            Y_slice = close_prices[s_idx:e_idx]
            
            extracted_X[i] = torch.tensor(X_slice, dtype=torch.float32, device=device)
            raw_Y = torch.tensor(Y_slice, dtype=torch.float32, device=device)
            extracted_Y[i] = (raw_Y - raw_Y[0]).unsqueeze(1)
            
    print(f"  Extraction completed in {time.time()-t1:.2f}s")
    
    # 3. Concatenate and Compute Batch
    print(f"\n[Compute] Running batched Cross-Fit for 10 segments against 80,000 curves...")
    t2 = time.time()
    
    X_list = [extracted_X[i] for i in range(10)]
    Y_list = [extracted_Y[i] for i in range(10)]
    
    # Track boundaries
    lengths = [x.shape[0] for x in X_list]
    boundaries = [0] + np.cumsum(lengths).tolist()
    
    X_batch = torch.cat(X_list, dim=0) # Total_L x P_total
    Y_batch = torch.cat(Y_list, dim=0) # Total_L x 1
    
    print(f"  Total Batch Shape: X={X_batch.shape}, Y={Y_batch.shape}")
    
    X_batch_poly = poly_expand_global_gpu(X_batch)
    Y_preds_batch = X_batch_poly @ B_dense
    residuals_batch = torch.abs(Y_preds_batch - Y_batch)
    
    # 4. Resolve back to individual segments
    for i in range(10):
        start = boundaries[i]
        end = boundaries[i+1]
        seg = target_segments[i]
        
        seg_res = residuals_batch[start:end, :] # (L_i x 80000)
        max_res, _ = torch.max(seg_res, dim=0) # 80000
        
        max_res_cpu = max_res.cpu().numpy()
        matches = 0
        for m_res in max_res_cpu:
            tier = classify_tier(np.array([m_res]), seg['error_band_used'])
            if tier <= 2:
                matches += 1
                
        print(f"  Segment {i} ({seg['day']}, len {lengths[i]}): Found {matches} acceptable matches out of 80,000.")
        
    print(f"  Batched matrix math completed in {time.time()-t2:.4f}s")

if __name__ == '__main__':
    main()
