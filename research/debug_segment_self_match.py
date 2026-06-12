import os
import sys
import json
import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from core_v2.features import load_features

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
    with open('artifacts/stage2_year_segments.json', 'r') as f:
        segments = json.load(f)
        
    valid = [s for s in segments if s['status'] in ['PRISTINE', 'RECOVERED']]
    P_total = 177
    total_expanded_features = P_total + (P_total * (P_total + 1)) // 2
    
    seg_idx = 6 # The one that failed
    seg = valid[seg_idx]
    print(f"Debugging Segment {seg_idx}")
    print(f"Original max residual: {seg['max_residual']}, Error Band: {seg['error_band_used']}")
    
    # 1. Build B_dense for just this segment
    B_dense = torch.zeros((total_expanded_features, 1), dtype=torch.float32, device=device)
    active_idx = seg['active_grid_cells']
    fixed_terms = seg['surviving_polynomial_indices']
    betas = seg['beta_coefficients']
    
    if isinstance(active_idx, (int, float)): active_idx = [int(active_idx)]
    if isinstance(fixed_terms, (int, float)): fixed_terms = [int(fixed_terms)]
    if isinstance(betas, (int, float)): betas = [float(betas)]
    
    for local_poly_idx, beta_val in zip(fixed_terms, betas):
        global_idx = map_local_poly_to_global(active_idx, local_poly_idx, P_total)
        B_dense[global_idx, 0] = beta_val
        
    # 2. Extract Data
    day = seg['day']
    atlas_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "DATA", "ATLAS"))
    features_root = os.path.join(atlas_root, 'FEATURES_5s_v2')
    
    df = load_features([day], root=features_root)
    ohlcv = pd.read_parquet(os.path.join(atlas_root, '5s', f"{day}.parquet"))
    
    min_len = min(len(df), len(ohlcv))
    df = df.iloc[:min_len]
    ohlcv = ohlcv.iloc[:min_len]
    
    features_cols = [c for c in df.columns if c != 'timestamp' and 'price_mean' not in c and 'vwap' not in c]
    scaler = StandardScaler()
    X_global = scaler.fit_transform(df[features_cols].values)
    
    valid_idx = ~np.isnan(X_global).any(axis=1)
    X_global = X_global[valid_idx]
    close_prices = ohlcv['close'].values[valid_idx]
    
    s_idx = seg['start_idx']
    e_idx = seg['end_idx']
    X_slice = X_global[s_idx:e_idx, :P_total]
    Y_slice = close_prices[s_idx:e_idx]
    
    X_A = torch.tensor(X_slice, dtype=torch.float32, device=device)
    raw_Y = torch.tensor(Y_slice, dtype=torch.float32, device=device)
    Y_A = (raw_Y - raw_Y[0]).unsqueeze(1)
    
    # 3. Predict
    X_A_poly = poly_expand_global_gpu(X_A)
    Y_preds = X_A_poly @ B_dense
    residuals = torch.abs(Y_preds - Y_A)
    max_res = torch.max(residuals).item()
    
    print(f"Re-computed max residual: {max_res}")
    print(f"Difference: {abs(max_res - seg['max_residual'])}")

if __name__ == '__main__':
    main()
