import os
import sys
import json
import time
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from core_v2.features import load_features

# Import classify_tier from archive
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'archive', 'research', 'Regression segments')))
try:
    from tiering import classify_tier
except ImportError:
    # Minimal fallback if import fails
    def classify_tier(residuals, error_band, max_tier=8):
        max_res = np.max(residuals)
        ratio = max_res / error_band
        if ratio <= 1.0: return 1
        if ratio <= 1.5: return 2
        if ratio <= 2.0: return 3
        if ratio <= 2.5: return 4
        return 8

def map_local_poly_to_global(active_idx, local_poly_idx, P_total):
    """
    Maps an index from a segment's local poly_expand_gpu to the global poly_expand_gpu.
    local_poly_idx: the index in the surviving_polynomial_indices
    P_total: total number of features (e.g. 150)
    """
    p_local = len(active_idx)
    
    # 1. Linear terms
    if local_poly_idx < p_local:
        return active_idx[local_poly_idx]
        
    # 2. Quadratic terms
    # In PyTorch triu_indices, order is:
    # (0,0), (0,1), (0,2)... (1,1), (1,2)...
    quad_offset = local_poly_idx - p_local
    
    # Find which (i, j) in the local active_idx this corresponds to
    # A simple loop is fast enough since p_local is small (< 20)
    count = 0
    local_i, local_j = -1, -1
    for i in range(p_local):
        for j in range(i, p_local):
            if count == quad_offset:
                local_i, local_j = i, j
                break
            count += 1
        if local_i != -1:
            break
            
    f1 = active_idx[local_i]
    f2 = active_idx[local_j]
    
    # Ensure f1 <= f2
    if f1 > f2:
        f1, f2 = f2, f1
        
    # Find the global index for (f1, f2)
    # The global quadratic terms start at P_total.
    # The number of elements before row f1 is:
    # Sum_{k=0}^{f1-1} (P_total - k) = f1 * P_total - f1*(f1-1)/2
    # The offset within row f1 is (f2 - f1)
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
    print(f"[A-vs-All] Using device: {device}")
    
    json_path = 'artifacts/stage2_year_segments.json'
    with open(json_path, 'r') as f:
        segments = json.load(f)
        
    valid = [s for s in segments if s['status'] in ['PRISTINE', 'RECOVERED']]
    print(f"[A-vs-All] Loaded {len(valid)} valid segments.")
    
    # 1. Find max features (P_total)
    max_feat = 0
    for s in valid:
        if len(s['active_grid_cells']) > 0:
            max_feat = max(max_feat, max(s['active_grid_cells']))
            
    P_total = max_feat + 1
    total_expanded_features = P_total + (P_total * (P_total + 1)) // 2
    print(f"[A-vs-All] Global feature space: P={P_total}, Expanded={total_expanded_features}")
    
    # 2. Build the Dense Beta Matrix
    print(f"[A-vs-All] Building Dense Beta Matrix ({total_expanded_features} x {len(valid)})...")
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
            except Exception as e:
                pass # skip invalid bounds
                
    print(f"  Matrix built in {time.time()-t0:.2f}s")
    
    # 3. Select Segment A
    seg_A = valid[0]
    print(f"\n[Segment A] Day: {seg_A['day']}, Span: {seg_A['start_idx']} -> {seg_A['end_idx']}, Error Band: {seg_A['error_band_used']}")
    
    # 4. Load Segment A's dataset
    print(f"[Data] Loading underlying parquets for {seg_A['day']}...")
    atlas_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "DATA", "ATLAS"))
    features_root = os.path.join(atlas_root, 'FEATURES_5s_v2')
    
    try:
        df = load_features([seg_A['day']], root=features_root)
        ohlcv = pd.read_parquet(os.path.join(atlas_root, '5s', f"{seg_A['day']}.parquet"))
    except Exception as e:
        print(f"[FATAL] Could not load data: {e}")
        return
        
    min_len = min(len(df), len(ohlcv))
    df = df.iloc[:min_len]
    ohlcv = ohlcv.iloc[:min_len]
    
    features_cols = [c for c in df.columns if c != 'timestamp' and 'price_mean' not in c and 'vwap' not in c]
    scaler = StandardScaler()
    X_global = scaler.fit_transform(df[features_cols].values)
    
    valid_idx = ~np.isnan(X_global).any(axis=1)
    X_global = X_global[valid_idx]
    close_prices = ohlcv['close'].values[valid_idx]
    
    s_idx = seg_A['start_idx']
    e_idx = seg_A['end_idx']
    
    # Ensure X_A is exactly P_total columns, ignoring any extra columns at the end
    X_A_np = X_global[s_idx:e_idx, :P_total]
    X_A = torch.tensor(X_A_np, dtype=torch.float32, device=device)
    raw_Y = torch.tensor(close_prices[s_idx:e_idx], dtype=torch.float32, device=device)
    Y_A = (raw_Y - raw_Y[0]).unsqueeze(1) # L x 1
    
    print(f"  Loaded X_A shape: {X_A.shape}")
    
    # 5. Global Cross-Fit
    print(f"\n[Compute] Running Global Cross-Fit of 80,000 curves against Segment A...")
    t1 = time.time()
    
    # Expand X_A to global poly
    X_A_poly = poly_expand_global_gpu(X_A) # L x 11475
    
    # Matrix mult
    Y_preds = X_A_poly @ B_dense # (L x 11475) @ (11475 x 80000) -> (L x 80000)
    
    # Residuals
    residuals = torch.abs(Y_preds - Y_A) # Y_A broadcasts across columns
    
    # We want max residual per curve
    max_residuals_per_curve, _ = torch.max(residuals, dim=0) # Shape: 80000
    
    print(f"  Cross-fit completed in {time.time()-t1:.3f}s")
    
    # 6. Apply Tiers
    max_res_cpu = max_residuals_per_curve.cpu().numpy()
    error_band = seg_A['error_band_used']
    
    tier_counts = {1:0, 2:0, 3:0, 4:0, 8:0}
    matching_indices = []
    
    for i, m_res in enumerate(max_res_cpu):
        tier = classify_tier(np.array([m_res]), error_band)
        if tier in tier_counts:
            tier_counts[tier] += 1
        if tier <= 2: # Define match as Tier 1 or 2
            matching_indices.append(i)
            
    print(f"\n[Results] Tiers achieved by the 80,000 curves on Segment A's dataset:")
    print(f"  Tier 1 (Pristine Match): {tier_counts.get(1, 0)}")
    print(f"  Tier 2 (Acceptable Match): {tier_counts.get(2, 0)}")
    print(f"  Tier 3 (Degraded): {tier_counts.get(3, 0)}")
    print(f"  Tier 8 (Total Failure): {tier_counts.get(8, 0)}")
    
    print(f"\nBucket 1 extracted! Found {len(matching_indices)} segments that explain Segment A's reality within Tier 2.")
    
if __name__ == "__main__":
    main()
