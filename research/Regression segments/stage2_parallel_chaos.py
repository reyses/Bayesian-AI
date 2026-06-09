"""Stage 2 chaos-gap re-segmentation (finer tiers 1-9).

⚠️ NON-CAUSAL / DIAGNOSTIC ONLY — same caveat as stage1_speed_pass.py. Segment
betas/tiers/boundaries are in-sample fits; never use them as a live feature or
training target.

NOTE: this stage deliberately uses CPU sklearn ElasticNetCV (not the GPU FISTA
path used in stage1). Stage 2 fans chaos blocks across a multiprocessing Pool;
sharing one CUDA context across forked workers causes PCIe/context thrashing
(the same reason run_year hardcodes parallel days = 1). CPU solving per worker is
the intentional trade-off. Consequence: tiers from stage1 and stage2 are not
strictly solver-identical and should not be compared at the margin.
"""
import os
import sys
import json
import time
import argparse
import numpy as np
import pandas as pd
import warnings
import torch
from multiprocessing import Pool
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNetCV
from group_lasso import GroupLasso

warnings.filterwarnings('ignore')

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from core_v2.features import load_features

SEED_BARS = 30

def max_consecutive(arr):
    if not np.any(arr): return 0
    padded = np.pad(arr, (1, 1), mode='constant')
    diff = np.diff(padded.astype(int))
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]
    return np.max(ends - starts)

def categorize_chaos_segment(residuals, E):
    max_res = np.max(residuals)
    
    # Check all tiers up to Tier 8 (45-50%)
    if max_res <= 1.5 * E and max_consecutive(residuals > 1.0 * E) < 3: return 1
    if max_res <= 2.0 * E and max_consecutive(residuals > 1.5 * E) < 3: return 2
    if max_res <= 2.5 * E and max_consecutive(residuals > 2.0 * E) < 3: return 3
    if max_res <= 3.0 * E and max_consecutive(residuals > 2.5 * E) < 3: return 4
    if max_res <= 3.5 * E and max_consecutive(residuals > 3.0 * E) < 3: return 5
    if max_res <= 4.0 * E and max_consecutive(residuals > 3.5 * E) < 3: return 6
    if max_res <= 4.5 * E and max_consecutive(residuals > 4.0 * E) < 3: return 7
    if max_res <= 5.0 * E and max_consecutive(residuals > 4.5 * E) < 3: return 8
    
    return 9 # Pure Chaos

def build_groups_from_columns(columns):
    groups = []
    tf_map = {'5s':0, '15s':1, '1m':2, '5m':3, '15m':4, '1h':5, '4h':6, '1D':7}
    for c in columns:
        if c == 'L0_time_of_day':
            groups.append(0)
            continue
        try:
            parts = c.split('_')
            layer = int(parts[0][1])
            tf = parts[1]
            tf_idx = tf_map[tf]
            groups.append(tf_idx * 3 + layer)
        except:
            groups.append(0)
    return np.array(groups)

def poly_expand_gpu(X_t):
    device = X_t.device
    N, p = X_t.shape
    idx_i, idx_j = torch.triu_indices(p, p, offset=0, device=device)
    quad = X_t[:, idx_i] * X_t[:, idx_j]
    return torch.cat([X_t, quad], dim=1)

def screen_pipeline_cpu(X_raw, Y, groups):
    try:
        grid = GroupLasso(groups=groups, group_reg=0.01, l1_reg=0.0, n_iter=500, tol=1e-3, fit_intercept=False, supress_warning=True)
        grid.fit(X_raw, Y)
        best_gl = grid
        best_group_reg = 0.01
    except:
        best_gl = GroupLasso(groups=groups, group_reg=0.01, l1_reg=0.0, fit_intercept=False, supress_warning=True)
        best_gl.fit(X_raw, Y)
        best_group_reg = 0.01
        
    w = best_gl.coef_.flatten()
    active_idx = np.where(np.abs(w) > 1e-5)[0]
    if len(active_idx) == 0: return [], best_group_reg, 0.01
        
    X_surv = X_raw[:, active_idx]
    enet = ElasticNetCV(l1_ratio=0.5, cv=3, n_jobs=1, fit_intercept=False, max_iter=1000)
    try:
        enet.fit(X_surv, Y)
        w_enet = enet.coef_
        final_survivors = np.where(np.abs(w_enet) > 1e-6)[0]
        final_idx = active_idx[final_survivors]
        return final_idx, best_group_reg, enet.alpha_
    except:
        return [], best_group_reg, 0.01

def ols_fit_pytorch(X_t, Y_t):
    p = X_t.shape[1]
    XtX = X_t.T @ X_t
    XtY = X_t.T @ Y_t
    ridge = torch.eye(p, dtype=torch.float32, device=X_t.device) * 1e-6
    try:
        beta = torch.linalg.solve(XtX + ridge, XtY)
        return beta
    except:
        return None

def batched_ols_scan_pytorch(X_t, Y_t, min_bars, max_bars, E):
    device = X_t.device
    p = X_t.shape[1]
    batch_size = max_bars - min_bars + 1
    
    XtX_base = torch.zeros((p, p), dtype=torch.float32, device=device)
    XtY_base = torch.zeros((p, 1), dtype=torch.float32, device=device)
    
    if min_bars > 1:
        X_sub = X_t[:min_bars-1]
        Y_sub = Y_t[:min_bars-1]
        XtX_base = X_sub.T @ X_sub
        XtY_base = X_sub.T @ Y_sub
        
    XtX_batch = torch.zeros((batch_size, p, p), dtype=torch.float32, device=device)
    XtY_batch = torch.zeros((batch_size, p, 1), dtype=torch.float32, device=device)
    
    for i, L in enumerate(range(min_bars, max_bars + 1)):
        x_row = X_t[L-1:L]
        y_val = Y_t[L-1:L]
        XtX_base = XtX_base + x_row.T @ x_row
        XtY_base = XtY_base + x_row.T @ y_val
        XtX_batch[i] = XtX_base
        XtY_batch[i] = XtY_base
        
    ridge = torch.eye(p, dtype=torch.float32, device=device).unsqueeze(0) * 1e-6
    XtX_batch = XtX_batch + ridge
    
    tiers_dict = {}
    max_residuals = {}
    betas_dict = {}
    
    try:
        betas = torch.linalg.solve(XtX_batch, XtY_batch) 
    except Exception as e:
        return {}, {}, {}
        
    for i, L in enumerate(range(min_bars, max_bars + 1)):
        beta = betas[i] 
        X_sub = X_t[:L]
        Y_sub = Y_t[:L]
        
        preds = X_sub @ beta
        residuals = torch.abs(Y_sub - preds)
        res_cpu = residuals.squeeze().cpu().numpy()
        
        tier = categorize_chaos_segment(res_cpu, E)
        
        tiers_dict[L] = tier
        max_residuals[L] = float(torch.max(residuals).item())
        betas_dict[L] = beta.squeeze().cpu().numpy().tolist()
        
    return tiers_dict, max_residuals, betas_dict

def evaluate_block(start_idx, length, E, X_global_t, close_prices_t, groups):
    X_t = X_global_t[start_idx : start_idx + length]
    raw_t = close_prices_t[start_idx : start_idx + length]
    Y_t = (raw_t - raw_t[0]).unsqueeze(1)
    
    X_cpu = X_t.cpu().numpy()
    Y_cpu = Y_t.cpu().numpy().flatten()
    
    active_idx, _, _ = screen_pipeline_cpu(X_cpu, Y_cpu, groups)
    if len(active_idx) == 0:
        return 9, [], [], 9999.0, []
        
    X_sub_t = X_t[:, active_idx]
    X_poly_t = poly_expand_gpu(X_sub_t)
    X_poly_cpu = X_poly_t.cpu().numpy()
    
    enet = ElasticNetCV(l1_ratio=0.5, cv=3, n_jobs=1, fit_intercept=False, max_iter=1000)
    try:
        enet.fit(X_poly_cpu, Y_cpu)
        w_enet = enet.coef_
        fixed_terms = np.where(np.abs(w_enet) > 1e-6)[0]
    except:
        return 9, [], [], 9999.0, active_idx.tolist() if hasattr(active_idx, "tolist") else active_idx
        
    max_features = max(1, length - 2)
    max_features = min(15, max_features)
    
    if len(fixed_terms) > max_features:
        fixed_terms = np.argsort(np.abs(w_enet))[::-1][:max_features].copy()
        
    if len(fixed_terms) == 0:
        return 9, [], [], 9999.0, active_idx.tolist() if hasattr(active_idx, "tolist") else active_idx
        
    X_poly_fixed_t = X_poly_t[:, fixed_terms]
    beta_t = ols_fit_pytorch(X_poly_fixed_t, Y_t)
    
    if beta_t is None:
        return 9, [], [], 9999.0, active_idx.tolist() if hasattr(active_idx, "tolist") else active_idx
        
    beta_cpu = beta_t.squeeze().cpu().numpy().tolist()
    if not isinstance(beta_cpu, list):
        beta_cpu = [beta_cpu]
        
    preds_t = X_poly_fixed_t @ beta_t
    preds_cpu = preds_t.cpu().numpy().flatten()
    
    residuals = np.abs(Y_cpu - preds_cpu)
    tier = categorize_chaos_segment(residuals, E)
    max_residual = float(np.max(residuals))
    
    return tier, fixed_terms.tolist(), beta_cpu, max_residual, active_idx.tolist()

# --- MULTIPROCESSING WORKER ---
def process_chaos_block(args):
    block_id, chaos_data, day, E, groups, X_chunk, Y_chunk = args
    
    start_global = chaos_data['start_idx']
    end_global = chaos_data['end_idx']
    N_GAP = end_global - start_global
    
    # Reconstruct tensors locally
    X_global_t = torch.tensor(X_chunk, dtype=torch.float32)
    close_prices_t = torch.tensor(Y_chunk, dtype=torch.float32)
    
    print(f"[WORKER {block_id}] Processing gap length {N_GAP} (E={E:.4f})")
    
    seed_start = 0
    segments = []
    
    while seed_start + SEED_BARS < N_GAP:
        t_seg0 = time.time()
        
        tier, fixed_terms, beta_cpu, max_residual, active_idx = evaluate_block(
            seed_start, SEED_BARS, E, X_global_t, close_prices_t, groups
        )
        
        if tier <= 8:
            L_star = SEED_BARS
            max_forward = min(500, N_GAP - seed_start) 
            if max_forward > SEED_BARS:
                X_exp_raw_t = X_global_t[seed_start:seed_start+max_forward, active_idx]
                X_exp_poly_t = poly_expand_gpu(X_exp_raw_t)[:, fixed_terms]
                Y_exp_t = (close_prices_t[seed_start:seed_start+max_forward] - close_prices_t[seed_start]).unsqueeze(1)
                
                tiers_exp, max_res_exp, betas_exp = batched_ols_scan_pytorch(X_exp_poly_t, Y_exp_t, SEED_BARS + 1, max_forward, E)
                
                for L_next in range(SEED_BARS + 1, max_forward + 1):
                    if L_next in tiers_exp and tiers_exp[L_next] <= tier:
                        L_star = L_next
                        beta_cpu = betas_exp[L_next]
                        max_residual = max_res_exp[L_next]
                    else:
                        break
                        
            print(f"  [WORKER {block_id}] Expanded Tier {tier} segment to {L_star} bars")
            
            segments.append({
                'day': day,
                'start_idx': start_global + seed_start,
                'end_idx': start_global + seed_start + L_star,
                'length': L_star,
                'active_grid_cells': active_idx,
                'surviving_terms_count': len(fixed_terms),
                'surviving_polynomial_indices': fixed_terms,
                'beta_coefficients': beta_cpu,
                'max_residual': float(max_residual),
                'error_band_used': E,
                'volatility_tier': tier,
                'status': 'RECOVERED'
            })
            seed_start += L_star
            
        else:
            # It's Pure Chaos (Tier 9). Chop it explicitly as pure chaos and move forward.
            segments.append({
                'day': day,
                'start_idx': start_global + seed_start,
                'end_idx': start_global + seed_start + SEED_BARS,
                'length': SEED_BARS,
                'active_grid_cells': [],
                'surviving_terms_count': 0,
                'surviving_polynomial_indices': [],
                'beta_coefficients': [],
                'max_residual': 9999.0,
                'error_band_used': E,
                'volatility_tier': 9,
                'status': 'PURE_CHAOS'
            })
            seed_start += SEED_BARS

    return segments

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--day', type=str, required=True)
    parser.add_argument('--atlas_root', type=str, default="C:/Users/reyse/OneDrive/Desktop/Bayesian-AI/DATA/ATLAS")
    args = parser.parse_args()
    
    print(f"[MAIN] Stage 2 Chaos Processor starting for {args.day}")
    
    with open(f"artifacts/stage1_segments_{args.day}.json", 'r') as f:
        all_segments = json.load(f)
        
    chaos_blocks = [s for s in all_segments if s['status'] == 'UNPROCESSED_CHAOS']
    print(f"[MAIN] Found {len(chaos_blocks)} Chaos blocks to process in parallel.")
    
    if len(chaos_blocks) == 0:
        return
        
    # Main process loads the dataset ONCE
    atlas_root = args.atlas_root
    features_root = os.path.join(atlas_root, 'FEATURES_5s_v2')
    
    try:
        df = load_features([args.day], root=features_root)
    except Exception as e:
        print(f"[FATAL] Feature load failed: {e}")
        return
        
    ohlcv = pd.read_parquet(os.path.join(atlas_root, '5s', f'{args.day}.parquet'))
    min_len = min(len(df), len(ohlcv))
    df = df.iloc[:min_len]
    ohlcv = ohlcv.iloc[:min_len]
    
    features_cols = [c for c in df.columns if c != 'timestamp' and 'price_mean' not in c and 'vwap' not in c]
    scaler = StandardScaler()
    X_global = scaler.fit_transform(df[features_cols].values)
    
    valid_idx = ~np.isnan(X_global).any(axis=1)
    raw_indices = np.where(valid_idx)[0]
    X_global = X_global[valid_idx]
    close_prices = ohlcv['close'].values[valid_idx]
    N_TOTAL = len(X_global)
    
    groups = build_groups_from_columns(features_cols)
    
    worker_args = []
    for i, block in enumerate(chaos_blocks):
        s_idx = block['start_idx']
        e_idx = block['end_idx']
        X_chunk = X_global[s_idx:e_idx]
        Y_chunk = close_prices[s_idx:e_idx]
        worker_args.append((i, block, args.day, block['error_band_used'], groups, X_chunk, Y_chunk))
        
    print(f"[MAIN] Launching Multiprocessing Pool...")
    
    final_segments = [s for s in all_segments if s['status'] == 'PRISTINE']
    
    import psutil
    available_ram_mb = psutil.virtual_memory().available / (1024 * 1024)
    safe_ram_mb = max(0, available_ram_mb - 1024) # Keep 1GB explicitly free
    max_workers_by_ram = max(1, int(safe_ram_mb / 150)) # Assume ~150MB per worker
    max_workers_by_cpu = max(1, os.cpu_count() * 2)
    
    num_workers = min(max_workers_by_ram, max_workers_by_cpu)
    print(f"[MAIN] Dynamic Scaling: {available_ram_mb:.0f}MB Free RAM -> RAM Limit: {max_workers_by_ram} workers. CPU Limit: {max_workers_by_cpu}. Spawning {num_workers} workers.")
    
    with Pool(processes=num_workers) as pool:
        results = pool.map(process_chaos_block, worker_args)
        
    for res in results:
        final_segments.extend(res)
        
    for seg in final_segments:
        s = seg['start_idx']
        e = seg['end_idx']
        seg['raw_start_idx'] = int(raw_indices[s]) if s < N_TOTAL else int(raw_indices[-1] + 1)
        seg['raw_end_idx'] = int(raw_indices[e]) if e < N_TOTAL else int(raw_indices[-1] + 1)
        
    # Sort them back chronologically
    final_segments.sort(key=lambda x: x['start_idx'])
    
    output_json = f"artifacts/stage2_segments_{args.day}.json"
    with open(output_json, 'w') as f:
        json.dump(final_segments, f, indent=2)
        
    print(f"[MAIN] Processing complete. Final segments saved to {output_json}")

if __name__ == "__main__":
    main()
