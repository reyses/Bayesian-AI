"""Stage 1 regime-segmentation speed pass.

⚠️ NON-CAUSAL / DIAGNOSTIC ONLY. Every segment's betas, volatility_tier, status,
and BOUNDARIES are fit in-sample over the whole segment (including bars in its
own future), and features are StandardScaler-fit over the entire day. These
labels describe which parts of a day turned out to be cleanly regress-able in
hindsight. They are valid as a post-hoc diagnostic only. Do NOT feed
volatility_tier / status / segment membership into any live decision or training
target — doing so reintroduces lookahead (see MEMORY: lookahead artifacts).
"""
import os
import sys
import argparse
import json
import time
import numpy as np
import pandas as pd
import warnings
import torch
import psutil

def get_safe_n_jobs(matrix_mb=0.1):
    """Dynamically calculates safe thread count leaving 1GB free."""
    worker_base_overhead_mb = 100  # Windows Python + Sklearn overhead
    mb_per_worker = worker_base_overhead_mb + (matrix_mb * 20) # 20x for Joblib serialization + CV folds
    
    available_ram_mb = psutil.virtual_memory().available / (1024 * 1024)
    safe_ram_mb = max(0, available_ram_mb - 1024)
    max_workers_by_ram = max(1, int(safe_ram_mb / mb_per_worker))
    max_workers_by_cpu = max(1, os.cpu_count() * 2)
    safe_jobs = min(max_workers_by_ram, max_workers_by_cpu)
    print(f"[MAIN] Dynamic RAM Logic: Matrix {matrix_mb:.2f}MB -> Worker Est {mb_per_worker:.0f}MB. Free RAM {available_ram_mb:.0f}MB. Spawning {safe_jobs} jobs.")
    return safe_jobs


from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNetCV
from group_lasso import GroupLasso

warnings.filterwarnings('ignore')

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from core_v2.features import load_features

# === PARAMETERS ===
SEED_BARS = 30                # bars in the initial probe block (~2.5 min at 5s)
INITIAL_ERROR_BAND = 1.00     # price-point residual tolerance for the FIRST segment
                             # (no prior segment delta exists yet to scale from)
ERROR_BAND_FRACTION = 0.10    # subsequent error band = 10% of prior segment's price range
                             # NOTE: this makes tiers path-dependent on processing
                             # order; partial-day runs differ from full-day runs.

def max_consecutive(arr):
    if not np.any(arr): return 0
    padded = np.pad(arr, (1, 1), mode='constant')
    diff = np.diff(padded.astype(int))
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]
    return np.max(ends - starts)

def categorize_segment(Y_clean, preds, E):
    residuals = np.abs(Y_clean - preds)
    max_res = np.max(residuals)
    
    if max_res <= 1.5 * E:
        out_10 = residuals > 1.0 * E
        if max_consecutive(out_10) < 3:
            return 1
            
    if max_res <= 2.0 * E:
        out_15 = residuals > 1.5 * E
        if max_consecutive(out_15) < 3:
            return 2
            
    return 3

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

from core_v2.math.fista_gpu import elasticnet_fista_cv

def screen_pipeline_cpu(X_raw, Y, groups):
    from sklearn.exceptions import ConvergenceWarning
    warnings.filterwarnings('ignore', category=ConvergenceWarning)
    
    try:
        grid = GroupLasso(groups=groups, group_reg=0.01, l1_reg=0.0, n_iter=100, tol=1e-2, fit_intercept=False, supress_warning=True)
        grid.fit(X_raw, Y)
        best_gl = grid
        best_group_reg = 0.01
    except:
        best_gl = GroupLasso(groups=groups, group_reg=0.01, l1_reg=0.0, n_iter=50, tol=1e-1, fit_intercept=False, supress_warning=True)
        try:
            best_gl.fit(X_raw, Y)
        except:
            return [], 0.01, 0.01
        best_group_reg = 0.01
        
    w = best_gl.coef_.flatten()
    active_idx = np.where(np.abs(w) > 1e-5)[0]
    
    if len(active_idx) == 0: return [], best_group_reg, 0.01
        
    X_surv = X_raw[:, active_idx]
    
    matrix_mb = (X_surv.nbytes + Y.nbytes) / (1024 * 1024)
    enet = ElasticNetCV(l1_ratio=0.5, cv=3, n_jobs=get_safe_n_jobs(matrix_mb), fit_intercept=False, max_iter=200, tol=1e-3)
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
        max_res = torch.max(residuals).item()
        
        res_cpu = residuals.squeeze().cpu().numpy()
        if max_res <= 1.5 * E:
            if max_consecutive(res_cpu > 1.0 * E) < 3:
                tier = 1
            elif max_res <= 2.0 * E and max_consecutive(res_cpu > 1.5 * E) < 3:
                tier = 2
            else:
                tier = 3
        elif max_res <= 2.0 * E and max_consecutive(res_cpu > 1.5 * E) < 3:
            tier = 2
        else:
            tier = 3
        
        tiers_dict[L] = tier
        max_residuals[L] = max_res
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
        return 8, [], [], 9999.0, []
        
    X_sub_t = X_t[:, active_idx]
    X_poly_t = poly_expand_gpu(X_sub_t)
    
    try:
        w_enet, _ = elasticnet_fista_cv(X_poly_t, Y_t, l1_ratio=0.5, cv=3, alphas=50)
        fixed_terms = torch.where(torch.abs(w_enet) > 1e-6)[0].cpu().numpy()
    except:
        return 8, [], [], 9999.0, active_idx.tolist() if hasattr(active_idx, "tolist") else active_idx
        
    max_features = max(1, length - 2)
    max_features = min(15, max_features)
    
    if len(fixed_terms) > max_features:
        fixed_terms = np.argsort(np.abs(w_enet.cpu().numpy()))[::-1][:max_features].copy()
        
    if len(fixed_terms) == 0:
        return 8, [], [], 9999.0, active_idx.tolist() if hasattr(active_idx, "tolist") else active_idx
        
    X_poly_fixed_t = X_poly_t[:, fixed_terms]
    beta_t = ols_fit_pytorch(X_poly_fixed_t, Y_t)
    
    if beta_t is None:
        return 8, [], [], 9999.0, active_idx.tolist() if hasattr(active_idx, "tolist") else active_idx
        
    beta_cpu = beta_t.squeeze().cpu().numpy().tolist()
    if not isinstance(beta_cpu, list):
        beta_cpu = [beta_cpu]
        
    preds_t = X_poly_fixed_t @ beta_t
    preds_cpu = preds_t.cpu().numpy().flatten()
    
    tier = categorize_segment(Y_cpu, preds_cpu, E)
    max_residual = float(np.max(np.abs(Y_cpu - preds_cpu)))
    
    return tier, fixed_terms.tolist(), beta_cpu, max_residual, active_idx.tolist() if hasattr(active_idx, "tolist") else active_idx

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--day', type=str, required=True, help="Format YYYY_MM_DD")
    parser.add_argument('--hours', type=int, default=24, help="Hours to process")
    parser.add_argument('--atlas_root', type=str, default="C:/Users/reyse/OneDrive/Desktop/Bayesian-AI/DATA/ATLAS", help="Path to ATLAS root")
    args = parser.parse_args()
    
    day = args.day
    hours = args.hours
    atlas_root = args.atlas_root
    
    print(f"[MAIN] Stage 1 Speed Pass starting for {day}")
    
    features_root = os.path.join(atlas_root, 'FEATURES_5s_v2')
    
    try:
        df = load_features([day], root=features_root)
    except Exception as e:
        print(f"[FATAL] Feature load failed: {e}")
        return
        
    if df is None or len(df) == 0: 
        return
        
    ohlcv = pd.read_parquet(os.path.join(atlas_root, '5s', f'{day}.parquet'))
    min_len = min(len(df), len(ohlcv))
    
    if hours < 24:
        max_bars = hours * 60 * 12
        min_len = min(min_len, max_bars)
        
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
    print(f"[DEBUG] Valid rows after NaN drop: {N_TOTAL}")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    X_global_t = torch.tensor(X_global, dtype=torch.float32, device=device)
    close_prices_t = torch.tensor(close_prices, dtype=torch.float32, device=device)
    groups = build_groups_from_columns(features_cols)
    
    sys.path.append(r"C:\Users\reyse\OneDrive\Desktop\Bayesian-AI")
    from core_v2.telemetry.reporter import TelemetryReporter
    day_reporter = TelemetryReporter(f"stage1_{day}")
    
    seed_start = 0
    segments = []
    prev_segment_delta = None
    output_json = f"artifacts/stage1_segments_{day}.json"
    
    while seed_start + SEED_BARS < N_TOTAL:
        day_reporter.update(seed_start, N_TOTAL, f"Day {day} (Stage 1)")
        t_seg0 = time.time()
        
        if prev_segment_delta is None:
            error_band = INITIAL_ERROR_BAND
        else:
            error_band = ERROR_BAND_FRACTION * prev_segment_delta
            
        print(f"[HUNT] Evaluating seed_start={seed_start} (E={error_band:.4f})")
            
        tier, fixed_terms, beta_cpu, max_residual, active_idx = evaluate_block(
            seed_start, SEED_BARS, error_band, X_global_t, close_prices_t, groups
        )
        
        if tier in [1, 2]:
            # We found a Pristine (Tier 1 or 2) block! Expand it!
            L_star = SEED_BARS
            
            max_forward = min(1000, N_TOTAL - seed_start) 
            if max_forward > SEED_BARS:
                X_exp_raw_t = X_global_t[seed_start:seed_start+max_forward, active_idx]
                X_exp_poly_t = poly_expand_gpu(X_exp_raw_t)[:, fixed_terms]
                Y_exp_t = (close_prices_t[seed_start:seed_start+max_forward] - close_prices_t[seed_start]).unsqueeze(1)
                
                tiers_exp, max_res_exp, betas_exp = batched_ols_scan_pytorch(X_exp_poly_t, Y_exp_t, SEED_BARS + 1, max_forward, error_band)
                
                for L_next in range(SEED_BARS + 1, max_forward + 1):
                    if L_next in tiers_exp and tiers_exp[L_next] <= tier:
                        L_star = L_next
                        beta_cpu = betas_exp[L_next]
                        max_residual = max_res_exp[L_next]
                    else:
                        break
                        
            print(f"  -> [CLEAN] Expanded Tier {tier} segment to length {L_star}")
            
            seg_data = {
                'day': day,
                'start_idx': seed_start,
                'end_idx': seed_start + L_star,
                'raw_start_idx': int(raw_indices[seed_start]),
                'raw_end_idx': int(raw_indices[seed_start + L_star]) if (seed_start + L_star) < N_TOTAL else int(raw_indices[-1] + 1),
                'length': L_star,
                'active_grid_cells': active_idx,
                'surviving_terms_count': len(fixed_terms),
                'surviving_polynomial_indices': fixed_terms,
                'beta_coefficients': beta_cpu,
                'max_residual': float(max_residual),
                'error_band_used': error_band,
                'volatility_tier': tier,
                'status': 'PRISTINE',
                'extraction_time_seconds': time.time() - t_seg0
            }
            
            segments.append(seg_data)
            with open(output_json, 'w') as f:
                json.dump(segments, f, indent=2)
                
            raw_clean_t = close_prices_t[seed_start:seed_start+L_star]
            new_delta = float(torch.max(raw_clean_t) - torch.min(raw_clean_t))
            if new_delta > 0:
                prev_segment_delta = new_delta
            seed_start += L_star
            
        else:
            # Did not fit Tier 1 or 2 cleanly. Hunt forward to isolate the pristine gap.
            found_k = None
            max_hunt = min(500, N_TOTAL - seed_start - SEED_BARS)
            
            print(f"  -> [MESSY] Initial seed was Tier {tier}. Hunting forward for Tier 1 or 2...")
            
            for k in range(1, max_hunt):
                tier_k, _, _, _, _ = evaluate_block(
                    seed_start + k, SEED_BARS, error_band, X_global_t, close_prices_t, groups
                )
                if tier_k in [1, 2]:
                    found_k = k
                    break
                    
            if found_k is None:
                found_k = max(1, max_hunt)
                
            print(f"  -> [ISOLATE] Clean regime found at +{found_k} bars! Boxing the {found_k}-bar chaos segment.")
            
            seg_data = {
                'day': day,
                'start_idx': seed_start,
                'end_idx': seed_start + found_k,
                'raw_start_idx': int(raw_indices[seed_start]),
                'raw_end_idx': int(raw_indices[seed_start + found_k]) if (seed_start + found_k) < N_TOTAL else int(raw_indices[-1] + 1),
                'length': found_k,
                'active_grid_cells': [],
                'surviving_terms_count': 0,
                'surviving_polynomial_indices': [],
                'beta_coefficients': [],
                'max_residual': 9999.0,
                'error_band_used': error_band,
                'volatility_tier': 99,
                'status': 'UNPROCESSED_CHAOS',
                'extraction_time_seconds': time.time() - t_seg0
            }
            segments.append(seg_data)
            
            with open(output_json, 'w') as f:
                json.dump(segments, f, indent=2)
                
            # DO NOT update prev_segment_delta with messy block values!
            seed_start += found_k
            
    day_reporter.clear()

if __name__ == "__main__":
    main()
