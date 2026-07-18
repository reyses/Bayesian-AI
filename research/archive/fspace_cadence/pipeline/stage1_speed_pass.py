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
from tqdm import tqdm

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

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))
from core_v2.features import load_features

# === PARAMETERS ===
SEED_BARS = 30                # bars in the initial probe block (~2.5 min at 5s)
INITIAL_ERROR_BAND = 1.00     # price-point residual tolerance for the FIRST segment
                             # (no prior segment delta exists yet to scale from)
ERROR_BAND_FRACTION = 0.10    # LEGACY (prior-segment-range band) — unused under the per-bar rule below.
                             # NOTE: this made tiers path-dependent on processing order.
# --- Per-bar self-scaling break rule (replaces the lagged prior-range band) ---
BAND_FRAC = 0.10             # tolerance = 10% of the CURRENT bar's |price delta|
TICK_FLOOR = 0.25            # 1 tick = least allowance (floor for flat bars)
BREAK_CONSEC = 5             # break a segment after this many consecutive 'off' bars
FLIP_TAIL_BARS = 20          # trace the R-curve this many bars PAST the break (distortion-through-flip)
FLIP_LOOKBACK_BARS = 15      # residuals kept just before the break (pre-flip signature; v2)

from tiering import classify_tier, max_consecutive

_DELTA_ABS = None  # set in main(): per-bar |close diff| on the filtered close series

def _seg_clean(residuals, delta_abs):
    """Per-bar break rule: bar 'off' if fit residual > 10% of that bar's |price move|
    (floored at 1 tick); segment stays clean until BREAK_CONSEC consecutive off-bars."""
    thr = np.maximum(BAND_FRAC * np.abs(delta_abs), TICK_FLOOR)
    return max_consecutive(residuals > thr) < BREAK_CONSEC

def build_groups_from_columns(columns):
    groups = []
    tf_map = {'5s':0, '15s':1, '1m':2, '3m':3, '5m':4, '10m':5, '15m':6, '1h':7, '4h':8, '1D':9}
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
    enet = ElasticNetCV(l1_ratio=0.5, cv=3, n_jobs=1, fit_intercept=False, max_iter=200, tol=1e-3)
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

def batched_ols_scan_pytorch(X_t, Y_t, min_bars, max_bars, E, delta_abs_win):
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
    r2_dict = {}
    noff_dict = {}
    consec_dict = {}

    try:
        betas = torch.linalg.solve(XtX_batch, XtY_batch)
    except Exception as e:
        return {}, {}, {}, {}, {}, {}

    for i, L in enumerate(range(min_bars, max_bars + 1)):
        beta = betas[i]
        X_sub = X_t[:L]
        Y_sub = Y_t[:L]

        preds = X_sub @ beta
        residuals = torch.abs(Y_sub - preds)
        max_res = torch.max(residuals).item()

        res_cpu = np.atleast_1d(residuals.squeeze().cpu().numpy())
        # R-curve quantities at length L (the map's raw material)
        y_np = np.atleast_1d(Y_sub.squeeze().cpu().numpy())
        ss_res = float(np.sum(res_cpu ** 2))
        ss_tot = float(np.sum((y_np - y_np.mean()) ** 2))
        r2_dict[L] = (1.0 - ss_res / ss_tot) if ss_tot > 1e-12 else 0.0
        off = res_cpu > np.maximum(BAND_FRAC * np.abs(delta_abs_win[:L]), TICK_FLOOR)
        noff_dict[L] = int(np.sum(off))
        consec_dict[L] = int(max_consecutive(off))

        tiers_dict[L] = 1 if consec_dict[L] < BREAK_CONSEC else 99
        max_residuals[L] = max_res
        betas_dict[L] = beta.squeeze().cpu().numpy().tolist()

    return tiers_dict, max_residuals, betas_dict, r2_dict, noff_dict, consec_dict

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
    
    X_poly_cpu = X_poly_t.cpu().numpy()
    enet2 = ElasticNetCV(l1_ratio=0.5, cv=3, n_jobs=1, fit_intercept=False, max_iter=200, tol=1e-3)
    
    try:
        enet2.fit(X_poly_cpu, Y_cpu)
        w_enet = enet2.coef_
        fixed_terms = np.where(np.abs(w_enet) > 1e-6)[0]
    except:
        return 8, [], [], 9999.0, active_idx.tolist() if hasattr(active_idx, "tolist") else active_idx
        
    max_features = min(40, length - 2)
    max_features = max(1, max_features)
    
    if len(fixed_terms) > max_features:
        fixed_terms = np.argsort(np.abs(w_enet))[::-1][:max_features].copy()
        
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
    
    residuals_cpu = np.abs(Y_cpu - preds_cpu)
    delta_seg = _DELTA_ABS[start_idx:start_idx + len(residuals_cpu)]
    tier = 1 if _seg_clean(residuals_cpu, delta_seg) else 99
    max_residual = float(np.max(residuals_cpu))
    
    return tier, fixed_terms.tolist(), beta_cpu, max_residual, active_idx.tolist() if hasattr(active_idx, "tolist") else active_idx

def main():
    global SEED_BARS
    parser = argparse.ArgumentParser()
    parser.add_argument('--day', type=str, required=True, help="Format YYYY_MM_DD")
    parser.add_argument('--hours', type=int, default=24, help="Hours to process")
    parser.add_argument('--atlas_root', type=str, default=os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "DATA", "ATLAS")), help="Path to ATLAS root")
    parser.add_argument('--features_root', type=str, default=None, help="Path to custom features directory")
    parser.add_argument('--run_name', type=str, default="default", help="Name of run for artifact naming")
    parser.add_argument('--tf', type=str, default='5s', help="Base timeframe to load (5s or 1s)")
    parser.add_argument('--seed_bars', type=int, default=SEED_BARS, help="seed block size (floor on regime length); lower to reveal sub-floor regimes")
    args = parser.parse_args()
    SEED_BARS = args.seed_bars
    
    day = args.day
    hours = args.hours
    atlas_root = args.atlas_root
    
    print(f"[MAIN] Stage 1 Speed Pass starting for {day} | Run: {args.run_name} | TF: {args.tf}")
    
    features_root = args.features_root if args.features_root else os.path.join(atlas_root, 'FEATURES_5s_v2')
    
    try:
        if "RUN_A" in features_root or "RUN_B" in features_root or "RUN_C" in features_root:
            df = pd.read_parquet(os.path.join(features_root, f'{day}.parquet'))
        else:
            df = load_features([day], root=features_root, require_all=False)
    except Exception as e:
        print(f"[FATAL] Feature load failed: {e}")
        return
        
    if df is None or len(df) == 0: 
        return
        
    df = df.dropna(axis=1, how='all')
        
    ohlcv = pd.read_parquet(os.path.join(atlas_root, args.tf, f'{day}.parquet'))
    min_len = min(len(df), len(ohlcv))
    
    if hours < 24:
        max_idx = int(hours * 3600 / (5 if args.tf == '5s' else 1))
        min_len = min(min_len, max_idx)
        
    df = df.iloc[:min_len].copy()
    ohlcv = ohlcv.iloc[:min_len].copy()
    
    valid_mask = ~df.isnull().any(axis=1)
    df = df[valid_mask]
    ohlcv = ohlcv[valid_mask]
    
    print(f"[DEBUG] Valid rows after NaN drop: {len(df)}")
    
    if len(df) == 0:
        return
    
    features_cols = [c for c in df.columns if c != 'timestamp' and 'price_mean' not in c and 'vwap' not in c]
        
    scaler = StandardScaler()
    X_global = scaler.fit_transform(df[features_cols].values)
    
    valid_idx = ~np.isnan(X_global).any(axis=1)
    raw_indices = np.where(valid_idx)[0]
    X_global = X_global[valid_idx]
    close_prices = ohlcv['close'].values[valid_idx]
    
    N_TOTAL = len(X_global)
    global _DELTA_ABS
    _DELTA_ABS = np.abs(np.diff(close_prices, prepend=close_prices[0]))
    print(f"[DEBUG] Valid rows after NaN drop: {N_TOTAL}")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    X_global_t = torch.tensor(X_global, dtype=torch.float32, device=device)
    close_prices_t = torch.tensor(close_prices, dtype=torch.float32, device=device)
    groups = build_groups_from_columns(features_cols)
    
    # Dynamically locate repository root
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    if repo_root not in sys.path:
        sys.path.append(repo_root)
    from core_v2.telemetry.reporter import TelemetryReporter
    day_reporter = TelemetryReporter(f"stage1_{args.run_name}_{day}")  # run_name-keyed: avoids same-day concurrency clobber
    
    seed_start = 0
    segments = []
    prev_segment_delta = None
    output_json = f"artifacts/stage1_{args.run_name}_segments_{day}.json"
    
    pbar = tqdm(total=N_TOTAL, desc=f"Scanning {day}", unit="bars")
    
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
                
                delta_win = _DELTA_ABS[seed_start:seed_start + max_forward]
                tiers_exp, max_res_exp, betas_exp, r2_exp, noff_exp, consec_exp = batched_ols_scan_pytorch(X_exp_poly_t, Y_exp_t, SEED_BARS + 1, max_forward, error_band, delta_win)
                
                for L_next in range(SEED_BARS + 1, max_forward + 1):
                    if L_next in tiers_exp and tiers_exp[L_next] <= tier:
                        L_star = L_next
                        beta_cpu = betas_exp[L_next]
                        max_residual = max_res_exp[L_next]
                    else:
                        break

                # MAP ENRICHMENT: keep the full R-curve (incl. FLIP_TAIL_BARS past the break)
                curve_end = min(L_star + FLIP_TAIL_BARS, max_forward)
                r_curve = [{'L': Lc, 'r2': round(float(r2_exp[Lc]), 5),
                            'max_resid': round(float(max_res_exp[Lc]), 4),
                            'n_off': int(noff_exp[Lc]), 'consec_off': int(consec_exp[Lc])}
                           for Lc in range(SEED_BARS + 1, curve_end + 1) if Lc in r2_exp]
                L_break = L_star + 1
                break_reason = 'end_of_data' if L_break > max_forward else 'consec_off'
                consec_at_break = int(consec_exp.get(L_break, 0))
            else:
                r_curve, break_reason, consec_at_break = [], 'no_expansion', 0

            print(f"  -> [CLEAN] Expanded Tier {tier} segment to length {L_star}")
            cp_seg = close_prices[seed_start:seed_start + L_star]
            net_move = float(cp_seg[-1] - cp_seg[0]) if len(cp_seg) > 1 else 0.0
            slope_ps = float(np.polyfit(np.arange(len(cp_seg)), cp_seg, 1)[0]) if len(cp_seg) > 1 else 0.0

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
                'r_curve': r_curve,
                'slope_pts_per_s': slope_ps,
                'net_move_pts': net_move,
                'direction': int(np.sign(net_move)),
                'break_reason': break_reason,
                'consec_off_at_break': consec_at_break,
                'extraction_time_seconds': time.time() - t_seg0
            }
            
            segments.append(seg_data)
            if len(segments) % 100 == 0:
                with open(output_json, 'w') as f:
                    json.dump(segments, f)

            raw_clean_t = close_prices_t[seed_start:seed_start+L_star]
            new_delta = float(torch.max(raw_clean_t) - torch.min(raw_clean_t))
            if new_delta > 0:
                prev_segment_delta = new_delta
            pbar.update(L_star)
            seed_start += L_star
            
        else:
            # Did not fit Tier 1 or 2 cleanly. Hunt forward to isolate the pristine gap.
            found_k = None
            max_hunt = min(500, N_TOTAL - seed_start - SEED_BARS)
            
            print(f"  -> [MESSY] Initial seed was Tier {tier}. Hunting forward for Tier 1 or 2...")
            
            STRIDE = 5
            if max_hunt <= STRIDE:
                for k in range(1, max_hunt):
                    tier_k, _, _, _, _ = evaluate_block(
                        seed_start + k, SEED_BARS, error_band, X_global_t, close_prices_t, groups
                    )
                    if tier_k in [1, 2]:
                        found_k = k
                        break
            else:
                found_coarse = None
                for k in range(STRIDE, max_hunt, STRIDE):
                    tier_k, _, _, _, _ = evaluate_block(
                        seed_start + k, SEED_BARS, error_band, X_global_t, close_prices_t, groups
                    )
                    if tier_k in [1, 2]:
                        found_coarse = k
                        break
                        
                if found_coarse is None:
                    found_k = max(1, max_hunt)
                else:
                    found_k = found_coarse
                    for k in range(found_coarse - 1, max(found_coarse - STRIDE, 0), -1):
                        tier_k, _, _, _, _ = evaluate_block(
                            seed_start + k, SEED_BARS, error_band, X_global_t, close_prices_t, groups
                        )
                        if tier_k in [1, 2]:
                            found_k = k
                        else:
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
            if len(segments) % 100 == 0:
                with open(output_json, 'w') as f:
                    json.dump(segments, f)

            # DO NOT update prev_segment_delta with messy block values!
            seed_start += found_k

    with open(output_json, 'w') as f:   # final flush (full segment list incl. r_curves)
        json.dump(segments, f)
    print(f"[MAIN] wrote {len(segments)} segments -> {output_json}")
    day_reporter.clear()

if __name__ == "__main__":
    main()
