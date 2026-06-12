import os
import numpy as np
import pandas as pd
from typing import Optional, List

from core_v2.features import load_features, TF_ORDER, TF_SECONDS
from core_v2.build_dataset import _last_closed_idx  # canonical lookahead-safe alignment
import core_v2.statistical_field_engine as sfe

# Constants with provenance
EPS = 0.1           # Log-floor for log|z|. NOT log(0) protection: z crosses ~0 routinely
                    # (price at the regression mean); without a substantial floor each
                    # crossing injects a log spike (~-14 at 1e-6) that swamps the k-window
                    # OLS slope. 0.1 caps the spike at ~-2.3 (per spec).
K_SWEEP = (12, 21, 30)
VR_WINDOWS = (10, 60)
Z_REF = 2.0         # V1 exact z_21 entry threshold
Z_EXIT_REF = 0.5    # V1 exact z_21 exit threshold

# Proxy mapping for cross-TF sigma ratio (fast -> slow)
PROXY_MAP = {
    '5s': '1m',
    '15s': '5m',
    '1m': '15m',
    '5m': '1h',
    '15m': '4h',
    '1h': '1D',
    '4h': '1D',
    '1D': None
}


def derive_day(
    day: str,
    atlas_root: str = "DATA/ATLAS",
    features_root: str = "DATA/ATLAS/FEATURES_5s_v2"
) -> pd.DataFrame:
    """
    Derives NMP state variables (lambda_hat, vr_exact, vr_proxy, z_21) for a single day.
    Aligned to the 5s anchor grid using _last_closed_idx semantics.
    """
    layers = ['L0', 'L2', 'L3']
    try:
        v2_df = load_features(days=[day], tfs=TF_ORDER, layers=layers, root=features_root, require_all=True)
    except FileNotFoundError:
        return pd.DataFrame()
        
    if len(v2_df) == 0:
        return pd.DataFrame()
        
    anchor_times = v2_df['timestamp'].values
    result_df = pd.DataFrame({'timestamp': anchor_times})
    
    for i, tf in enumerate(TF_ORDER):
        raw_path = os.path.join(atlas_root, tf, f"{day}.parquet")
        if not os.path.exists(raw_path):
            continue
            
        raw_df = pd.read_parquet(raw_path)
        if len(raw_df) == 0:
            continue
            
        closes = raw_df['close'].values
        raw_times = raw_df['timestamp'].values
        n_bars = len(closes)
        
        # 1. vr_exact
        w_fast, w_slow = VR_WINDOWS
        std_fast = pd.Series(closes).rolling(w_fast).std(ddof=1).values
        std_slow = pd.Series(closes).rolling(w_slow).std(ddof=1).values
        safe_std_slow = np.where(std_slow == 0, np.nan, std_slow)
        vr_exact = std_fast / safe_std_slow
        
        # 2. z_21
        window = 21
        z_21 = np.full(n_bars, np.nan)
        x = np.arange(window)
        x_mean = np.mean(x)
        x_var_sum = np.sum((x - x_mean)**2)
        
        for k in range(window - 1, n_bars):
            y = closes[k - window + 1 : k + 1]
            y_mean = np.mean(y)
            cov = np.sum((x - x_mean) * (y - y_mean))
            if x_var_sum > 0:
                slope = cov / x_var_sum
                intercept = y_mean - slope * x_mean
                endpoint = slope * (window - 1) + intercept
                resids = y - (slope * x + intercept)
                
                if window > 2:
                    var_resid = np.sum(resids**2) / (window - 2)
                    if var_resid > 0:
                        std_resid = np.sqrt(var_resid)
                        z_21[k] = (y[-1] - endpoint) / std_resid
                        
        # Canonical lookahead-safe alignment (build_dataset._last_closed_idx).
        # NOTE: a bare searchsorted(raw_times, anchor_times) selects the still-OPEN
        # bar (ATLAS bars are open-stamped) -> intra-bar lookahead for vr_exact/z_21
        # at the anchor level (the V1 sin). The `- period` inside _last_closed_idx
        # is the load-bearing part.
        idx = _last_closed_idx(raw_times, anchor_times, TF_SECONDS[tf])
        valid_mask = idx >= 0
        safe_idx = np.where(valid_mask, idx, 0)
        
        result_df[f'L3_{tf}_vr_exact'] = np.where(valid_mask, vr_exact[safe_idx], np.nan)
        result_df[f'L3_{tf}_z_21'] = np.where(valid_mask, z_21[safe_idx], np.nan)
        
        # 3. lambda_hat
        n_base = sfe.N_BASE[tf]
        col_z_se = f'L3_{tf}_z_se_{n_base}'
        
        if col_z_se in v2_df.columns:
            z_se_5s = v2_df[col_z_se].values
            
            # Extract closed-bar sequence from 5s-anchored piecewise constant data
            z_se_closed = np.full(n_bars, np.nan)
            z_se_closed[idx[valid_mask]] = z_se_5s[valid_mask]
            
            log_z = np.log(np.abs(z_se_closed) + EPS)
            
            for k in K_SWEEP:
                lambda_hat = np.full(n_bars, np.nan)
                lambda_se = np.full(n_bars, np.nan)
                lambda_t = np.full(n_bars, np.nan)
                
                x_k = np.arange(k)
                x_k_mean = np.mean(x_k)
                x_k_var_sum = np.sum((x_k - x_k_mean)**2)
                
                for j in range(k - 1, n_bars):
                    y_k = log_z[j - k + 1 : j + 1]
                    if np.isnan(y_k).any():
                        continue
                        
                    y_k_mean = np.mean(y_k)
                    cov_k = np.sum((x_k - x_k_mean) * (y_k - y_k_mean))
                    if x_k_var_sum > 0:
                        slope_k = cov_k / x_k_var_sum
                        intercept_k = y_k_mean - slope_k * x_k_mean
                        resids_k = y_k - (slope_k * x_k + intercept_k)
                        
                        if k > 2:
                            var_resid_k = np.sum(resids_k**2) / (k - 2)
                            if var_resid_k > 0:
                                se_k = np.sqrt(var_resid_k / x_k_var_sum)
                                t_stat = slope_k / se_k
                            else:
                                se_k = 0.0
                                t_stat = 0.0
                        else:
                            se_k = np.nan
                            t_stat = np.nan
                            
                        lambda_hat[j] = slope_k
                        lambda_se[j] = se_k
                        lambda_t[j] = t_stat
                        
                result_df[f'L3_{tf}_lambda_hat_{k}'] = np.where(valid_mask, lambda_hat[safe_idx], np.nan)
                result_df[f'L3_{tf}_lambda_se_{k}'] = np.where(valid_mask, lambda_se[safe_idx], np.nan)
                result_df[f'L3_{tf}_lambda_t_{k}'] = np.where(valid_mask, lambda_t[safe_idx], np.nan)
                
        # 4. vr_proxy
        slow_tf = PROXY_MAP.get(tf)
        if slow_tf is not None:
            fast_n = sfe.N_BASE[tf]
            slow_n = sfe.N_BASE[slow_tf]
            col_fast = f'L2_{tf}_price_sigma_{fast_n}'
            col_slow = f'L2_{slow_tf}_price_sigma_{slow_n}'
            
            if col_fast in v2_df.columns and col_slow in v2_df.columns:
                sigma_fast = v2_df[col_fast].values
                sigma_slow = v2_df[col_slow].values
                safe_sigma_slow = np.where(sigma_slow == 0, np.nan, sigma_slow)
                
                # proxy is naturally on the 5s anchor grid because we use v2_df columns
                result_df[f'L3_{tf}_vr_proxy'] = sigma_fast / safe_sigma_slow
                
    return result_df
