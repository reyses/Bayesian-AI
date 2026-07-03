import numpy as np
import pandas as pd
import glob
import os

"""Local amplitude-regime scale for the labeler (hindsight/centered allowed).

Anchor every 1m bar at its close; first return THROUGH that level = one oscillation:
per = first-return time (min), amp = peak |excursion| before return (pt). Regime scale =
windowed median of amp/sqrt(per), rescaled to the 10-min reference period:
    scale(t) = median_{anchors near t}(amp/sqrt(per)) * sqrt(REF_PERIOD_MIN)
Same semantics as amplitude_evolution.py's ref_amp ('typical ~10-min swing'), but computable
at any window. Uses amp/sqrt(per) over all returned anchors (not the 8-15min band) so small
windows keep enough samples; the sqrt rescale removes the period-mix dependence.
Provenance: research/recovery_dynamics/tools/anchor_period.py (day_periods),
amplitude_evolution.py (vol_scale/ref_amp). Hindsight-legal for LABELS only.
Note: Anchors near EOD are censoring-truncated (only short periods return), causing a slight low bias at the close.
"""

AMP_MAXLOOK_MIN     = 360   # forward first-return cap (min) — mirrors anchor_period.MAXLOOK
REF_PERIOD_MIN      = 10.0  # reference period: center of the 8-15min ref_amp band
MIN_WINDOW_SAMPLES  = 30    # fewer returned anchors than this in a window -> widen x2 (median too noisy)
MIN_DAY_SAMPLES     = 50    # amplitude_evolution's day-skip criterion, reused as day-fallback trigger
GLOBAL_FALLBACK_PT  = 8.0   # last-resort scale: pooled 2024-25 8-15min median amp ~8-12pt (anchor_period.md)

def anchor_samples(close: np.ndarray):
    """
    Returns (idx_array, ratio_array)
    where ratio = amp/sqrt(per), returned anchors only.
    """
    n = len(close)
    idxs = []
    ratios = []
    
    for i in range(n):
        anchor_val = close[i]
        
        # Skip flat-next-bar
        if i + 1 < n and close[i+1] == anchor_val:
            continue
            
        lookahead = min(n - i, AMP_MAXLOOK_MIN + 1)
        future = close[i+1:i+lookahead]
        if len(future) == 0:
            continue
            
        is_up = future[0] > anchor_val
        
        if is_up:
            returns = np.where(future <= anchor_val)[0]
        else:
            returns = np.where(future >= anchor_val)[0]
            
        if len(returns) > 0:
            ret_idx = returns[0]
            per = ret_idx + 1
            if per > 0:
                segment = future[:ret_idx]
                if len(segment) > 0:
                    if is_up:
                        amp = np.max(segment) - anchor_val
                    else:
                        amp = anchor_val - np.min(segment)
                    
                    idxs.append(i)
                    ratios.append(amp / np.sqrt(per))
                    
    return np.array(idxs, dtype=int), np.array(ratios, dtype=float)

def scale_series(close: np.ndarray, window_min: int):
    """
    centered ±window_min median(ratio) * sqrt(REF_PERIOD_MIN) -> float[n]
    single-day only, no filesystem access. Window widening is its ONLY fallback; if even the
    whole day is unusable it returns all-NaN and the CALLER resolves.
    """
    idxs, ratios = anchor_samples(close)
    n = len(close)
    out = np.full(n, np.nan)
    
    if len(ratios) < MIN_DAY_SAMPLES:
        return out
        
    for i in range(n):
        w = window_min
        while True:
            left = i - w
            right = i + w
            
            start_idx = np.searchsorted(idxs, left)
            end_idx = np.searchsorted(idxs, right, side='right')
            
            slice_len = end_idx - start_idx
            if slice_len >= MIN_WINDOW_SAMPLES:
                med = np.median(ratios[start_idx:end_idx])
                out[i] = med * np.sqrt(REF_PERIOD_MIN)
                break
            else:
                w *= 2
                if w > n: # whole day
                    if len(ratios) >= MIN_DAY_SAMPLES:
                        out[i] = np.median(ratios) * np.sqrt(REF_PERIOD_MIN)
                    break
                    
    return out

def scale_scalar(close: np.ndarray):
    """
    whole-day median(ratio) * sqrt(REF_PERIOD_MIN); np.nan if
    fewer than MIN_DAY_SAMPLES returned anchors (caller resolves — no fallback logic here).
    """
    idxs, ratios = anchor_samples(close)
    if len(ratios) < MIN_DAY_SAMPLES:
        return np.nan
    return np.median(ratios) * np.sqrt(REF_PERIOD_MIN)

def scale_for_day(date_key: str, close: np.ndarray, mode: str, one_m_dir: str, cache: dict):
    """
    OWNS all cross-day context and ALL day-level fallbacks.
    Returns array of size n.
    """
    # Initialize cache
    if ("files",) not in cache:
        # Sort files to ensure chronological order for neighbor-day modes
        files = sorted(glob.glob(os.path.join(one_m_dir, "*.parquet")))
        cache[("files",)] = files
    files = cache[("files",)]
    
    def _get_close(dk):
        key = ("close", dk)
        if key not in cache:
            target = os.path.join(one_m_dir, f"{dk}.parquet")
            if not os.path.exists(target):
                cache[key] = None
            else:
                df = pd.read_parquet(target)
                cache[key] = df['close'].values
        return cache[key]
        
    def _get_scalar(dk):
        key = ("scalar", dk)
        if key not in cache:
            c = _get_close(dk)
            if c is None:
                cache[key] = np.nan
            else:
                cache[key] = scale_scalar(c)
        return cache[key]
        
    n = len(close)
    
    if mode.startswith("w"):
        w = int(mode[1:])
        raw = scale_series(close, w)
        if np.isnan(raw).all():
            mode = "day" # Fallback to day scalar
        else:
            day_med = np.nanmedian(raw)
            if np.isnan(day_med):
                mode = "day"
            else:
                raw[np.isnan(raw)] = day_med
                return raw
                
    if mode == "day":
        val = _get_scalar(date_key)
        if np.isnan(val):
            # Nearest available neighbor fallback
            base_idx = next((i for i, f in enumerate(files) if date_key in f), -1)
            if base_idx == -1:
                val = GLOBAL_FALLBACK_PT
            else:
                offset = 1
                while np.isnan(val) and offset < len(files):
                    for d_idx in [base_idx - offset, base_idx + offset]:
                        if 0 <= d_idx < len(files):
                            neighbor_dk = os.path.basename(files[d_idx]).split('.')[0]
                            val = _get_scalar(neighbor_dk)
                            if not np.isnan(val):
                                break
                    offset += 1
                if np.isnan(val):
                    val = GLOBAL_FALLBACK_PT
        return np.full(n, val)
        
    if mode.startswith("day_c"):
        w = int(mode[5:])
        base_idx = next((i for i, f in enumerate(files) if date_key in f), -1)
        
        if base_idx == -1:
            return np.full(n, GLOBAL_FALLBACK_PT)
            
        half_w = w // 2
        start_idx = max(0, base_idx - half_w)
        end_idx = min(len(files), base_idx + half_w + 1)
        
        vals = []
        for i in range(start_idx, end_idx):
            dk = os.path.basename(files[i]).split('.')[0]
            vals.append(_get_scalar(dk))
            
        med = np.nanmedian(vals)
        if np.isnan(med):
            med = GLOBAL_FALLBACK_PT
            
        return np.full(n, med)
        
    raise ValueError(f"Unknown mode: {mode}")
