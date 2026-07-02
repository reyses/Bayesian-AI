import numpy as np
import pandas as pd

def _slope(arr, lb, n):
    s = np.full(n, np.nan)
    if n > lb:
        s[lb:] = (arr[lb:] - arr[:-lb]) / lb
    return s

def _delta(arr, lb, n):
    d = np.full(n, np.nan)
    if n > lb:
        d[lb:] = (arr[lb:] - arr[:-lb]) / lb
    return d

def cross_history(fast: np.ndarray, slow: np.ndarray):
    """Returns (bars_since_cross_up, bars_since_cross_dn).
    bars_since == -1 means no such crossing yet in this window."""
    diff = fast - slow
    sign = np.where(diff > 0, 1, np.where(diff < 0, -1, 0))
    n = len(sign)
    b_up = np.full(n, -1, dtype=int)
    b_dn = np.full(n, -1, dtype=int)
    last_up = -1
    last_dn = -1
    for i in range(1, n):
        if not (np.isfinite(diff[i]) and np.isfinite(diff[i-1])):
            continue
        # Sign flip from <=0 to >0 = upcross (fast crosses above slow)
        if sign[i-1] <= 0 and sign[i] > 0:
            last_up = i
        if sign[i-1] >= 0 and sign[i] < 0:
            last_dn = i
        if last_up >= 0:
            b_up[i] = i - last_up
        if last_dn >= 0:
            b_dn[i] = i - last_dn
    return b_up, b_dn

def _pct_rank_array(values: np.ndarray, window: int) -> np.ndarray:
    """Rolling percentile rank of values within last `window` bars. Vectorized for speed."""
    import pandas as pd
    s = pd.Series(values)
    return s.rolling(window, min_periods=2).apply(
        lambda x: (x < x.iloc[-1]).sum() / len(x.dropna()) if len(x.dropna()) >= 2 else 0.5,
        raw=False
    ).values

def compute_primitive_arrays(close: np.ndarray, anchors: dict):
    """
    Computes all causal feature arrays for a given sequence of close prices and anchors.
    anchors dict should contain keys like: M_15s, S_15s, M_1m, S_1m, M_15m, S_15m, Mh_1h, Sh_1h, etc.
    Returns a dictionary of feature arrays of the same length as `close`.
    """
    n = len(close)
    
    # Safely extract anchors (fallback to NaNs if missing)
    def _get(key):
        return anchors.get(key, np.full(n, np.nan))
        
    M_15s, S_15s = _get('M_15s'), _get('S_15s')
    M_1m,  S_1m  = _get('M_1m'),  _get('S_1m')
    M_15m, S_15m = _get('M_15m'), _get('S_15m')
    Mh,    Sh    = _get('Mh_1h'), _get('Sh_1h')
    Ml,    Sl    = _get('Ml_1h'), _get('Sl_1h')
    Mc,    Sc    = _get('Mc_1h'), _get('Sc_1h')

    # Slopes
    slope_15s_3m  = _slope(M_15s, 3, n)
    slope_15s_10m = _slope(M_15s, 10, n)
    slope_1m_10m  = _slope(M_1m,  10, n)
    slope_15m_5m  = _slope(M_15m, 5, n)
    slope_15m_15m = _slope(M_15m, 15, n)
    slope_15m_decel = slope_15m_5m - slope_15m_15m
    curv_15m = slope_15m_decel / 10.0

    # Compression
    band_width = Mh - Ml
    band_rank_60 = _pct_rank_array(band_width, 60)
    sigma_15m_rank_60 = _pct_rank_array(S_15m, 60)

    # CRM Crossings
    bu_15s_1m,  bd_15s_1m  = cross_history(M_15s, M_1m)
    bu_1m_15m,  bd_1m_15m  = cross_history(M_1m,  M_15m)
    bu_15s_15m, bd_15s_15m = cross_history(M_15s, M_15m)
    bu_px_15m,  bd_px_15m  = cross_history(close, M_15m)
    bu_15s_Mh,  bd_15s_Mh  = cross_history(M_15s, Mh)
    bu_15s_Ml,  bd_15s_Ml  = cross_history(M_15s, Ml)
    bu_15m_Mh,  bd_15m_Mh  = cross_history(M_15m, Mh)
    bu_15m_Ml,  bd_15m_Ml  = cross_history(M_15m, Ml)

    # CRM Distances
    safe_S_1m = np.where(S_1m > 0, S_1m, np.nan)
    safe_S_15m = np.where(S_15m > 0, S_15m, np.nan)
    dist_15s_1m = (M_15s - M_1m) / safe_S_1m
    dist_1m_15m = (M_1m  - M_15m) / safe_S_15m
    dist_15s_15m = (M_15s - M_15m) / safe_S_15m
    fan_width = np.abs(dist_15s_1m) + np.abs(dist_1m_15m) + np.abs(dist_15s_15m)

    delta_dist_15s_1m_10m  = _delta(dist_15s_1m, 10, n)
    delta_dist_1m_15m_10m  = _delta(dist_1m_15m, 10, n)
    delta_dist_15s_15m_10m = _delta(dist_15s_15m, 10, n)

    # Z-scores
    def _z_arr(num, denom):
        return np.where((denom > 0) & (~np.isnan(num)) & (~np.isnan(denom)), (close - num) / denom, np.nan)

    z_15s = _z_arr(M_15s, S_15s)
    z_1m = _z_arr(M_1m, S_1m)
    z_15m = _z_arr(M_15m, S_15m)
    z_1h_high = _z_arr(Mh, Sh)
    z_1h_low = _z_arr(Ml, Sl)
    z_1h_close = _z_arr(Mc, Sc)

    features = {
        'z_15s': z_15s, 'z_1m': z_1m, 'z_15m': z_15m,
        'z_1h_high': z_1h_high, 'z_1h_low': z_1h_low, 'z_1h_close': z_1h_close,
        'slope_15s_3m': slope_15s_3m, 'slope_15s_10m': slope_15s_10m,
        'slope_1m_10m': slope_1m_10m, 'slope_15m_5m': slope_15m_5m,
        'slope_15m_15m': slope_15m_15m, 'slope_15m_decel': slope_15m_decel,
        'curv_15m': curv_15m,
        'band_width': band_width, 'band_rank_60': band_rank_60,
        'sigma_15m_rank_60': sigma_15m_rank_60,
        
        # Cross history
        'bu_15s_1m': bu_15s_1m, 'bd_15s_1m': bd_15s_1m,
        'bu_1m_15m': bu_1m_15m, 'bd_1m_15m': bd_1m_15m,
        'bu_15s_15m': bu_15s_15m, 'bd_15s_15m': bd_15s_15m,
        'bu_px_15m': bu_px_15m, 'bd_px_15m': bd_px_15m,
        'bu_15s_Mh': bu_15s_Mh, 'bd_15s_Mh': bd_15s_Mh,
        'bu_15s_Ml': bu_15s_Ml, 'bd_15s_Ml': bd_15s_Ml,
        'bu_15m_Mh': bu_15m_Mh, 'bd_15m_Mh': bd_15m_Mh,
        'bu_15m_Ml': bu_15m_Ml, 'bd_15m_Ml': bd_15m_Ml,

        # Fan
        'fan_width': fan_width,
        'delta_dist_15s_1m_10m': delta_dist_15s_1m_10m,
        'delta_dist_1m_15m_10m': delta_dist_1m_15m_10m,
        'delta_dist_15s_15m_10m': delta_dist_15s_15m_10m,
    }
    
    return features
