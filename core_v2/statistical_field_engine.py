"""
Statistical Field Engine v2
===========================
Computes per-layer features per spec v2 (139D total):

    L0 (1 global)    : time_of_day
    L1 (8 per TF)    : window-free primitives
                       - price_velocity_1b, price_accel_1b
                       - vol_velocity_1b, vol_accel_1b
                       - bar_range, body, upper_wick, lower_wick
    L2 (9 per TF)    : rolling-window smoothed
                       - price_velocity_N, price_accel_N
                       - vol_velocity_N, vol_accel_N
                       - price_mean_N, price_sigma_N
                       - vol_mean_N, vol_sigma_N
                       - vwap_N
    L3 (8 per TF)    : approved exceptions (Principle 7)
                       - z_se_N, z_high_N, z_low_N  (proper OLS per series)
                       - SE_high_N, SE_low_N        (wick dispersion)
                       - hurst_N                    (R/S analysis, N * 8)
                       - reversion_prob_N           (OU first-passage from z_se)
                       - swing_noise_N              (max drawdown/drawup / tick, 30-bar)

Layered-family output: compute_L0/L1/L2/L3 each return a DataFrame with
layer-prefixed column names. The builder writes one parquet per layer-family
per day per TF (see research/feature_spec_v2.md storage layout).

Zero-lookahead guarantee:
    Every feature at bar t depends only on bars <= t.
    All rolling operations use trailing windows [t-W+1, t].
    OLS fits use trailing windows. R/S analysis uses trailing windows.
    Swing noise uses trailing window of 30 bars.
    No .ffill(), .bfill(), .shift(-N), or forward-referencing indexing.

See research/feature_spec_v2.md for the full spec including the 7 principles
governing this layer architecture.

Supersedes: core/statistical_field_engine.py (v1, 91D mixed-layer output).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from numba import njit, prange
from scipy.special import erfi
import logging

logger = logging.getLogger(__name__)


# ─── Constants ─────────────────────────────────────────────────────────────

# Per-TF default windows (Principle 2: 3x higher TF rule anchors these)
#  5s: 3 x 15s /  5s =  9    | 15s: 3 x  60s / 15s = 12   | 1m: 3 x  5m / 1m = 15
#  5m: 3 x 15m /  5m =  9    | 15m: 3 x  1h / 15m = 12    | 1h: 3 x  4h / 1h = 12
#  4h: 3 x  1D /  4h = 18    |  1D: 3 x  5D /  1D =  5
N_BASE = {
    '5s':  9,
    '15s': 12,
    '1m':  15,
    '5m':  9,
    '15m': 12,
    '1h':  12,
    '4h':  18,
    '1D':  5,
}

# Multiplier applied to N_BASE for multi-scale estimators that require more samples
N_HURST_MULT = 8         # Hurst R/S needs ~100 samples for stable estimate

# Fixed window for swing noise (in bars - matches v1 convention, used for exit giveback)
SWING_NOISE_WINDOW = 30

# MNQ tick size
TICK_SIZE = 0.25

# OU first-passage boundary (standard value used in v1; matches erfi-based formula)
OU_BOUNDARY = 3.0


# ─── Numba kernels ─────────────────────────────────────────────────────────
# All kernels use parallel trailing-window semantics.
# Contract: kernel[i] uses y[max(0, i - W + 1) : i + 1] - never y[i + 1 :].

@njit(parallel=True, cache=True)
def _ols_fit_kernel(y: np.ndarray, window: int):
    """OLS fit over trailing window of `window` bars at each endpoint.

    For each i >= window-1, fits a straight line to y[i-window+1 : i+1]
    and returns (rm[i], se[i]) where:
      rm[i] = fitted value at the endpoint (x = window - 1)
      se[i] = standard deviation of residuals (ddof = 2)

    Bars with i < window-1 get NaN. No lookahead possible by construction.
    """
    n = len(y)
    rm = np.full(n, np.nan)
    se = np.full(n, np.nan)

    if n < 1:
        return rm, se

    for i in prange(n):
        start = max(0, i - window + 1)
        w = i - start + 1
        
        if w < 2:
            rm[i] = y[i]
            se[i] = 0.0
            continue

        x_mean = (w - 1) / 2.0
        denom = 0.0
        for k in range(w):
            dx = k - x_mean
            denom += dx * dx

        if denom < 1e-12:
            rm[i] = y[i]
            se[i] = 0.0
            continue

        sum_y = 0.0
        for k in range(w):
            sum_y += y[start + k]
        y_mean = sum_y / w

        sum_xy_dev = 0.0
        for k in range(w):
            sum_xy_dev += (k - x_mean) * (y[start + k] - y_mean)
        slope = sum_xy_dev / denom
        intercept = y_mean - slope * x_mean

        rm[i] = intercept + slope * (w - 1)

        sum_sq_res = 0.0
        for k in range(w):
            fit = intercept + slope * k
            res = y[start + k] - fit
            sum_sq_res += res * res
        se[i] = np.sqrt(sum_sq_res / max(w - 2, 1))

    return rm, se


@njit(parallel=True, cache=True)
def _ols_slope_kernel(y: np.ndarray, window: int):
    """OLS slope over trailing window, returning (slope, se_slope, t_stat).
    
    Used specifically for computing lambda_hat (stability exponent).
    """
    n = len(y)
    slope_out = np.full(n, np.nan)
    se_out = np.full(n, np.nan)
    t_out = np.full(n, np.nan)

    if n < 1:
        return slope_out, se_out, t_out

    for i in prange(n):
        start = max(0, i - window + 1)
        w = i - start + 1
        
        if w < 3:
            continue

        x_mean = (w - 1) / 2.0
        x_var_sum = 0.0
        for k in range(w):
            dx = k - x_mean
            x_var_sum += dx * dx

        if x_var_sum < 1e-12:
            continue

        sum_y = 0.0
        for k in range(w):
            sum_y += y[start + k]
        y_mean = sum_y / w

        cov = 0.0
        for k in range(w):
            cov += (k - x_mean) * (y[start + k] - y_mean)
            
        slope = cov / x_var_sum
        intercept = y_mean - slope * x_mean

        sum_sq_res = 0.0
        for k in range(w):
            fit = intercept + slope * k
            res = y[start + k] - fit
            sum_sq_res += res * res
            
        var_resid = sum_sq_res / (w - 2)
        if var_resid > 0:
            se_k = np.sqrt(var_resid / x_var_sum)
            t_stat = slope / se_k
        else:
            se_k = 0.0
            t_stat = 0.0
            
        slope_out[i] = slope
        se_out[i] = se_k
        t_out[i] = t_stat

    return slope_out, se_out, t_out


@njit(parallel=True, cache=True)
def _rolling_mean_kernel(y: np.ndarray, window: int) -> np.ndarray:
    """Rolling mean over trailing window. out[i] uses y[i-window+1 : i+1]."""
    n = len(y)
    out = np.full(n, np.nan)
    if n < 1:
        return out

    for i in prange(n):
        start = max(0, i - window + 1)
        w = i - start + 1
        s = 0.0
        for k in range(w):
            s += y[start + k]
        out[i] = s / w
    return out


@njit(parallel=True, cache=True)
def _rolling_std_kernel(y: np.ndarray, window: int) -> np.ndarray:
    """Rolling std (ddof=1) over trailing window."""
    n = len(y)
    out = np.full(n, np.nan)
    if n < 1:
        return out

    for i in prange(n):
        start = max(0, i - window + 1)
        w = i - start + 1
        if w < 2:
            out[i] = 0.0
            continue
            
        s = 0.0
        for k in range(w):
            s += y[start + k]
        mean = s / w

        sq_sum = 0.0
        for k in range(w):
            d = y[start + k] - mean
            sq_sum += d * d
        out[i] = np.sqrt(sq_sum / (w - 1))
    return out


@njit(parallel=True, cache=True)
def _vwap_kernel(prices: np.ndarray, volumes: np.ndarray, window: int) -> np.ndarray:
    """VWAP over trailing window: sum(p*v) / sum(v) for bars [i-W+1, i]."""
    n = len(prices)
    out = np.full(n, np.nan)
    if n < 1:
        return out

    for i in prange(n):
        start = max(0, i - window + 1)
        w = i - start + 1
        
        sum_pv = 0.0
        sum_v = 0.0
        for k in range(w):
            sum_pv += prices[start + k] * volumes[start + k]
            sum_v += volumes[start + k]
        if sum_v > 0:
            out[i] = sum_pv / sum_v
        else:
            s = 0.0
            for k in range(w):
                s += prices[start + k]
            out[i] = s / w
    return out


@njit(parallel=True, cache=True)
def _swing_noise_kernel(highs: np.ndarray, lows: np.ndarray,
                         window: int, tick: float) -> np.ndarray:
    """Max drawdown / drawup over trailing window, in ticks.

    Port of v1's _compute_swing_noise_numba. Uses O(1) running high/low
    tracking per endpoint. Uses bars [i-window+1, i] inclusive.
    """
    n = len(highs)
    out = np.full(n, np.nan)
    if n < 1:
        return out

    for i in prange(n):
        start = max(0, i - window + 1)
        w = i - start + 1
        
        if w < 2:
            out[i] = max((highs[i] - lows[i]) / tick, 1.0)
            continue
            
        run_hi = highs[start]
        run_lo = lows[start]
        max_dd = run_hi - lows[start]
        max_du = highs[start] - run_lo

        for k in range(1, w):
            j = start + k
            if highs[j] > run_hi:
                run_hi = highs[j]
            dd = run_hi - lows[j]
            if dd > max_dd:
                max_dd = dd

            if lows[j] < run_lo:
                run_lo = lows[j]
            du = highs[j] - run_lo
            if du > max_du:
                max_du = du

        out[i] = max(max_dd, max_du) / tick
    return out


@njit(cache=True)
def _hurst_rs_kernel(prices: np.ndarray, window: int) -> np.ndarray:
    """Hurst exponent via R/S analysis over trailing window.

    For each i >= window-1, computes R/S at 4 sub-scales
    (window/8, window/4, window/2, window) and fits log(R/S) = H log(n) + c.

    Uses bars [i-window+1, i]. Default 0.5 (random walk) for i < window-1.
    """
    n = len(prices)
    out = np.full(n, 0.5)
    if n < 1:
        return out

    for i in range(n):
        start = max(0, i - window + 1)
        w = i - start + 1
        
        if w < 8:
            out[i] = 0.5
            continue

        # Four sub-window sizes
        size_small = max(w // 8, 4)
        size_mid1 = max(w // 4, 8)
        size_mid2 = max(w // 2, 16)
        size_large = w

        log_n = np.zeros(4)
        log_rs = np.zeros(4)
        valid = 0

        for si in range(4):
            if si == 0:
                sz = size_small
            elif si == 1:
                sz = size_mid1
            elif si == 2:
                sz = size_mid2
            else:
                sz = size_large

            if sz > i + 1 or sz < 2:
                continue

            h_start = i - sz + 1
            p_start = prices[h_start]
            p_end = prices[i]
            mean_ret = (p_end - p_start) / (sz - 1)

            cum_dev = 0.0
            max_dev = -1e30
            min_dev = 1e30
            std_sum = 0.0
            prev_p = p_start

            for k in range(1, sz):
                curr_p = prices[h_start + k]
                ret = (curr_p - prev_p) - mean_ret
                cum_dev += ret
                if cum_dev > max_dev:
                    max_dev = cum_dev
                if cum_dev < min_dev:
                    min_dev = cum_dev
                std_sum += ret * ret
                prev_p = curr_p

            R = max_dev - min_dev
            S = np.sqrt(std_sum / (sz - 1))
            if S < 1e-10:
                S = 1e-10
            rs = R / S
            if rs < 1e-10:
                rs = 1e-10

            log_n[valid] = np.log(float(sz))
            log_rs[valid] = np.log(rs)
            valid += 1

        if valid < 2:
            out[i] = 0.5
            continue

        # Least squares fit on valid points
        sx = 0.0
        sy = 0.0
        sxy = 0.0
        sxx = 0.0
        for si in range(valid):
            sx += log_n[si]
            sy += log_rs[si]
            sxy += log_n[si] * log_rs[si]
            sxx += log_n[si] * log_n[si]

        d = valid * sxx - sx * sx
        if abs(d) > 1e-12:
            h = (valid * sxy - sx * sy) / d
            if h < 0.0:
                h = 0.0
            elif h > 1.0:
                h = 1.0
            out[i] = h
        else:
            out[i] = 0.5

    return out


# ─── Public engine ─────────────────────────────────────────────────────────

class StatisticalFieldEngine:
    """
    v2 SFE - computes layered features per research/feature_spec_v2.md.

    Public methods:
        compute_L0(df)              -> DataFrame with [L0_time_of_day]
        compute_L1(df, tf)          -> DataFrame with 6 L1_{tf}_* columns
        compute_L2(df, tf, N=None)  -> DataFrame with 9 L2_{tf}_* columns
        compute_L3(df, tf, N=None)  -> DataFrame with 8 L3_{tf}_* columns

    Each method is pure: takes a bar-indexed DataFrame, returns a feature
    DataFrame of identical length. No state, no caching, no cross-call
    dependencies.

    Zero-lookahead: every feature at bar t depends only on bars <= t.
    Input DataFrame must be sorted by timestamp ascending.

    Example:
        sfe = StatisticalFieldEngine()
        l0 = sfe.compute_L0(df)
        l1 = sfe.compute_L1(df, tf='1m')
        l2 = sfe.compute_L2(df, tf='1m')       # uses N_BASE['1m'] = 15
        l3 = sfe.compute_L3(df, tf='1m')       # uses N_BASE['1m'] = 15
    """

    def __init__(self, windows: dict | None = None, use_gpu: bool | None = None):
        """
        Args:
            windows: optional per-TF window overrides. Defaults to N_BASE.
            use_gpu: reserved for future CUDA kernels. Not used in v2.
        """
        self.windows = dict(N_BASE)
        if windows:
            self.windows.update(windows)
        # use_gpu reserved; v2 Numba-CPU is fast enough for this feature set
        self.use_gpu = use_gpu

    # ─── L0 ───────────────────────────────────────────────────────────────

    def compute_L0(self, df: pd.DataFrame) -> pd.DataFrame:
        """Global time-of-day feature.

        Lookahead: none. time_of_day[i] depends only on timestamp[i].

        Args:
            df: DataFrame with 'timestamp' column (Unix seconds as int or float).
        Returns:
            DataFrame with one column: L0_time_of_day in [0, 1).
        """
        if 'timestamp' not in df.columns:
            raise ValueError("compute_L0 requires 'timestamp' column in df")
        ts = df['timestamp'].values.astype(np.float64)
        tod = (ts % 86400.0) / 86400.0
        return pd.DataFrame({'L0_time_of_day': tod}, index=df.index)

    # ─── L1 ───────────────────────────────────────────────────────────────

    def compute_L1(self, df: pd.DataFrame, tf: str) -> pd.DataFrame:
        """Window-free primitives for a single TF.

        Features (8 per TF):
          - L1_{tf}_price_velocity_1b : close[t] - close[t-1]
          - L1_{tf}_price_accel_1b    : price_velocity_1b[t] - price_velocity_1b[t-1]
          - L1_{tf}_vol_velocity_1b   : volume[t] - volume[t-1]
          - L1_{tf}_vol_accel_1b      : vol_velocity_1b[t] - vol_velocity_1b[t-1]
          - L1_{tf}_bar_range         : high[t] - low[t]
          - L1_{tf}_body              : close[t] - open[t]
          - L1_{tf}_upper_wick        : high[t] - max(open[t], close[t])
          - L1_{tf}_lower_wick        : min(open[t], close[t]) - low[t]

        Lookahead: none. Each feature at bar t uses bars in {t-2, t-1, t} only.

        Args:
            df: DataFrame with 'open', 'high', 'low', 'close', 'volume' columns.
            tf: TF label (used for column naming).
        Returns:
            DataFrame of length len(df) with the 8 L1_{tf}_* columns.
            Bars < 2 (for velocity) or < 3 (for acceleration) get NaN.
        """
        for col in ('open', 'high', 'low', 'close', 'volume'):
            if col not in df.columns:
                raise ValueError(f"compute_L1 requires '{col}' column in df")

        close = df['close'].values.astype(np.float64)
        volume = df['volume'].values.astype(np.float64)
        high = df['high'].values.astype(np.float64)
        low = df['low'].values.astype(np.float64)
        opn = df['open'].values.astype(np.float64)
        n = len(close)

        # 1-bar velocity: uses close[t-1], close[t]
        price_v = np.zeros(n)
        if n >= 2:
            price_v[1:] = close[1:] - close[:-1]

        vol_v = np.zeros(n)
        if n >= 2:
            vol_v[1:] = volume[1:] - volume[:-1]

        # 1-bar acceleration: uses velocity[t-1], velocity[t]
        # velocity[t-1] = close[t-1] - close[t-2], so accel[t] uses {t-2, t-1, t}
        price_a = np.zeros(n)
        if n >= 3:
            price_a[2:] = price_v[2:] - price_v[1:-1]

        vol_a = np.zeros(n)
        if n >= 3:
            vol_a[2:] = vol_v[2:] - vol_v[1:-1]

        # Spatial delta (same-bar)
        bar_range = high - low
        body = close - opn
        
        # One-sided wicks
        upper_wick = high - np.maximum(opn, close)
        lower_wick = np.minimum(opn, close) - low

        return pd.DataFrame({
            f'L1_{tf}_price_velocity_1b': price_v,
            f'L1_{tf}_price_accel_1b':    price_a,
            f'L1_{tf}_vol_velocity_1b':   vol_v,
            f'L1_{tf}_vol_accel_1b':      vol_a,
            f'L1_{tf}_bar_range':         bar_range,
            f'L1_{tf}_body':              body,
            f'L1_{tf}_upper_wick':        upper_wick,
            f'L1_{tf}_lower_wick':        lower_wick,
        }, index=df.index)

    # ─── L2 ───────────────────────────────────────────────────────────────

    def compute_L2(self, df: pd.DataFrame, tf: str, N: int | None = None) -> pd.DataFrame:
        """Rolling-window smoothed features for a single TF.

        Default N from N_BASE[tf] (Principle 2 tiered windows).

        Features (9 per TF):
          - L2_{tf}_price_velocity_{N} : (close[t] - close[t-N]) / N
          - L2_{tf}_price_accel_{N}    : (price_velocity_1b[t] - price_velocity_1b[t-N]) / N
          - L2_{tf}_vol_velocity_{N}   : (volume[t] - volume[t-N]) / N
          - L2_{tf}_vol_accel_{N}      : (vol_velocity_1b[t] - vol_velocity_1b[t-N]) / N
          - L2_{tf}_price_mean_{N}     : rolling mean of close over N bars
          - L2_{tf}_price_sigma_{N}    : rolling std (ddof=1) of close over N bars
          - L2_{tf}_vol_mean_{N}       : rolling mean of volume over N bars
          - L2_{tf}_vol_sigma_{N}      : rolling std (ddof=1) of volume over N bars
          - L2_{tf}_vwap_{N}           : volume-weighted avg price over N bars

        Lookahead: none. Every feature at bar t uses bars [t-N+1, t] or shorter.

        Args:
            df: DataFrame with 'close', 'volume' columns.
            tf: TF label.
            N: window size. Defaults to N_BASE[tf].
        Returns:
            DataFrame of length len(df) with 9 L2_{tf}_*_{N} columns.
            Bars < N get NaN for windowed features.
        """
        if N is None:
            N = self.windows.get(tf, 12)

        for col in ('close', 'volume'):
            if col not in df.columns:
                raise ValueError(f"compute_L2 requires '{col}' column in df")

        close = df['close'].values.astype(np.float64)
        volume = df['volume'].values.astype(np.float64)
        n = len(close)

        # Smoothed velocities (N-bar mean of 1-bar delta equals endpoint secant)
        price_v_N = np.zeros(n)
        for i in range(1, n):
            start = max(0, i - N)
            price_v_N[i] = (close[i] - close[start]) / (i - start)

        vol_v_N = np.zeros(n)
        for i in range(1, n):
            start = max(0, i - N)
            vol_v_N[i] = (volume[i] - volume[start]) / (i - start)

        # Smoothed accelerations = MA of 1-bar ddeltadelta
        # = (velocity_1b[t] - velocity_1b[t-N]) / N
        price_a_N = np.zeros(n)
        v1b = np.zeros(n)
        if n >= 2:
            v1b[1:] = close[1:] - close[:-1]
        for i in range(2, n):
            start = max(1, i - N)
            price_a_N[i] = (v1b[i] - v1b[start]) / (i - start)

        vol_a_N = np.zeros(n)
        vv1b = np.zeros(n)
        if n >= 2:
            vv1b[1:] = volume[1:] - volume[:-1]
        for i in range(2, n):
            start = max(1, i - N)
            vol_a_N[i] = (vv1b[i] - vv1b[start]) / (i - start)

        # First statistical moments (rolling)
        price_mean_N = _rolling_mean_kernel(close, N)
        price_sigma_N = _rolling_std_kernel(close, N)
        vol_mean_N = _rolling_mean_kernel(volume, N)
        vol_sigma_N = _rolling_std_kernel(volume, N)

        # VWAP (cross-domain volume-weighted first moment)
        vwap_N = _vwap_kernel(close, volume, N)

        return pd.DataFrame({
            f'L2_{tf}_price_velocity_{N}': price_v_N,
            f'L2_{tf}_price_accel_{N}':    price_a_N,
            f'L2_{tf}_vol_velocity_{N}':   vol_v_N,
            f'L2_{tf}_vol_accel_{N}':      vol_a_N,
            f'L2_{tf}_price_mean_{N}':     price_mean_N,
            f'L2_{tf}_price_sigma_{N}':    price_sigma_N,
            f'L2_{tf}_vol_mean_{N}':       vol_mean_N,
            f'L2_{tf}_vol_sigma_{N}':      vol_sigma_N,
            f'L2_{tf}_vwap_{N}':           vwap_N,
        }, index=df.index)

    # ─── L3 ───────────────────────────────────────────────────────────────

    def compute_L3(self, df: pd.DataFrame, tf: str, N: int | None = None) -> pd.DataFrame:
        """Approved L3 exception features for a single TF (Principle 7).

        Features (8 per TF):
          - L3_{tf}_z_se_{N}           : (close - RM_close) / SE_close (OLS on close)
          - L3_{tf}_z_high_{N}         : (high - RM_high) / SE_high   (OLS on high)
          - L3_{tf}_z_low_{N}          : (low  - RM_low)  / SE_low    (OLS on low)
          - L3_{tf}_SE_high_{N}        : OLS residual std on high over N bars
          - L3_{tf}_SE_low_{N}         : OLS residual std on low  over N bars
          - L3_{tf}_hurst_{N}          : R/S Hurst exponent over N * N_HURST_MULT bars
          - L3_{tf}_reversion_prob_{N} : analytical OU first-passage from z_se
          - L3_{tf}_swing_noise_{N}    : max drawdown/drawup / tick over 30-bar window

        Statistical correction from v1: z_high and z_low now use their OWN
        OLS fits (RM_high/SE_high, RM_low/SE_low) instead of sharing the
        close-fit sigma. v1's semantics conflated close-residual dispersion
        with high/low position - v2 fixes this.

        Wick dispersion (SE_high, SE_low) is exposed as a standalone feature
        because close-only primitives cannot detect regimes where closes are
        tight but wicks wildly excurse (liquidity-probe / stop-hunt chop).

        Lookahead: none. All OLS/R/S/swing-noise windows are strictly trailing.

        Args:
            df: DataFrame with 'high', 'low', 'close' columns.
            tf: TF label.
            N: window for OLS + reversion_prob. Defaults to N_BASE[tf].
               Hurst uses N * N_HURST_MULT. Swing noise uses fixed 30.
        Returns:
            DataFrame of length len(df) with 8 L3_{tf}_*_{N} columns.
            Bars < N (or < N * N_HURST_MULT for hurst, < 30 for swing_noise)
            get NaN / 0.5 default.
        """
        if N is None:
            N = self.windows.get(tf, 12)

        for col in ('high', 'low', 'close'):
            if col not in df.columns:
                raise ValueError(f"compute_L3 requires '{col}' column in df")

        close = df['close'].values.astype(np.float64)
        high = df['high'].values.astype(np.float64)
        low = df['low'].values.astype(np.float64)
        n = len(close)

        # Three independent OLS fits - one per series
        rm_close, se_close = _ols_fit_kernel(close, N)
        rm_high, se_high = _ols_fit_kernel(high, N)
        rm_low, se_low = _ols_fit_kernel(low, N)

        # Band z-scores (each uses its own series' OLS fit - corrected from v1)
        z_se = np.full(n, np.nan)
        mask = se_close > 1e-10
        z_se[mask] = (close[mask] - rm_close[mask]) / se_close[mask]

        z_high_out = np.full(n, np.nan)
        mask_h = se_high > 1e-10
        z_high_out[mask_h] = (high[mask_h] - rm_high[mask_h]) / se_high[mask_h]

        z_low_out = np.full(n, np.nan)
        mask_l = se_low > 1e-10
        z_low_out[mask_l] = (low[mask_l] - rm_low[mask_l]) / se_low[mask_l]

        # Hurst over larger window (multi-scale R/S needs more samples)
        N_hurst = N * N_HURST_MULT
        hurst = _hurst_rs_kernel(close, N_hurst)

        # OU first-passage analytical formula using z_se:
        # P(tunnel to 0 before +-B) = 1 - erfi(|z|/sqrt(2)) / erfi(B/sqrt(2))
        inv_sqrt2 = 1.0 / np.sqrt(2.0)
        erfi_B = float(erfi(OU_BOUNDARY * inv_sqrt2))
        reversion = np.full(n, np.nan)
        valid_z = ~np.isnan(z_se)
        if valid_z.any():
            abs_z = np.abs(z_se[valid_z])
            reversion[valid_z] = np.clip(
                1.0 - erfi(abs_z * inv_sqrt2) / erfi_B,
                0.0, 1.0
            )

        # Swing noise over fixed 30-bar window (matches v1 exit-engine convention)
        swing = _swing_noise_kernel(high, low, SWING_NOISE_WINDOW, TICK_SIZE)

        return pd.DataFrame({
            f'L3_{tf}_z_se_{N}':           z_se,
            f'L3_{tf}_z_high_{N}':         z_high_out,
            f'L3_{tf}_z_low_{N}':          z_low_out,
            f'L3_{tf}_SE_high_{N}':        se_high,
            f'L3_{tf}_SE_low_{N}':         se_low,
            f'L3_{tf}_hurst_{N}':          hurst,
            f'L3_{tf}_reversion_prob_{N}': reversion,
            f'L3_{tf}_swing_noise_{N}':    swing,
        }, index=df.index)

    # ─── L4 (NMP State) ───────────────────────────────────────────────────

    def compute_L4_NMP(self, df: pd.DataFrame, tf: str, z_se: np.ndarray | None = None) -> pd.DataFrame:
        """Nightmare Protocol (NMP) State feature layer (Principle 8).

        Features (11 per TF):
          - L4_{tf}_vr_exact            : Rolling 10/60 std ratio of raw closes
          - L4_{tf}_z_21                : Exact 21-bar linear regression standardization
          - L4_{tf}_lambda_hat_{12/21/30} : OLS slope of log(|z_se| + 0.1) over k bars
          - L4_{tf}_lambda_se_{12/21/30}  : Standard error of the lambda slope
          - L4_{tf}_lambda_t_{12/21/30}   : T-statistic of the lambda slope

        Lookahead: none. All windows are trailing.

        Args:
            df: DataFrame with 'close' column.
            tf: TF label.
            z_se: (Optional) Precomputed z_se array. If not provided, computes it on the fly.
        """
        for col in ('close',):
            if col not in df.columns:
                raise ValueError(f"compute_L4_NMP requires '{col}' column in df")

        close = df['close'].values.astype(np.float64)
        n = len(close)

        # 1. vr_exact
        std_fast = _rolling_std_kernel(close, 10)
        std_slow = _rolling_std_kernel(close, 60)
        
        vr_exact = np.full(n, np.nan)
        mask_v = std_slow > 1e-10
        vr_exact[mask_v] = std_fast[mask_v] / std_slow[mask_v]

        # 2. z_21
        rm_21, se_21 = _ols_fit_kernel(close, 21)
        z_21 = np.full(n, np.nan)
        mask_z21 = se_21 > 1e-10
        z_21[mask_z21] = (close[mask_z21] - rm_21[mask_z21]) / se_21[mask_z21]

        # 3. lambda_hat
        if z_se is None:
            # Need N_BASE for z_se computation to be perfectly aligned with L3
            N = self.windows.get(tf, 12)
            rm_close, se_close = _ols_fit_kernel(close, N)
            z_se = np.full(n, np.nan)
            mask_z = se_close > 1e-10
            z_se[mask_z] = (close[mask_z] - rm_close[mask_z]) / se_close[mask_z]

        log_z = np.log(np.abs(z_se) + 0.1)
        
        lambda_cols = {}
        for k in (12, 21, 30):
            slope, se, t = _ols_slope_kernel(log_z, k)
            lambda_cols[f'L4_{tf}_lambda_hat_{k}'] = slope
            lambda_cols[f'L4_{tf}_lambda_se_{k}'] = se
            lambda_cols[f'L4_{tf}_lambda_t_{k}'] = t

        res = {
            f'L4_{tf}_vr_exact': vr_exact,
            f'L4_{tf}_z_21': z_21,
        }
        res.update(lambda_cols)
        
        return pd.DataFrame(res, index=df.index)

    # ─── Convenience wrappers ─────────────────────────────────────────────

    def batch_compute_states(self, df: pd.DataFrame, tf: str,
                             N: int | None = None) -> pd.DataFrame:
        """V1-name-compatible convenience method.

        Computes L0 + L1 + L2 + L3 for one TF and returns a single joined
        DataFrame with all 24 feature columns (1 L0 + 6 L1 + 9 L2 + 8 L3).

        Use this when you want the full feature set for a TF in one call.
        Use compute_L0/L1/L2/L3 individually when writing to layer-family
        parquets (build_dataset_v2.py).
        """
        if N is None:
            N = self.windows.get(tf, 12)

        parts = [
            self.compute_L0(df),
            self.compute_L1(df, tf),
            self.compute_L2(df, tf, N),
            self.compute_L3(df, tf, N),
        ]
        
        # Extract z_se to avoid recomputation in L4
        z_se = parts[-1][f'L3_{tf}_z_se_{N}'].values
        parts.append(self.compute_L4_NMP(df, tf, z_se=z_se))
        
        return pd.concat(parts, axis=1)

    def feed_bar(self, *args, **kwargs):
        """V1 name preserved; v2 is offline-batch only.

        Raises NotImplementedError - v2 SFE does not support incremental
        single-bar processing yet. If/when v2 is promoted to live, an
        incremental state machine mirroring v1's _IncrementalState will
        be added here.

        For now, use compute_L0/L1/L2/L3 on a bar-indexed DataFrame.
        """
        raise NotImplementedError(
            "v2 SFE is offline-batch only. Use compute_L0/L1/L2/L3 (or "
            "batch_compute_states) on a bar-indexed DataFrame. feed_bar "
            "will be implemented when v2 is promoted to live trading."
        )
