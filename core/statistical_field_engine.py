"""
Statistical Field Engine
Computes regression bands, z-scores, probability distributions,
and mean-reversion/breakout statistics from price data.
GPU-accelerated via Numba CUDA kernels.

================================================================
WHAT THIS ENGINE COMPUTES (per TF, per bar)
================================================================

Input (per bar):    price (OHLC), volume, timestamp
Rolling window:     SFE_WINDOW = 300 bars (max buffer fed by caller)
Inner windows:      each feature below has its own sub-window

| # | Feature         | Layer | Computation                                  | Window |
|---|-----------------|-------|----------------------------------------------|--------|
| 0 | z_se            | L5    | (close - RM_N) / SE_N, OLS regression        | N      |
| 1 | dmi_diff        | L3    | DI+ minus DI-                                | 14     |
| 2 | variance_ratio  | L6    | σ²_short / σ²_long (Lo-MacKinlay)            | short, long |
| 3 | velocity        | L4*   | WINDOWED slope (NOT 1-bar rate)              | N      |
| 4 | acceleration    | L4*   | WINDOWED change in slope                     | N      |
| 5 | vol_rel         | L8    | volume / 30-bar SMA                          | 30     |
| 6 | bar_range       | L3    | (high - low) / tick                          | none   |
| 7 | hurst           | L7    | Hurst exponent (R/S or DFA)                  | HURST_WINDOW |
| 8 | reversion_prob  | L5    | OU first-passage P(reach center within τ)    | N      |
| 9 | p_at_center     | L5    | 3-class probability near regression mean     | N      |
|10 | z_high          | L5    | (high - RM_N) / SE_N                         | N      |
|11 | z_low           | L5    | (low  - RM_N) / SE_N                         | N      |

Helpers (derived from above):
| 0 | dmi_gap         | L3    | abs(dmi_diff)                                | —      |
| 1 | dir_vol         | L8    | sign(velocity) * vol_rel                     | —      |
| 2 | wick_ratio      | L3    | 1 - abs(close-open)/range                    | none   |

Notes:
 * "velocity" and "acceleration" (rows 3, 4) are WINDOWED SLOPES, not true
   1-bar Δ. They're effectively L4 (smoothed kinematics) mislabeled as L2.
   See research/feature_spec_v2.md for the corrected layering.
 * The 300-bar SFE_WINDOW is the OUTER buffer fed to this engine. Each
   feature has its own INNER window (N, 14, 30, HURST_WINDOW, etc.).
 * z_high / z_low use the SAME RM_N + SE_N fit as z_se — they're the
   bar's extreme extents in σ-space.
================================================================
"""
import numpy as np
import pandas as pd
from numba import cuda, njit, prange
import logging
from scipy.special import erfi

from core.market_state import MarketState
from core.pattern_utils import (
    detect_geometric_patterns_vectorized, detect_candlestick_patterns_vectorized
)

from core.physics_utils import compute_adx_dmi_cpu, ADX_PERIOD, HURST_WINDOW

# Core CUDA Physics
try:
    from core.cuda_statistics import (
        compute_regression_kernel, detect_pattern_flags_kernel,
        compute_dm_tr_kernel, compute_hurst_kernel
    )
    CUDA_PHYSICS_AVAILABLE = True
except ImportError:
    CUDA_PHYSICS_AVAILABLE = False

# Optional: CUDA Pattern Detector
try:
    from core.cuda_pattern_detector import detect_patterns_cuda, NUMBA_AVAILABLE as CUDA_PATTERNS_AVAILABLE
except ImportError:
    CUDA_PATTERNS_AVAILABLE = False


# PID Control Constants (Default/Fallback)
DEFAULT_PID_KP = 0.5
DEFAULT_PID_KI = 0.1
DEFAULT_PID_KD = 0.2
DEFAULT_PID_INTEGRAL_WINDOW = 30   # Rolling window for integral term (bars)
DEFAULT_REVERSION_THETA = 0.5


logger = logging.getLogger(__name__)


# ─── Numba JIT kernels ─────────────────────────────────────────────────────
# Ported from bolt/numba-rolling-aggregations PR #312.
# These replace Python-level loops with parallel Numba kernels.

@njit(parallel=True, cache=True)
def _compute_rolling_std_numba(z_scores, n, window):
    """Rolling standard deviation (ddof=1) via parallel Numba kernel.

    Replaces sliding_window_view + .std(axis=1) which allocates a full
    (n, window) view array.  This version is O(1) extra space per thread.
    """
    osc_std = np.full(n, np.nan)
    if n >= window:
        for i in prange(window - 1, n):
            sum_val = 0.0
            for j in range(i - window + 1, i + 1):
                sum_val += z_scores[j]
            mean_val = sum_val / window

            sum_sq_diff = 0.0
            for j in range(i - window + 1, i + 1):
                diff = z_scores[j] - mean_val
                sum_sq_diff += diff * diff

            osc_std[i] = np.sqrt(sum_sq_diff / (window - 1))

        for i in range(window - 1):
            osc_std[i] = osc_std[window - 1]

    return osc_std


@njit(parallel=True, cache=True)
def _compute_swing_noise_numba(highs, lows, n, noise_window, tick_size):
    """Max drawdown/drawup over a rolling window via parallel Numba kernel.

    Replaces a Python loop with np.maximum.accumulate per iteration.
    Uses O(1) running-hi/running-lo tracking per thread instead.
    """
    swing_noise = np.full(n, 35.0)
    for _ni in prange(noise_window, n):
        start_idx = _ni - noise_window
        run_hi = highs[start_idx]
        run_lo = lows[start_idx]
        max_dd = run_hi - lows[start_idx]
        max_du = highs[start_idx] - run_lo

        for j in range(start_idx + 1, _ni + 1):
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

        swing_noise[_ni] = max(max_dd, max_du) / tick_size
    return swing_noise


class _IncrementalState:
    """Internal state for feed_bar() incremental computation."""
    __slots__ = [
        'bar_count', 'prices', 'highs', 'lows', 'closes', 'z_scores',
        'prev_close', 'prev_high', 'prev_low',
        # ADX/DMI Wilder smoothing
        'adx_init_tr', 'adx_init_plus', 'adx_init_minus',
        'smooth_tr', 'smooth_plus', 'smooth_minus',
        'dx_sum', 'dx_count', 'prev_adx', 'prev_dmi_plus', 'prev_dmi_minus',
        # Previous bar values for derivative/prev fields
        'prev_z',
    ]

    def __init__(self):
        self.bar_count = 0
        # Ring buffers as plain lists (trimmed to maxlen manually)
        self.prices = []    # maxlen 30 (hurst window)
        self.highs = []     # maxlen 30 (swing noise)
        self.lows = []      # maxlen 30
        self.closes = []    # maxlen 30 (DM/TR needs prev close)
        self.z_scores = []  # maxlen 30 (PID integral)
        self.prev_close = 0.0
        self.prev_high = 0.0
        self.prev_low = 0.0
        # ADX/DMI — accumulate raw values during warmup, then Wilder smooth
        self.adx_init_tr = []   # collect first ADX_PERIOD TR values (indices 1..period)
        self.adx_init_plus = []
        self.adx_init_minus = []
        self.smooth_tr = 0.0
        self.smooth_plus = 0.0
        self.smooth_minus = 0.0
        self.dx_sum = 0.0
        self.dx_count = 0
        self.prev_adx = 0.0
        self.prev_dmi_plus = 0.0
        self.prev_dmi_minus = 0.0
        self.prev_z = 0.0

    def _trim(self, buf, maxlen=30):
        if len(buf) > maxlen:
            del buf[:-maxlen]


class StatisticalFieldEngine:
    """
    Unified statistical field calculator  -- GPU-accelerated when CUDA available.
    Computes regression bands, z-scores, mean-reversion forces, probability
    distributions, and pattern flags from price data.
    """

    def __init__(self, regression_period: int = 21, use_gpu: bool = None):
        self.regression_period = regression_period
        self.SIGMA_EXTREME_MULTIPLIER = 2.0
        self.SIGMA_BREAKOUT_MULTIPLIER = 3.0

        # Statistical constants matching CUDA kernel
        self.REVERSION_THETA = 0.5
        self.BAND_PRESSURE_EPSILON = 0.01
        self.BAND_PRESSURE_CAP = 100.0
        self.VELOCITY_THRESHOLD = 0.5
        self.MOMENTUM_THRESHOLD = 5.0
        self.ENTROPY_THRESHOLD = 0.3

        # New Constant for entropy calculation
        self.LOG_3 = np.log(3.0)

        # Precompute regression constants
        # This removes duplication between CUDA setup and CPU fallback
        rp = self.regression_period
        sum_x = 0.0
        sum_xx = 0.0
        for _k in range(rp):
            sum_x += float(_k)
            sum_xx += float(_k * _k)

        self.mean_x = sum_x / rp
        self.denom = sum_xx - (sum_x * sum_x) / rp
        self.inv_reg_period = 1.0 / rp
        self.inv_denom = 0.0
        if abs(self.denom) > 1e-9:
            self.inv_denom = 1.0 / self.denom

        # Historical residuals for fat-tail sigma calculation (rolling window)
        # 30 bars — minimum for CLT, responsive to current regime
        self.residual_history = []
        self.residual_window = 30

        # === GPU SETUP ===
        if use_gpu is not None:
             self.use_gpu = use_gpu
             if self.use_gpu and not (cuda.is_available() and CUDA_PHYSICS_AVAILABLE):
                 logger.warning("GPU requested but CUDA/Kernels unavailable. Falling back to CPU.")
                 self.use_gpu = False
        else:
             # Auto-detect
             self.use_gpu = False
             if cuda.is_available() and CUDA_PHYSICS_AVAILABLE:
                 self.use_gpu = True
             else:
                 logger.warning("CUDA accelerator not available. Falling back to vectorized CPU execution.")

        # Incremental state (lazy-initialized on first feed_bar call)
        self._inc = _IncrementalState()

    # ═══════════════════════════════════════════════════════════════════════
    # INCREMENTAL API — feed one bar at a time, maintain state
    # ═══════════════════════════════════════════════════════════════════════

    def feed_bar(self, open_: float, high: float, low: float, close: float,
                 volume: float, timestamp: float = 0.0) -> 'MarketState | None':
        """Process one bar incrementally. Returns MarketState or None during warmup.

        Produces identical output to batch_compute_states(df)[-1] for the same
        input sequence. CPU-only — O(80) ops per bar.

        Args:
            open_, high, low, close, volume: bar OHLCV
            timestamp: bar timestamp (epoch seconds)

        Returns:
            MarketState if warmed up (bar_count >= 30), else None.
        """
        import math
        inc = self._inc
        rp = self.regression_period

        price = close  # SFE uses close as price
        inc.bar_count += 1

        # Append to ring buffers
        # prices: 30 (hurst window). highs/lows: 31 (swing noise uses noise_window+1).
        # z_scores: populated during warmup with 0.0 to match batch's z_scores array.
        inc.prices.append(price)
        inc.highs.append(high)
        inc.lows.append(low)
        inc.closes.append(close)
        inc._trim(inc.prices, 31)  # hurst needs i >= 30 (31st bar) + 30 lookback
        inc._trim(inc.highs, 31)
        inc._trim(inc.lows, 31)
        inc._trim(inc.closes, 31)

        # --- DM/TR raw (needs prev bar) ---
        if inc.bar_count == 1:
            tr = high - low
            plus_dm = 0.0
            minus_dm = 0.0
        else:
            hl = high - low
            hc = abs(high - inc.prev_close)
            lc = abs(low - inc.prev_close)
            tr = max(hl, hc, lc)

            up_move = high - inc.prev_high
            down_move = inc.prev_low - low

            plus_dm = up_move if (up_move > down_move and up_move > 0) else 0.0
            minus_dm = down_move if (down_move > up_move and down_move > 0) else 0.0

        # --- ADX/DMI Wilder smoothing (mirrors physics_utils.compute_adx_dmi_cpu) ---
        # Bar indexing: batch skips index 0 for initial sums (tr_raw[1:period+1])
        # So we accumulate bars 2..ADX_PERIOD+1 (bar_count 2 through ADX_PERIOD+1)
        adx_val = 0.0
        dmi_plus_val = 0.0
        dmi_minus_val = 0.0

        if inc.bar_count >= 2 and len(inc.adx_init_tr) < ADX_PERIOD:
            # Accumulation phase (bars 2 through ADX_PERIOD+1)
            inc.adx_init_tr.append(tr)
            inc.adx_init_plus.append(plus_dm)
            inc.adx_init_minus.append(minus_dm)

            if len(inc.adx_init_tr) == ADX_PERIOD:
                # First DI values
                inc.smooth_tr = sum(inc.adx_init_tr)
                inc.smooth_plus = sum(inc.adx_init_plus)
                inc.smooth_minus = sum(inc.adx_init_minus)

                if inc.smooth_tr > 0:
                    dmi_plus_val = 100.0 * inc.smooth_plus / inc.smooth_tr
                    dmi_minus_val = 100.0 * inc.smooth_minus / inc.smooth_tr

                # First DX
                di_sum = dmi_plus_val + dmi_minus_val
                dx = 100.0 * abs(dmi_plus_val - dmi_minus_val) / di_sum if di_sum > 0 else 0.0
                inc.dx_sum = dx
                inc.dx_count = 1

        elif len(inc.adx_init_tr) >= ADX_PERIOD:
            # Wilder smoothing phase
            inc.smooth_tr = inc.smooth_tr - (inc.smooth_tr / ADX_PERIOD) + tr
            inc.smooth_plus = inc.smooth_plus - (inc.smooth_plus / ADX_PERIOD) + plus_dm
            inc.smooth_minus = inc.smooth_minus - (inc.smooth_minus / ADX_PERIOD) + minus_dm

            if inc.smooth_tr > 0:
                dmi_plus_val = 100.0 * inc.smooth_plus / inc.smooth_tr
                dmi_minus_val = 100.0 * inc.smooth_minus / inc.smooth_tr

            di_sum = dmi_plus_val + dmi_minus_val
            dx = 100.0 * abs(dmi_plus_val - dmi_minus_val) / di_sum if di_sum > 0 else 0.0

            if inc.dx_count < ADX_PERIOD:
                inc.dx_sum += dx
                inc.dx_count += 1
                if inc.dx_count == ADX_PERIOD:
                    adx_val = inc.dx_sum / ADX_PERIOD
            else:
                adx_val = (inc.prev_adx * (ADX_PERIOD - 1) + dx) / ADX_PERIOD

        # Store prev values for next bar
        prev_adx_out = inc.prev_adx
        prev_dmi_plus_out = inc.prev_dmi_plus
        prev_dmi_minus_out = inc.prev_dmi_minus
        inc.prev_adx = adx_val
        inc.prev_dmi_plus = dmi_plus_val
        inc.prev_dmi_minus = dmi_minus_val
        inc.prev_close = close
        inc.prev_high = high
        inc.prev_low = low

        # --- Warmup check ---
        # Match batch behavior: return states from bar rp (21), not hurst window (30)
        # Batch returns states for bars >= rp-1 with hurst=0.5 default for bars < 30
        # Populate z_scores with 0.0 during warmup (batch has z=0 for bars < rp)
        # Buffer sized to SFE_WINDOW (300) so rolling window matches batch exactly
        if inc.bar_count < rp:
            inc.z_scores.append(0.0)
            inc._trim(inc.z_scores, 300)
            return None

        # --- Regression (21-bar OLS) ---
        # Port from compute_regression_kernel lines 61-109
        buf = inc.prices
        n_buf = len(buf)
        sum_y = 0.0
        sum_xy = 0.0
        sum_yy = 0.0
        for k in range(rp):
            val = buf[n_buf - rp + k]
            x = float(k)
            sum_y += val
            sum_xy += x * val
            sum_yy += val * val

        mean_y = sum_y * self.inv_reg_period
        slope = (sum_xy - self.mean_x * sum_y) * self.inv_denom
        center = mean_y + slope * ((rp - 1) - self.mean_x)

        sst = sum_yy - rp * mean_y * mean_y
        rss = sst - slope * slope * self.denom
        if rss < 0.0:
            rss = 0.0
        sigma = math.sqrt(rss / (rp - 2)) if rp > 2 else 0.0
        if sigma < 1e-6:
            sigma = 1e-6

        z = (price - center) / sigma

        # --- Velocity & Momentum ---
        velocity = price - buf[n_buf - 2] if n_buf >= 2 else 0.0
        momentum_val = (velocity * volume) / sigma

        # --- Forces (kernel lines 123-146) ---
        F_gravity = -self.REVERSION_THETA * (z * sigma)
        upper_sing = center + self.SIGMA_EXTREME_MULTIPLIER * sigma
        lower_sing = center - self.SIGMA_EXTREME_MULTIPLIER * sigma
        dist_upper = abs(price - upper_sing) / sigma
        dist_lower = abs(price - lower_sing) / sigma

        F_upper = 0.0
        if z > 0:
            F_upper = 1.0 / (dist_upper ** 3 + self.BAND_PRESSURE_EPSILON)
            if F_upper > self.BAND_PRESSURE_CAP:
                F_upper = self.BAND_PRESSURE_CAP
        F_lower = 0.0
        if z < 0:
            F_lower = 1.0 / (dist_lower ** 3 + self.BAND_PRESSURE_EPSILON)
            if F_lower > self.BAND_PRESSURE_CAP:
                F_lower = self.BAND_PRESSURE_CAP
        repulsion = -F_upper if z > 0 else F_lower
        net_force = F_gravity + momentum_val + repulsion

        # --- Probabilities & Entropy (kernel lines 148-176) ---
        E0 = -(z * z) / 2.0
        E1 = -((z - 2.0) ** 2) / 2.0
        E2 = -((z + 2.0) ** 2) / 2.0
        max_E = max(E0, E1, E2)
        p0 = math.exp(E0 - max_E)
        p1 = math.exp(E1 - max_E)
        p2 = math.exp(E2 - max_E)
        total_p = p0 + p1 + p2
        p0 /= total_p
        p1 /= total_p
        p2 /= total_p
        eps = 1e-10
        entropy_val = -(p0 * math.log(p0 + eps) + p1 * math.log(p1 + eps) + p2 * math.log(p2 + eps))
        entropy_norm = entropy_val / self.LOG_3

        # --- Hurst exponent (kernel lines 248-319) ---
        # Batch returns hurst=0.5 (default) for bars with i < HURST_WINDOW
        h_buf = inc.prices
        h_n = len(h_buf)
        ws = HURST_WINDOW

        hurst = 0.5  # default — matches CUDA kernel for i < window_size
        # CUDA kernel: `if i < window_size: return` -> needs index >= 30 (31+ bars total)
        if h_n > ws:
            sizes = [max(ws // 8, 4), max(ws // 4, 8), max(ws // 2, 16), ws]
            log_n_arr = []
            log_rs_arr = []

            for sz in sizes:
                start = h_n - sz
                p_start = h_buf[start]
                p_end = h_buf[h_n - 1]
                mean_ret = (p_end - p_start) / (sz - 1) if sz > 1 else 0.0

                cum_dev = 0.0
                max_dev = -1e30
                min_dev = 1e30
                std_sum = 0.0
                prev_p = p_start

                for k_idx in range(start + 1, h_n):
                    curr_p = h_buf[k_idx]
                    ret = (curr_p - prev_p) - mean_ret
                    cum_dev += ret
                    if cum_dev > max_dev:
                        max_dev = cum_dev
                    if cum_dev < min_dev:
                        min_dev = cum_dev
                    std_sum += ret * ret
                    prev_p = curr_p

                R = max_dev - min_dev
                S = math.sqrt(std_sum / (sz - 1)) if sz > 1 else 1e-10
                S = max(S, 1e-10)
                rs = R / S
                log_n_arr.append(math.log(float(sz)))
                log_rs_arr.append(math.log(max(rs, 1e-10)))

            # Linear regression: log(R/S) = H * log(n) + c
            sx = sum(log_n_arr)
            sy = sum(log_rs_arr)
            sxy = sum(a * b for a, b in zip(log_n_arr, log_rs_arr))
            sxx = sum(a * a for a in log_n_arr)
            d = 4.0 * sxx - sx * sx
            hurst = (4.0 * sxy - sx * sy) / d if abs(d) > 1e-12 else 0.5
            hurst = max(0.0, min(1.0, hurst))

        # --- z_scores buffer for PID & oscillation ---
        # Sized to SFE_WINDOW (300) so PID rolling window matches batch's cumsum
        inc.z_scores.append(z)
        inc._trim(inc.z_scores, 300)

        # --- PID Control Force ---
        # Batch: rolling mean of z_scores over _pid_window (30).
        # For bars < window: expanding mean (cumsum[i] / (i+1)).
        # For bars >= window: (cumsum[i] - cumsum[i-window]) / window.
        # feed_bar's z_scores buffer matches batch's z_scores array (includes warmup zeros).
        pid_kp = DEFAULT_PID_KP
        pid_ki = DEFAULT_PID_KI
        pid_kd = DEFAULT_PID_KD
        _pid_window = DEFAULT_PID_INTEGRAL_WINDOW  # 30
        pid_p = pid_kp * z
        zs = inc.z_scores
        n_zs = len(zs)
        if n_zs > _pid_window:
            # Batch: rolling mean when n > _pid_window
            rolling_integral = sum(zs[-_pid_window:]) / _pid_window
        else:
            # Batch fallback: raw cumulative sum (NOT mean) when n <= _pid_window
            rolling_integral = sum(zs)
        pid_i_val = pid_ki * max(-10.0, min(10.0, rolling_integral))
        pid_d_val = pid_kd * (z - inc.prev_z)
        term_pid = pid_p + pid_i_val + pid_d_val
        inc.prev_z = z

        # --- Oscillation Entropy (5-bar rolling std, ddof=1) ---
        # Batch uses _compute_rolling_std_numba which backfills first (window-1) bars
        # with osc_std[window-1]. We match by requiring full window before computing.
        _ow = min(5, rp)
        if n_zs >= _ow:
            osc_slice = zs[-_ow:]
            osc_mean = sum(osc_slice) / _ow
            osc_var = sum((v - osc_mean) ** 2 for v in osc_slice) / (_ow - 1)
            osc_std = math.sqrt(max(osc_var, 0.0))
        else:
            osc_std = 0.0
        osc_entropy_norm = 1.0 / (1.0 + osc_std)

        # --- Swing Noise (30-bar max drawdown/drawup) ---
        # Batch kernel: start_idx = _ni - noise_window, loop start_idx+1.._ni inclusive
        # That's noise_window+1 bars total (start_idx through _ni). Need 31 in buffer.
        _noise_window = 30
        _tick_size = 0.25
        h_arr = inc.highs
        l_arr = inc.lows
        sw_n = len(h_arr)
        if sw_n > _noise_window:  # need noise_window+1 bars (31)
            sw_start = sw_n - _noise_window - 1
            run_hi = h_arr[sw_start]
            run_lo = l_arr[sw_start]
            max_dd = run_hi - l_arr[sw_start]
            max_du = h_arr[sw_start] - run_lo
            for j in range(sw_start + 1, sw_n):
                if h_arr[j] > run_hi:
                    run_hi = h_arr[j]
                dd = run_hi - l_arr[j]
                if dd > max_dd:
                    max_dd = dd
                if l_arr[j] < run_lo:
                    run_lo = l_arr[j]
                du = h_arr[j] - run_lo
                if du > max_du:
                    max_du = du
            swing_noise = max(max_dd, max_du) / _tick_size
        else:
            swing_noise = 35.0

        # --- OU First-Passage Probabilities ---
        _B = 3.0
        _inv_sqrt2 = 1.0 / math.sqrt(2.0)
        _erfi_B = float(erfi(_B * _inv_sqrt2))
        _erfi_z_val = float(erfi(abs(z) * _inv_sqrt2))
        rev_prob = max(0.0, min(1.0, 1.0 - _erfi_z_val / _erfi_B))
        brk_prob = max(0.0, min(1.0, _erfi_z_val / _erfi_B))
        rev_potential = max(0.0, 0.025 * (9.0 - z * z))

        # --- Band Zone ---
        abs_z = abs(z)
        if abs_z < 1.0:
            band_zone = 'INNER'
        elif abs_z < 2.0:
            band_zone = 'CHAOS'
        elif z >= 2.0:
            band_zone = 'UPPER_EXTREME'
        else:
            band_zone = 'LOWER_EXTREME'

        # --- Band flags ---
        cascade_detected = abs_z > 2.0 and abs(velocity) > self.VELOCITY_THRESHOLD
        structure_confirmed = abs(momentum_val) > self.MOMENTUM_THRESHOLD and entropy_norm < self.ENTROPY_THRESHOLD

        # --- Trend Direction ---
        slope_strength = (abs(slope) * rp) / (sigma + 1e-6)
        if slope_strength > 1.0:
            trend_dir = 'UP' if slope > 0 else 'DOWN'
        else:
            trend_dir = 'RANGE'

        # --- Volume Delta ---
        vol_delta = volume if close > open_ else (-volume if close < open_ else 0.0)

        # --- Construct MarketState ---
        state = MarketState(
            regression_center=center,
            upper_band_2sigma=center + 2.0 * sigma,
            lower_band_2sigma=center - 2.0 * sigma,
            upper_band_3sigma=center + 3.0 * sigma,
            lower_band_3sigma=center - 3.0 * sigma,
            price=price,
            velocity=velocity,
            z_score=z,
            mean_reversion_force=-0.5 * z * sigma,
            F_upper_band=0.0,
            F_lower_band=0.0,
            F_momentum=momentum_val,
            net_force=net_force,
            prob_weight_center=math.sqrt(p0),
            prob_weight_upper=math.sqrt(p1),
            prob_weight_lower=math.sqrt(p2),
            P_at_center=p0,
            P_near_upper=p1,
            P_near_lower=p2,
            entropy=entropy_val,
            entropy_normalized=entropy_norm,
            pattern_maturity=0.0,
            momentum_strength=momentum_val,
            structure_confirmed=structure_confirmed,
            cascade_detected=cascade_detected,
            reversal_confirmed=False,
            band_zone=band_zone,
            stability_index=1.0,
            reversion_probability=rev_prob,
            breakout_probability=brk_prob,
            reversion_potential=rev_potential,
            pattern_type='NONE',
            candlestick_pattern='NONE',
            trend_direction_15m=trend_dir,
            hurst_exponent=hurst,
            adx_strength=adx_val,
            dmi_plus=dmi_plus_val,
            dmi_minus=dmi_minus_val,
            adx_prev=prev_adx_out,
            di_plus_prev=prev_dmi_plus_out,
            di_minus_prev=prev_dmi_minus_out,
            volume_delta=vol_delta,
            regression_sigma=sigma,
            term_pid=term_pid,
            oscillation_entropy_normalized=osc_entropy_norm,
            lyapunov_exponent=0.0,
            market_regime='STABLE',
            swing_noise_ticks=swing_noise,
            timestamp=timestamp,
        )

        return state

    def reset_incremental(self):
        """Clear all incremental state, start fresh."""
        self._inc = _IncrementalState()

    def get_incremental_state(self) -> dict:
        """Serializable snapshot of incremental state."""
        inc = self._inc
        return {
            'bar_count': inc.bar_count,
            'prices': list(inc.prices),
            'highs': list(inc.highs),
            'lows': list(inc.lows),
            'closes': list(inc.closes),
            'z_scores': list(inc.z_scores),
            'prev_close': inc.prev_close,
            'prev_high': inc.prev_high,
            'prev_low': inc.prev_low,
            'adx_init_tr': list(inc.adx_init_tr),
            'adx_init_plus': list(inc.adx_init_plus),
            'adx_init_minus': list(inc.adx_init_minus),
            'smooth_tr': inc.smooth_tr,
            'smooth_plus': inc.smooth_plus,
            'smooth_minus': inc.smooth_minus,
            'dx_sum': inc.dx_sum,
            'dx_count': inc.dx_count,
            'prev_adx': inc.prev_adx,
            'prev_dmi_plus': inc.prev_dmi_plus,
            'prev_dmi_minus': inc.prev_dmi_minus,
            'prev_z': inc.prev_z,
        }

    def set_incremental_state(self, state: dict):
        """Restore incremental state from snapshot."""
        inc = self._inc
        inc.bar_count = state['bar_count']
        inc.prices = list(state['prices'])
        inc.highs = list(state['highs'])
        inc.lows = list(state['lows'])
        inc.closes = list(state['closes'])
        inc.z_scores = list(state['z_scores'])
        inc.prev_close = state['prev_close']
        inc.prev_high = state['prev_high']
        inc.prev_low = state['prev_low']
        inc.adx_init_tr = list(state['adx_init_tr'])
        inc.adx_init_plus = list(state['adx_init_plus'])
        inc.adx_init_minus = list(state['adx_init_minus'])
        inc.smooth_tr = state['smooth_tr']
        inc.smooth_plus = state['smooth_plus']
        inc.smooth_minus = state['smooth_minus']
        inc.dx_sum = state['dx_sum']
        inc.dx_count = state['dx_count']
        inc.prev_adx = state['prev_adx']
        inc.prev_dmi_plus = state['prev_dmi_plus']
        inc.prev_dmi_minus = state['prev_dmi_minus']
        inc.prev_z = state['prev_z']

    def calculate_market_state(
        self, 
        df_macro: pd.DataFrame,   # 15min bars
        df_micro: pd.DataFrame,   # 15sec bars
        current_price: float,
        current_volume: float,
        tick_velocity: float,
        context: dict = None,     # Optional multi-timeframe context
        params: dict = None       # Optional physics parameters
    ) -> MarketState:
        """
        Compute complete MarketState from macro/micro dataframes.
        Uses GPU batch computation internally for the window.
        """
        if len(df_macro) < self.regression_period:
            return MarketState.null_state()

        # We can reuse batch_compute_states logic but for a single step (the last one)
        # Pass use_cuda explicitly based on self.use_gpu to allow fallback
        results = self.batch_compute_states(df_macro, use_cuda=self.use_gpu)
        if not results:
             return MarketState.null_state()

        # Get the last state
        last_result = results[-1]
        state = last_result['state']
        
        # Inject context if provided
        context_args = {}
        if context:
            if 'daily' in context and context['daily']:
                context_args['daily_trend'] = context['daily'].trend
                context_args['daily_volatility'] = context['daily'].volatility
                context_args['daily_pattern'] = context['daily'].fractal_pattern
            if 'h4' in context and context['h4']:
                context_args['h4_trend'] = context['h4'].trend
                context_args['session'] = context['h4'].session
                context_args['h4_pattern'] = context['h4'].fractal_pattern
            if 'h1' in context and context['h1']:
                context_args['h1_trend'] = context['h1'].trend
                context_args['h1_pattern'] = context['h1'].fractal_pattern
            if 'context_level' in context:
                context_args['context_level'] = context['context_level']

        return state

    def _detect_patterns_unified(self, opens: np.ndarray, highs: np.ndarray, lows: np.ndarray, closes: np.ndarray):
        """
        Detects both geometric and candlestick patterns using unified CUDA kernel if available/efficient.
        Falls back to vectorized CPU implementation.
        """
        # Threshold to avoid Numba performance warning on small grids (grid size 1)
        MIN_CUDA_LEN = 1024

        if self.use_gpu and CUDA_PATTERNS_AVAILABLE and len(highs) >= MIN_CUDA_LEN:
            try:
                return detect_patterns_cuda(opens, highs, lows, closes)
            except Exception as e:
                pass

        geo = detect_geometric_patterns_vectorized(highs, lows)
        cdl = detect_candlestick_patterns_vectorized(opens, highs, lows, closes)
        return geo, cdl

    # ═══════════════════════════════════════════════════════════════════════
    # VECTORIZED BATCH COMPUTATION (processes all bars at once)
    # ═══════════════════════════════════════════════════════════════════════

    def batch_compute_states(self, day_data: pd.DataFrame, use_cuda: bool = True, params: dict = None) -> list:
        """
        Compute ALL MarketState objects for a day using Fused CUDA Physics Kernels or Optimized CPU Fallback.
        """
        params = params or {}
        n = len(day_data)
        rp = self.regression_period

        if n < rp:
            return []

        # Prepare input data
        # Ensure contiguous float64 arrays
        prices = day_data['price'].values.astype(np.float64) if 'price' in day_data.columns else day_data['close'].values.astype(np.float64)
        if not prices.flags['C_CONTIGUOUS']:
            prices = np.ascontiguousarray(prices)

        volumes = day_data['volume'].values.astype(np.float64) if 'volume' in day_data.columns else np.zeros(n, dtype=np.float64)
        if not volumes.flags['C_CONTIGUOUS']:
            volumes = np.ascontiguousarray(volumes)

        # Unified pattern detection (Geometric/Candlestick)
        if 'high' in day_data.columns:
            highs = day_data['high'].values.astype(np.float64)
            lows = day_data['low'].values.astype(np.float64)
            opens = day_data['open'].values.astype(np.float64)
        else:
            highs = prices
            lows = prices
            opens = prices

        if 'close' in day_data.columns:
            closes = day_data['close'].values.astype(np.float64)
        else:
            closes = prices

        # Volume delta: buy bar (+vol), sell bar (-vol), doji (0)
        if 'volume' in day_data.columns:
            _vol = day_data['volume'].values.astype(np.float64)
            _open = day_data['open'].values.astype(np.float64) if 'open' in day_data.columns else prices
            _close = day_data['close'].values.astype(np.float64) if 'close' in day_data.columns else prices
            volume_delta_arr = np.where(_close > _open, _vol, np.where(_close < _open, -_vol, 0.0))
        else:
            volume_delta_arr = np.zeros(n, dtype=np.float64)

        pattern_types, candlestick_types = self._detect_patterns_unified(opens, highs, lows, closes)

        # Output variables to be filled by either GPU or CPU path
        center = None
        sigma = None
        slope = None
        z_scores = None
        velocity = None
        force = None
        momentum = None
        entropy_normalized = None
        entropy = None
        prob0 = None
        prob1 = None
        prob2 = None
        band_snap = None
        trend_drive = None

        # New Indicators
        hurst_arr = None
        adx_arr = None
        dmi_plus_arr = None
        dmi_minus_arr = None

        if use_cuda and self.use_gpu and CUDA_PHYSICS_AVAILABLE:
            # Output arrays (allocated on GPU directly or mapped)
            d_prices = cuda.to_device(prices)
            d_volumes = cuda.to_device(volumes)

            # Needed for ADX
            d_highs = cuda.to_device(highs)
            d_lows = cuda.to_device(lows)
            d_closes = cuda.to_device(closes)

            d_center = cuda.device_array(n, dtype=np.float64)
            d_sigma = cuda.device_array(n, dtype=np.float64)
            d_slope = cuda.device_array(n, dtype=np.float64)
            d_z = cuda.device_array(n, dtype=np.float64)
            d_velocity = cuda.device_array(n, dtype=np.float64)
            d_force = cuda.device_array(n, dtype=np.float64)
            d_momentum = cuda.device_array(n, dtype=np.float64)
            d_entropy_normalized = cuda.device_array(n, dtype=np.float64)
            d_entropy = cuda.device_array(n, dtype=np.float64)
            d_prob0 = cuda.device_array(n, dtype=np.float64)
            d_prob1 = cuda.device_array(n, dtype=np.float64)
            d_prob2 = cuda.device_array(n, dtype=np.float64)

            d_band_snap = cuda.device_array(n, dtype=np.bool_)
            d_trend_drive = cuda.device_array(n, dtype=np.bool_)

            # Arrays for ADX
            d_tr = cuda.device_array(n, dtype=np.float64)
            d_plus_dm = cuda.device_array(n, dtype=np.float64)
            d_minus_dm = cuda.device_array(n, dtype=np.float64)

            # Array for Hurst
            d_hurst = cuda.device_array(n, dtype=np.float64)

            # Kernel Launch Configuration
            threads_per_block = 256
            blocks_per_grid = (n + (threads_per_block - 1)) // threads_per_block

            # Use precomputed regression constants from self
            # This avoids recalculating them here

            # 1. Physics Kernel
            compute_regression_kernel[blocks_per_grid, threads_per_block](
                d_prices, d_volumes,
                d_center, d_sigma, d_slope,
                d_z, d_velocity, d_force, d_momentum,
                d_entropy_normalized, d_entropy,
                d_prob0, d_prob1, d_prob2,
                rp, self.mean_x, self.inv_reg_period, self.inv_denom, self.denom
            )

            # 2. Pattern Flags Kernel
            detect_pattern_flags_kernel[blocks_per_grid, threads_per_block](
                d_z, d_velocity, d_momentum, d_entropy_normalized,
                d_band_snap, d_trend_drive
            )

            # 3. ADX/DMI Kernel (Pass 1)
            compute_dm_tr_kernel[blocks_per_grid, threads_per_block](
                d_highs, d_lows, d_closes,
                d_tr, d_plus_dm, d_minus_dm
            )

            # 4. Hurst Kernel
            compute_hurst_kernel[blocks_per_grid, threads_per_block](
                d_prices, d_hurst, HURST_WINDOW
            )

            # Synchronize before CPU pass
            cuda.synchronize()

            # Copy back results
            center = d_center.copy_to_host()
            sigma = d_sigma.copy_to_host()
            slope = d_slope.copy_to_host()
            z_scores = d_z.copy_to_host()
            velocity = d_velocity.copy_to_host()
            force = d_force.copy_to_host()
            momentum = d_momentum.copy_to_host()
            entropy_normalized = d_entropy_normalized.copy_to_host()
            entropy = d_entropy.copy_to_host()
            prob0 = d_prob0.copy_to_host()
            prob1 = d_prob1.copy_to_host()
            prob2 = d_prob2.copy_to_host()

            band_snap = d_band_snap.copy_to_host()
            trend_drive = d_trend_drive.copy_to_host()

            hurst_arr = d_hurst.copy_to_host()

            # Pass 2 ADX on CPU
            tr_raw = d_tr.copy_to_host()
            plus_dm_raw = d_plus_dm.copy_to_host()
            minus_dm_raw = d_minus_dm.copy_to_host()
            adx_arr, dmi_plus_arr, dmi_minus_arr = compute_adx_dmi_cpu(tr_raw, plus_dm_raw, minus_dm_raw, ADX_PERIOD)

        else:
            raise RuntimeError(
                "CUDA not available. StatisticalFieldEngine requires GPU. "
                "Install numba with CUDA support."
            )

        # Extract timestamps
        timestamps = np.zeros(n, dtype=np.float64)
        if 'timestamp' in day_data.columns:
            ts_col = day_data['timestamp']
            if pd.api.types.is_datetime64_any_dtype(ts_col):
                # Convert to seconds (float)
                timestamps = ts_col.astype('int64').values / 1e9
            else:
                # Assuming already seconds or float
                timestamps = ts_col.values.astype(np.float64)
        elif isinstance(day_data.index, pd.DatetimeIndex):
            timestamps = day_data.index.astype('int64').values / 1e9

        # ─── PID Control Force ──────────────────────────────────────────────────────
        # Models the algorithmic market-maker control force acting on price at each bar.
        # P = proportional to current deviation from equilibrium (z_score)
        # I = integral of accumulated deviation (cumulative bias)
        # D = derivative = rate of change of z_score (dampening)
        pid_kp = params.get('pid_kp', DEFAULT_PID_KP)   # 0.5
        pid_ki = params.get('pid_ki', DEFAULT_PID_KI)   # 0.1
        pid_kd = params.get('pid_kd', DEFAULT_PID_KD)   # 0.2

        pid_p       = pid_kp * z_scores
        # Integral term: rolling mean over window (not cumsum)
        # Rolling mean converges after window bars regardless of start point.
        # With window=30, OOS and live produce identical values after 30 bars.
        _pid_window = params.get('pid_integral_window', DEFAULT_PID_INTEGRAL_WINDOW)
        if _pid_window > 0 and n > _pid_window:
            # Rolling mean of z_scores over last _pid_window bars
            _cs = np.cumsum(z_scores)
            _cs[_pid_window:] = _cs[_pid_window:] - _cs[:-_pid_window]
            _rolling_mean = np.zeros_like(z_scores)
            _rolling_mean[:_pid_window] = _cs[:_pid_window] / np.arange(1, _pid_window + 1)
            _rolling_mean[_pid_window:] = _cs[_pid_window:] / _pid_window
            pid_i = pid_ki * np.clip(_rolling_mean, -10.0, 10.0)
        else:
            # Fallback: standard cumsum (for short sequences or window=0)
            pid_i = pid_ki * np.clip(np.cumsum(z_scores), -10.0, 10.0)
        pid_d       = np.zeros_like(z_scores)
        pid_d[1:]   = pid_kd * np.diff(z_scores)
        term_pid_arr = pid_p + pid_i + pid_d   # shape: (n,)

        # ─── Oscillation Coherence ──────────────────────────────────────────────────
        # Rolling std of z_score over a short window.  Low std = tight periodic
        # oscillation (PID regime).  High std = chaotic / trending.
        # Inverted and normalised to (0, 1] so 1 = perfectly tight oscillation.
        _ow = min(5, rp)
        osc_std = _compute_rolling_std_numba(z_scores, n, _ow)

        oscillation_entropy_normalized_arr = 1.0 / (1.0 + osc_std)   # (0, 1]
        np.nan_to_num(oscillation_entropy_normalized_arr, copy=False, nan=0.0)

        # ─── Analytical OU First-Passage Probabilities ────────────────────────
        # Replaces Monte Carlo (500 paths × 600 steps) with exact solution.
        # For OU process dz = -θz dt + σ_z dW, the probability of hitting
        # boundary 0 (center) before boundary B (3σ breakout boundary) starting at z:
        #   P(tunnel) = 1 - erfi(|z|/√2) / erfi(B/√2)
        # This is universal for any θ,σ (the ratio θ/σ² cancels in z-space).
        _B = 3.0  # breakout boundary in z-score units
        _inv_sqrt2 = 1.0 / np.sqrt(2.0)
        _erfi_B = erfi(_B * _inv_sqrt2)  # scalar constant ≈ 28.3
        _abs_z_arr = np.abs(z_scores)
        _erfi_z = erfi(_abs_z_arr * _inv_sqrt2)
        reversion_prob = np.clip(1.0 - _erfi_z / _erfi_B, 0.0, 1.0)
        breakout_prob = np.clip(_erfi_z / _erfi_B, 0.0, 1.0)
        # OU potential V(z) = θz²/2, barrier = V(B) - V(z)
        reversion_potential_arr = np.clip(0.025 * (9.0 - z_scores**2), 0.0, np.inf)

        # Reconstruct result list
        # Optimization: Vectorize logic to reduce Python loop overhead

        # 1. Band Zones
        abs_z = np.abs(z_scores)
        cond_stable = abs_z < 1.0
        cond_chaos = abs_z < 2.0
        cond_upper_extreme = z_scores >= 2.0

        # Default is LOWER_EXTREME (z <= -2.0)
        # 'CHAOS' = transition zone (1σ < |z| < 2σ)  -- name is historical, do not rename
        # (renaming would invalidate all BayesianBrain hash keys)
        band_zone_arr = np.select(
            [cond_stable, cond_chaos, cond_upper_extreme],
            ['INNER', 'CHAOS', 'UPPER_EXTREME'],
            default='LOWER_EXTREME'
        )

        # 2. Probability weights (sqrt of probabilities)
        pw_center_arr = np.sqrt(prob0)
        pw_upper_arr = np.sqrt(prob1)
        pw_lower_arr = np.sqrt(prob2)

        # 3. Trend Direction
        # slope_strength = (abs(slope[i]) * rp) / (sigma[i] + 1e-6)
        slope_strength = (np.abs(slope) * rp) / (sigma + 1e-6)

        # 'UP' if slope > 0 else 'DOWN'
        trend_dir_temp = np.where(slope > 0, 'UP', 'DOWN')
        trend_direction_arr = np.where(slope_strength > 1.0, trend_dir_temp, 'RANGE')

        # 4. Confirmation signals (Already boolean arrays)
        # band_snap and trend_drive are boolean arrays from kernels/vectorized logic

        # --- Swing noise: max intra-wave pullback over rolling window ---
        # Measures "how much pullback is normal right now" in ticks.
        # Used by exit engine to set dynamic giveback threshold.
        _noise_window = 30  # 30 bars — consistent with other windows
        _tick_size = params.get('tick_size', 0.25)
        swing_noise = _compute_swing_noise_numba(highs, lows, n, _noise_window, _tick_size)

        results = [
            {
                'bar_idx': i,
                'state': MarketState(
                    regression_center=center[i],
                    upper_band_2sigma=center[i] + 2.0 * sigma[i],
                    lower_band_2sigma=center[i] - 2.0 * sigma[i],
                    upper_band_3sigma=center[i] + 3.0 * sigma[i],
                    lower_band_3sigma=center[i] - 3.0 * sigma[i],
                    price=prices[i],
                    velocity=velocity[i],
                    z_score=z_scores[i],
                    mean_reversion_force=-0.5 * z_scores[i] * sigma[i],
                    F_upper_band=0.0,
                    F_lower_band=0.0,
                    F_momentum=momentum[i],
                    net_force=force[i],
                    prob_weight_center=pw_center_arr[i],
                    prob_weight_upper=pw_upper_arr[i],
                    prob_weight_lower=pw_lower_arr[i],
                    P_at_center=prob0[i],
                    P_near_upper=prob1[i],
                    P_near_lower=prob2[i],
                    entropy=entropy[i],
                    entropy_normalized=entropy_normalized[i],
                    pattern_maturity=0.0,
                    momentum_strength=momentum[i],
                    structure_confirmed=bool(trend_drive[i]),
                    cascade_detected=bool(band_snap[i]),
                    reversal_confirmed=False,
                    band_zone=band_zone_arr[i],
                    stability_index=1.0,
                    reversion_probability=reversion_prob[i],
                    breakout_probability=breakout_prob[i],
                    reversion_potential=reversion_potential_arr[i],
                    pattern_type=str(pattern_types[i]),
                    candlestick_pattern=str(candlestick_types[i]),
                    trend_direction_15m=trend_direction_arr[i],
                    hurst_exponent=hurst_arr[i],
                    adx_strength=adx_arr[i], dmi_plus=dmi_plus_arr[i], dmi_minus=dmi_minus_arr[i],
                    adx_prev=adx_arr[i-1] if i > 0 else 0.0,
                    di_plus_prev=dmi_plus_arr[i-1] if i > 0 else 0.0,
                    di_minus_prev=dmi_minus_arr[i-1] if i > 0 else 0.0,
                    volume_delta=float(volume_delta_arr[i]),
                    regression_sigma=sigma[i],
                    term_pid=float(term_pid_arr[i]),
                    oscillation_entropy_normalized=float(oscillation_entropy_normalized_arr[i]),
                    lyapunov_exponent=0.0,
                    market_regime='STABLE',
                    swing_noise_ticks=float(swing_noise[i]),
                    timestamp=timestamps[i]
                ),
                'price': prices[i],
                'structure_ok': band_zone_arr[i] in ('UPPER_EXTREME', 'LOWER_EXTREME')
            }
            for i in range(n)
        ]

        return results
