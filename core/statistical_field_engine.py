"""
Statistical Field Engine
Computes regression bands, z-scores, probability distributions,
and mean-reversion/breakout statistics from price data.
GPU-accelerated via Numba CUDA kernels.
"""
import numpy as np
import pandas as pd
from numba import cuda
from numpy.lib.stride_tricks import sliding_window_view
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
DEFAULT_REVERSION_THETA = 0.5


logger = logging.getLogger(__name__)

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
        # Using 500 bars (~2 hours at 15s) to estimate distribution
        self.residual_history = []
        self.residual_window = 500

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
        pid_i       = pid_ki * np.clip(np.cumsum(z_scores), -10.0, 10.0)
        pid_d       = np.zeros_like(z_scores)
        pid_d[1:]   = pid_kd * np.diff(z_scores)
        term_pid_arr = pid_p + pid_i + pid_d   # shape: (n,)

        # ─── Oscillation Coherence ──────────────────────────────────────────────────
        # Rolling std of z_score over a short window.  Low std = tight periodic
        # oscillation (PID regime).  High std = chaotic / trending.
        # Inverted and normalised to (0, 1] so 1 = perfectly tight oscillation.
        _ow = min(5, rp)
        osc_std = np.full(n, np.nan)
        if n >= _ow:
             z_windows = sliding_window_view(z_scores, window_shape=_ow)
             osc_std[_ow-1:] = z_windows.std(axis=1, ddof=1)
             if n > _ow - 1:
                  osc_std[:_ow - 1] = osc_std[_ow - 1]

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
        _noise_window = 32  # ~8 min at 15s bars
        _tick_size = params.get('tick_size', 0.25)
        swing_noise = np.full(n, 35.0)  # default 35 ticks
        for _ni in range(_noise_window, n):
            _seg_hi = highs[_ni - _noise_window:_ni + 1]
            _seg_lo = lows[_ni - _noise_window:_ni + 1]
            # Max drawdown from running high (long-side noise)
            _run_hi = np.maximum.accumulate(_seg_hi)
            _dd = (_run_hi - _seg_lo).max() / _tick_size
            # Max drawup from running low (short-side noise)
            _run_lo = np.minimum.accumulate(_seg_lo)
            _du = (_seg_hi - _run_lo).max() / _tick_size
            swing_noise[_ni] = max(_dd, _du)

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
