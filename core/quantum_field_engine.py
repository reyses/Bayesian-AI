"""
Quantum Field Calculator
Computes three-body gravitational fields + quantum wave function
Integrates Nightmare Protocol gravity calculations
"""
import numpy as np
import pandas as pd
import math
import numba
from numba import cuda
from numpy.lib.stride_tricks import sliding_window_view
import logging

from core.three_body_state import ThreeBodyQuantumState
from core.risk_engine import QuantumRiskEngine
from core.pattern_utils import (
    PATTERN_NONE, PATTERN_COMPRESSION, PATTERN_WEDGE, PATTERN_BREAKDOWN,
    detect_geometric_pattern, detect_candlestick_pattern,
    detect_geometric_patterns_vectorized, detect_candlestick_patterns_vectorized
)

from core.physics_utils import compute_adx_dmi_cpu, ADX_PERIOD, HURST_WINDOW

# Core CUDA Physics
try:
    from core.cuda_physics import (
        compute_physics_kernel, detect_archetype_kernel,
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

# Optional: Fractal & Trend Libraries
try:
    import matplotlib
    # Force matplotlib to use a non-interactive backend if possible to avoid some display issues
    matplotlib.use('Agg')
    import matplotlib.pyplot
except ImportError:
    pass

try:
    import pandas_ta as ta
    PANDAS_TA_AVAILABLE = True
except ImportError:
    PANDAS_TA_AVAILABLE = False

try:
    from hurst import compute_Hc
    HURST_AVAILABLE = True
except ImportError:
    HURST_AVAILABLE = False

# Cascade Constants
VELOCITY_CASCADE_THRESHOLD = 1.0  # Points per second
RANGE_CASCADE_THRESHOLD = 10.0    # Points range in candle

# Risk & Trend Constants
RISK_THETA = 0.1
RISK_HORIZON_SECONDS = 600
# HURST_WINDOW and ADX_LENGTH might be redefined here but we prefer constants from cuda_physics if available
ADX_LENGTH = ADX_PERIOD

# PID Control Constants (Default/Fallback)
DEFAULT_PID_KP = 0.5
DEFAULT_PID_KI = 0.1
DEFAULT_PID_KD = 0.2
DEFAULT_GRAVITY_THETA = 0.5

# Optional CUDA support
try:
    import torch
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

logger = logging.getLogger(__name__)

class QuantumFieldEngine:
    """
    Unified field calculator — GPU-accelerated when CUDA available
    - Nightmare Protocol (gravity wells, O-U process)
    - Three-body dynamics (Lagrange points, tidal forces)
    - Quantum mechanics (superposition, tunneling)
    """

    def __init__(self, regression_period: int = 21, use_gpu: bool = None):
        self.regression_period = regression_period
        self.SIGMA_ROCHE_MULTIPLIER = 2.0
        self.SIGMA_EVENT_MULTIPLIER = 3.0
        self.TIDAL_FORCE_EXPONENT = 2.0

        # Physics Constants matching CUDA kernel
        self.GRAVITY_THETA = 0.5
        self.REPULSION_EPSILON = 0.01
        self.REPULSION_FORCE_CAP = 100.0
        self.VELOCITY_THRESHOLD = 0.5
        self.MOMENTUM_THRESHOLD = 5.0
        self.COHERENCE_THRESHOLD = 0.3

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

        # Keep Torch device for legacy compatibility if needed, but primary compute is Numba
        if TORCH_AVAILABLE and torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = None

        # Risk Engine (Monte Carlo)
        self.risk_engine = QuantumRiskEngine(
            theta=RISK_THETA,
            horizon_seconds=RISK_HORIZON_SECONDS
        )
    
    def calculate_three_body_state(
        self, 
        df_macro: pd.DataFrame,   # 15min bars
        df_micro: pd.DataFrame,   # 15sec bars
        current_price: float,
        current_volume: float,
        tick_velocity: float,
        context: dict = None,     # Optional multi-timeframe context
        params: dict = None       # Optional physics parameters
    ) -> ThreeBodyQuantumState:
        """
        MASTER FUNCTION: Computes complete quantum state using NIGHTMARE FIELD EQUATION
        Uses GPU batch computation internally for the window.
        """
        if len(df_macro) < self.regression_period:
            return ThreeBodyQuantumState.null_state()

        # We can reuse batch_compute_states logic but for a single step (the last one)
        # Pass use_cuda explicitly based on self.use_gpu to allow fallback
        results = self.batch_compute_states(df_macro, use_cuda=self.use_gpu)
        if not results:
             return ThreeBodyQuantumState.null_state()

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

    def _compute_hurst_numpy(self, prices: np.ndarray, window: int = 100):
        """
        Numpy Hurst exponent via R/S method (Vectorized).
        Significantly faster than iterative polyfit.
        """
        n = len(prices)
        hurst = np.full(n, 0.5)

        if n < window:
            return hurst

        all_returns = np.diff(prices)

        raw_sizes = [window//8, window//4, window//2, window]
        valid_sizes = [max(sz, 4) for sz in raw_sizes]

        # Check if we have enough distinct sizes for regression
        if len(set(valid_sizes)) < 2:
            return hurst

        log_ns = np.log(valid_sizes)

        # Precompute pseudo-inverse
        A = np.vstack([log_ns, np.ones(len(log_ns))]).T

        if np.linalg.matrix_rank(A) < 2:
            return hurst

        pinv = np.linalg.pinv(A)
        pinv_slope = pinv[0, :] # Shape (4,)

        unique_sizes = sorted(list(set(valid_sizes)))
        size_results = {}

        for sz in unique_sizes:
            w_ret = sz - 1
            # Vectorized R/S over all possible windows of size w_ret
            windows = sliding_window_view(all_returns, window_shape=w_ret)

            mean_r = windows.mean(axis=1, keepdims=True)
            devs = np.cumsum(windows - mean_r, axis=1)
            R = devs.max(axis=1) - devs.min(axis=1)
            S = windows.std(axis=1, ddof=1)
            S = np.maximum(S, 1e-10)

            RS = R / S
            log_RS = np.log(np.maximum(RS, 1e-10))

            size_results[sz] = log_RS

        Y_rows = []
        for sz in valid_sizes:
            res = size_results[sz]
            w_ret = sz - 1
            start_idx = window - w_ret

            # Ensure we don't go out of bounds (although start_idx >= 1 usually)
            if start_idx < 0:
                 start_idx = 0

            # Slice results to align with `hurst[window:]`
            # res has length `n - w_ret`.
            # We need `n - window` points.
            sliced_res = res[start_idx : start_idx + (n - window)]
            Y_rows.append(sliced_res)

        Y = np.vstack(Y_rows)

        slopes = pinv_slope @ Y
        slopes = np.clip(slopes, 0.0, 1.0)

        hurst[window:] = slopes

        return hurst

    def _batch_compute_cpu(self, prices: np.ndarray, highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, volumes: np.ndarray, rp: int) -> dict:
        """
        Vectorized CPU implementation of physics calculations using NumPy sliding window view.
        Matches logic in core/cuda_physics.py.
        """
        n = len(prices)

        # Initialize output arrays
        center = np.zeros(n, dtype=np.float64)
        sigma = np.zeros(n, dtype=np.float64)
        slope = np.zeros(n, dtype=np.float64)
        z_scores = np.zeros(n, dtype=np.float64)
        velocity = np.zeros(n, dtype=np.float64)
        force = np.zeros(n, dtype=np.float64)
        momentum = np.zeros(n, dtype=np.float64)
        coherence = np.ones(n, dtype=np.float64)
        entropy = np.zeros(n, dtype=np.float64)
        prob0 = np.ones(n, dtype=np.float64)
        prob1 = np.zeros(n, dtype=np.float64)
        prob2 = np.zeros(n, dtype=np.float64)

        roche_snap = np.zeros(n, dtype=bool)
        structural_drive = np.zeros(n, dtype=bool)

        # Indicators
        hurst = np.full(n, 0.5, dtype=np.float64)
        adx_strength = np.zeros(n, dtype=np.float64)
        dmi_plus = np.zeros(n, dtype=np.float64)
        dmi_minus = np.zeros(n, dtype=np.float64)

        if n < rp:
            return {
                'center': center, 'sigma': sigma, 'slope': slope, 'z': z_scores,
                'velocity': velocity, 'force': force, 'momentum': momentum,
                'coherence': coherence, 'entropy': entropy,
                'prob0': prob0, 'prob1': prob1, 'prob2': prob2,
                'roche': roche_snap, 'drive': structural_drive,
                'hurst': hurst, 'adx': adx_strength, 'dmi_plus': dmi_plus, 'dmi_minus': dmi_minus
            }

        # 1. Rolling Linear Regression
        # Use convolution for O(N) calculation instead of O(N*rp)

        # Kernel for sum_y: ones
        kernel_sum = np.ones(rp)

        # Kernel for sum_xy: x reversed [rp-1, ..., 0]
        # We want sum(prices[i+j] * x[j]) for j in 0..rp-1
        # Convolve(f, g)[n] = sum f[m] g[n-m].
        # Let n be the index of the result. Corresponds to window ending at n (in valid mode).
        # We want kernel such that k[j] matches x[rp-1-j].
        # x = [0, 1, ... rp-1].
        # kernel = [rp-1, ..., 0].
        x = np.arange(rp)
        kernel_xy = x[::-1]

        sum_y = np.convolve(prices, kernel_sum, mode='valid')
        sum_xy = np.convolve(prices, kernel_xy, mode='valid')
        sum_yy = np.convolve(prices**2, kernel_sum, mode='valid')

        mean_y = sum_y / rp

        # Slope
        # slope = (sum_xy - mean_x * sum_y) * inv_denom
        slopes_valid = (sum_xy - self.mean_x * sum_y) * self.inv_denom

        # Center (at end of window)
        # center = mean_y + slope * ((rp - 1) - mean_x)
        centers_valid = mean_y + slopes_valid * ((rp - 1) - self.mean_x)

        # Sigma
        # sst = sum_yy - rp * mean_y * mean_y
        sst = sum_yy - rp * mean_y * mean_y
        # rss = sst - slope * slope * denom
        rss = sst - slopes_valid * slopes_valid * self.denom
        rss = np.maximum(rss, 0.0) # Clamp

        if rp > 2:
            sigmas_valid = np.sqrt(rss / (rp - 2))
        else:
            sigmas_valid = np.zeros_like(rss)

        sigmas_valid = np.maximum(sigmas_valid, 1e-6)

        # Fill arrays (offset by rp-1)
        start_idx = rp - 1
        slope[start_idx:] = slopes_valid
        center[start_idx:] = centers_valid
        sigma[start_idx:] = sigmas_valid

        # 2. Z-Score & Velocity & Momentum
        # Vectorized operations on full arrays (valid where i >= rp-1)

        # Z-Score
        z_scores[start_idx:] = (prices[start_idx:] - center[start_idx:]) / sigma[start_idx:]

        # Velocity
        velocity[1:] = prices[1:] - prices[:-1]

        # Momentum
        momentum[start_idx:] = (velocity[start_idx:] * volumes[start_idx:]) / sigma[start_idx:]

        # Forces (Gravity)
        F_gravity = -self.GRAVITY_THETA * (z_scores * sigma)

        # Repulsion
        upper_sing = center + self.SIGMA_ROCHE_MULTIPLIER * sigma
        lower_sing = center - self.SIGMA_ROCHE_MULTIPLIER * sigma

        # Use slicing to avoid division by zero on uninitialized parts
        dist_upper = np.abs(prices[start_idx:] - upper_sing[start_idx:]) / sigma[start_idx:]
        dist_lower = np.abs(prices[start_idx:] - lower_sing[start_idx:]) / sigma[start_idx:]

        # Avoid division by zero with epsilon
        dist_upper_cubed = np.power(dist_upper, 3) + self.REPULSION_EPSILON
        dist_lower_cubed = np.power(dist_lower, 3) + self.REPULSION_EPSILON

        # Slice z_scores for condition check
        z_valid = z_scores[start_idx:]

        F_upper = np.where(z_valid > 0, 1.0 / dist_upper_cubed, 0.0)
        F_upper = np.minimum(F_upper, self.REPULSION_FORCE_CAP)

        F_lower = np.where(z_valid < 0, 1.0 / dist_lower_cubed, 0.0)
        F_lower = np.minimum(F_lower, self.REPULSION_FORCE_CAP)

        repulsion = np.where(z_valid > 0, -F_upper, F_lower)

        F_net = F_gravity[start_idx:] + momentum[start_idx:] + repulsion
        force[start_idx:] = F_net

        # 4. Wave Function
        z = z_scores
        E0 = - (z * z) / 2.0
        E1 = - (z - 2.0)**2 / 2.0
        E2 = - (z + 2.0)**2 / 2.0

        max_E = np.maximum(E0, np.maximum(E1, E2))

        p0 = np.exp(E0 - max_E)
        p1 = np.exp(E1 - max_E)
        p2 = np.exp(E2 - max_E)

        total_p = p0 + p1 + p2
        p0 /= total_p
        p1 /= total_p
        p2 /= total_p

        prob0[start_idx:] = p0[start_idx:]
        prob1[start_idx:] = p1[start_idx:]
        prob2[start_idx:] = p2[start_idx:]

        eps = 1e-10
        ent = - (p0 * np.log(p0 + eps) +
                 p1 * np.log(p1 + eps) +
                 p2 * np.log(p2 + eps))

        entropy[start_idx:] = ent[start_idx:]
        coherence[start_idx:] = ent[start_idx:] / self.LOG_3

        # Archetypes
        roche_snap[start_idx:] = (np.abs(z_scores[start_idx:]) > 2.0) & (np.abs(velocity[start_idx:]) > self.VELOCITY_THRESHOLD)
        structural_drive[start_idx:] = (np.abs(momentum[start_idx:]) > self.MOMENTUM_THRESHOLD) & (coherence[start_idx:] < self.COHERENCE_THRESHOLD)

        # ═════ INDICATORS (CPU Fallback) ═════

        # 1. ADX/DMI
        # Compute TR, +DM, -DM manually
        tr_raw = np.zeros(n)
        plus_dm_raw = np.zeros(n)
        minus_dm_raw = np.zeros(n)

        tr_raw[0] = highs[0] - lows[0]
        # Rest computed in loop or vectorized
        # Vectorized TR/DM
        hl = highs[1:] - lows[1:]
        hc = np.abs(highs[1:] - closes[:-1])
        lc = np.abs(lows[1:] - closes[:-1])
        tr_raw[1:] = np.maximum(hl, np.maximum(hc, lc))

        up_move = highs[1:] - highs[:-1]
        down_move = lows[:-1] - lows[1:]

        plus_dm_raw[1:] = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
        minus_dm_raw[1:] = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

        # Use the provided CPU function for Wilder smoothing
        # Note: compute_adx_dmi_cpu is imported from cuda_physics (it's a python function)
        if 'compute_adx_dmi_cpu' in globals():
             adx_strength, dmi_plus, dmi_minus = compute_adx_dmi_cpu(tr_raw, plus_dm_raw, minus_dm_raw, ADX_PERIOD)

        # 2. Hurst
        hurst = self._compute_hurst_numpy(prices, HURST_WINDOW)

        return {
            'center': center, 'sigma': sigma, 'slope': slope, 'z': z_scores,
            'velocity': velocity, 'force': force, 'momentum': momentum,
            'coherence': coherence, 'entropy': entropy,
            'prob0': prob0, 'prob1': prob1, 'prob2': prob2,
            'roche': roche_snap, 'drive': structural_drive,
            'hurst': hurst, 'adx': adx_strength, 'dmi_plus': dmi_plus, 'dmi_minus': dmi_minus
        }

    # ═══════════════════════════════════════════════════════════════════════
    # VECTORIZED BATCH COMPUTATION (processes all bars at once)
    # ═══════════════════════════════════════════════════════════════════════

    def batch_compute_states(self, day_data: pd.DataFrame, use_cuda: bool = True, params: dict = None) -> list:
        """
        Compute ALL ThreeBodyQuantumState objects for a day using Fused CUDA Physics Kernels or Optimized CPU Fallback.
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

        pattern_types, candlestick_types = self._detect_patterns_unified(opens, highs, lows, closes)

        # Output variables to be filled by either GPU or CPU path
        center = None
        sigma = None
        slope = None
        z_scores = None
        velocity = None
        force = None
        momentum = None
        coherence = None
        entropy = None
        prob0 = None
        prob1 = None
        prob2 = None
        roche_snap = None
        structural_drive = None

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
            d_coherence = cuda.device_array(n, dtype=np.float64)
            d_entropy = cuda.device_array(n, dtype=np.float64)
            d_prob0 = cuda.device_array(n, dtype=np.float64)
            d_prob1 = cuda.device_array(n, dtype=np.float64)
            d_prob2 = cuda.device_array(n, dtype=np.float64)

            d_roche = cuda.device_array(n, dtype=np.bool_)
            d_drive = cuda.device_array(n, dtype=np.bool_)

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
            compute_physics_kernel[blocks_per_grid, threads_per_block](
                d_prices, d_volumes,
                d_center, d_sigma, d_slope,
                d_z, d_velocity, d_force, d_momentum,
                d_coherence, d_entropy,
                d_prob0, d_prob1, d_prob2,
                rp, self.mean_x, self.inv_reg_period, self.inv_denom, self.denom
            )

            # 2. Archetype Kernel
            detect_archetype_kernel[blocks_per_grid, threads_per_block](
                d_z, d_velocity, d_momentum, d_coherence,
                d_roche, d_drive
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
            coherence = d_coherence.copy_to_host()
            entropy = d_entropy.copy_to_host()
            prob0 = d_prob0.copy_to_host()
            prob1 = d_prob1.copy_to_host()
            prob2 = d_prob2.copy_to_host()

            roche_snap = d_roche.copy_to_host()
            structural_drive = d_drive.copy_to_host()

            hurst_arr = d_hurst.copy_to_host()

            # Pass 2 ADX on CPU
            tr_raw = d_tr.copy_to_host()
            plus_dm_raw = d_plus_dm.copy_to_host()
            minus_dm_raw = d_minus_dm.copy_to_host()
            adx_arr, dmi_plus_arr, dmi_minus_arr = compute_adx_dmi_cpu(tr_raw, plus_dm_raw, minus_dm_raw, ADX_PERIOD)

        else:
            # CPU Fallback
            cpu_results = self._batch_compute_cpu(prices, highs, lows, closes, volumes, rp)
            center = cpu_results['center']
            sigma = cpu_results['sigma']
            slope = cpu_results['slope']
            z_scores = cpu_results['z']
            velocity = cpu_results['velocity']
            force = cpu_results['force']
            momentum = cpu_results['momentum']
            coherence = cpu_results['coherence']
            entropy = cpu_results['entropy']
            prob0 = cpu_results['prob0']
            prob1 = cpu_results['prob1']
            prob2 = cpu_results['prob2']
            roche_snap = cpu_results['roche']
            structural_drive = cpu_results['drive']

            hurst_arr = cpu_results['hurst']
            adx_arr = cpu_results['adx']
            dmi_plus_arr = cpu_results['dmi_plus']
            dmi_minus_arr = cpu_results['dmi_minus']


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

        oscillation_coherence_arr = 1.0 / (1.0 + osc_std)   # (0, 1]
        np.nan_to_num(oscillation_coherence_arr, copy=False, nan=0.0)

        # Reconstruct result list
        results = []

        # Loop from rp to n
        for i in range(rp, n):
            # Map simple Lagrange zones
            z = z_scores[i]
            abs_z = abs(z)
            if abs_z < 1.0:
                lz = 'L1_STABLE'
            elif abs_z < 2.0:
                lz = 'CHAOS'
            elif z >= 2.0:
                lz = 'L2_ROCHE'
            else:
                lz = 'L3_ROCHE'

            # Map Archetypes to Flags
            structure_confirmed = bool(structural_drive[i])
            cascade_detected = bool(roche_snap[i])

            # Amplitudes (Real approximation)
            a0 = math.sqrt(prob0[i])
            a1 = math.sqrt(prob1[i])
            a2 = math.sqrt(prob2[i])

            # Trend Direction Logic (Restored)
            slope_strength = (abs(slope[i]) * rp) / (sigma[i] + 1e-6)
            if slope_strength > 1.0:
                trend_direction = 'UP' if slope[i] > 0 else 'DOWN'
            else:
                trend_direction = 'RANGE'

            state = ThreeBodyQuantumState(
                center_position=center[i],
                upper_singularity=center[i] + 2.0 * sigma[i],
                lower_singularity=center[i] - 2.0 * sigma[i],
                event_horizon_upper=center[i] + 3.0 * sigma[i],
                event_horizon_lower=center[i] - 3.0 * sigma[i],
                particle_position=prices[i],
                particle_velocity=velocity[i],
                z_score=z,
                F_reversion=-0.5 * z * sigma[i],
                F_upper_repulsion=0.0,
                F_lower_repulsion=0.0,
                F_momentum=momentum[i],
                F_net=force[i],
                amplitude_center=a0,
                amplitude_upper=a1,
                amplitude_lower=a2,
                P_at_center=prob0[i],
                P_near_upper=prob1[i],
                P_near_lower=prob2[i],
                entropy=entropy[i],
                coherence=coherence[i],
                pattern_maturity=0.0,
                momentum_strength=momentum[i],
                structure_confirmed=structure_confirmed,
                cascade_detected=cascade_detected,
                spin_inverted=False,
                lagrange_zone=lz,
                stability_index=1.0,
                tunnel_probability=0.0, escape_probability=0.0,
                barrier_height=0.0,
                pattern_type=str(pattern_types[i]),
                candlestick_pattern=str(candlestick_types[i]),
                trend_direction_15m=trend_direction, # Restored logic
                hurst_exponent=hurst_arr[i],
                adx_strength=adx_arr[i], dmi_plus=dmi_plus_arr[i], dmi_minus=dmi_minus_arr[i],
                sigma_fractal=sigma[i],
                term_pid=float(term_pid_arr[i]),
                oscillation_coherence=float(oscillation_coherence_arr[i]),
                lyapunov_exponent=0.0,
                market_regime='STABLE',
                timestamp=timestamps[i]
            )

            results.append({
                'bar_idx': i,
                'state': state,
                'price': prices[i],
                'structure_ok': lz in ('L2_ROCHE', 'L3_ROCHE')
            })

        return results
