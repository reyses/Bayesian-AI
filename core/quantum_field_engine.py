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

from core.three_body_state import ThreeBodyQuantumState
from core.risk_engine import QuantumRiskEngine
from core.pattern_utils import (
    PATTERN_NONE, PATTERN_COMPRESSION, PATTERN_WEDGE, PATTERN_BREAKDOWN,
    detect_geometric_pattern, detect_candlestick_pattern,
    detect_geometric_patterns_vectorized, detect_candlestick_patterns_vectorized
)

# Core CUDA Physics
try:
    from core.cuda_physics import compute_physics_kernel, detect_archetype_kernel
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
HURST_WINDOW = 100
ADX_LENGTH = 14

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

class QuantumFieldEngine:
    """
    Unified field calculator — GPU-accelerated when CUDA available
    - Nightmare Protocol (gravity wells, O-U process)
    - Three-body dynamics (Lagrange points, tidal forces)
    - Quantum mechanics (superposition, tunneling)
    """

    def __init__(self, regression_period: int = 21):
        self.regression_period = regression_period
        self.SIGMA_ROCHE_MULTIPLIER = 2.0
        self.SIGMA_EVENT_MULTIPLIER = 3.0
        self.TIDAL_FORCE_EXPONENT = 2.0

        # Historical residuals for fat-tail sigma calculation (rolling window)
        # Using 500 bars (~2 hours at 15s) to estimate distribution
        self.residual_history = []
        self.residual_window = 500

        # === GPU SETUP ===
        self.use_gpu = False
        if cuda.is_available() and CUDA_PHYSICS_AVAILABLE:
            self.use_gpu = True

        # Keep Torch device for legacy compatibility if needed, but primary compute is Numba
        if TORCH_AVAILABLE and torch.cuda.is_available():
            self.device = torch.device('cuda')
        elif TORCH_AVAILABLE:
            self.device = torch.device('cpu')
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

    def _calculate_wave_function(self, z, sigma, momentum_strength):
        """
        Computes quantum wave function components (CPU version).
        Helper for unit tests and CPU fallback.
        """
        # Energies
        E0 = - (z**2) / 2.0
        E1 = - ((z - 2.0)**2) / 2.0
        E2 = - ((z + 2.0)**2) / 2.0

        max_E = max(E0, E1, E2)
        p0 = math.exp(E0 - max_E)
        p1 = math.exp(E1 - max_E)
        p2 = math.exp(E2 - max_E)

        total_p = p0 + p1 + p2
        if total_p == 0:
            return {'P0': 0.33, 'P1': 0.33, 'P2': 0.33}

        return {
            'P0': p0 / total_p,
            'P1': p1 / total_p,
            'P2': p2 / total_p
        }

    def _calculate_physics_cpu(self, prices, volumes, timestamps):
        """
        CPU implementation of physics kernel (numpy-based).
        Matches CUDA kernel logic but sequential/vectorized where possible.
        """
        n = len(prices)
        rp = self.regression_period

        center = np.zeros(n)
        sigma = np.zeros(n)
        slope = np.zeros(n)
        z_scores = np.zeros(n)
        velocity = np.zeros(n)
        force = np.zeros(n)
        momentum = np.zeros(n)
        coherence = np.ones(n)
        entropy = np.zeros(n)
        prob0 = np.ones(n)
        prob1 = np.zeros(n)
        prob2 = np.zeros(n)
        roche_snap = np.zeros(n, dtype=bool)
        structural_drive = np.zeros(n, dtype=bool)

        # Precompute X
        x = np.arange(rp)
        mean_x = np.mean(x)
        sum_xx = np.sum(x**2)
        denom = sum_xx - len(x) * mean_x**2

        # Iterate
        for i in range(rp - 1, n):
            # Window: [i - rp + 1 : i + 1] (Coincident)
            window_prices = prices[i - rp + 1 : i + 1]

            # Regression
            mean_y = np.mean(window_prices)
            sum_xy = np.sum(x * window_prices)

            slope_val = 0.0
            if abs(denom) > 1e-9:
                slope_val = (sum_xy - len(x) * mean_x * mean_y) / denom

            # Center at end of window
            center_val = mean_y + slope_val * (x[-1] - mean_x)

            center[i] = center_val
            slope[i] = slope_val

            # Sigma (RSS)
            residuals = window_prices - (slope_val * x + (mean_y - slope_val * mean_x))
            rss = np.sum(residuals**2)
            sigma_val = 0.0
            if rp > 2:
                sigma_val = math.sqrt(rss / (rp - 2))
            if sigma_val < 1e-6:
                sigma_val = 1e-6
            sigma[i] = sigma_val

            # Z-Score
            z = (prices[i] - center_val) / sigma_val
            z_scores[i] = z

            # Velocity
            vel = prices[i] - prices[i-1] if i > 0 else 0.0
            velocity[i] = vel

            # Momentum
            mom = (vel * volumes[i]) / sigma_val
            momentum[i] = mom

            # Forces
            F_gravity = -DEFAULT_GRAVITY_THETA * (z * sigma_val)

            upper_sing = center_val + self.SIGMA_ROCHE_MULTIPLIER * sigma_val
            lower_sing = center_val - self.SIGMA_ROCHE_MULTIPLIER * sigma_val

            dist_upper = abs(prices[i] - upper_sing) / sigma_val
            dist_lower = abs(prices[i] - lower_sing) / sigma_val

            F_upper = 0.0
            if z > 0:
                F_upper = 1.0 / (dist_upper**3 + 0.01)
                if F_upper > 100.0: F_upper = 100.0

            F_lower = 0.0
            if z < 0:
                F_lower = 1.0 / (dist_lower**3 + 0.01)
                if F_lower > 100.0: F_lower = 100.0

            repulsion = -F_upper if z > 0 else F_lower
            force[i] = F_gravity + mom + repulsion

            # Wave Function
            wf = self._calculate_wave_function(z, sigma_val, mom)
            prob0[i] = wf['P0']
            prob1[i] = wf['P1']
            prob2[i] = wf['P2']

            eps = 1e-10
            ent = - (prob0[i] * math.log(prob0[i] + eps) +
                     prob1[i] * math.log(prob1[i] + eps) +
                     prob2[i] * math.log(prob2[i] + eps))
            entropy[i] = ent
            coherence[i] = ent / 1.09861228867

            # Archetypes (using inline thresholds for now, could be constants)
            if abs(z) > 2.0 and abs(vel) > VELOCITY_CASCADE_THRESHOLD:
                 roche_snap[i] = True

            if abs(mom) > 5.0 and coherence[i] < 0.3:
                 structural_drive[i] = True

        return (center, sigma, slope, z_scores, velocity, force, momentum,
                coherence, entropy, prob0, prob1, prob2, roche_snap, structural_drive)


    # ═══════════════════════════════════════════════════════════════════════
    # VECTORIZED BATCH COMPUTATION (processes all bars at once)
    # ═══════════════════════════════════════════════════════════════════════

    def batch_compute_states(self, day_data: pd.DataFrame, use_cuda: bool = True, params: dict = None) -> list:
        """
        Compute ALL ThreeBodyQuantumState objects for a day using Fused CUDA Physics Kernels.
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

        # Extract timestamps early for CPU fallback if needed
        timestamps = np.zeros(n, dtype=np.float64)
        if 'timestamp' in day_data.columns:
            ts_col = day_data['timestamp']
            if pd.api.types.is_datetime64_any_dtype(ts_col):
                timestamps = ts_col.astype('int64').values / 1e9
            else:
                timestamps = ts_col.values.astype(np.float64)
        elif isinstance(day_data.index, pd.DatetimeIndex):
            timestamps = day_data.index.astype('int64').values / 1e9

        if use_cuda and self.use_gpu:
            # Output arrays (allocated on GPU directly or mapped)
            d_prices = cuda.to_device(prices)
            d_volumes = cuda.to_device(volumes)

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

            # Kernel Launch Configuration
            threads_per_block = 256
            blocks_per_grid = (n + (threads_per_block - 1)) // threads_per_block

            # Precompute regression constants
            sum_x = 0.0
            sum_xx = 0.0
            for _k in range(rp):
                sum_x += float(_k)
                sum_xx += float(_k * _k)

            mean_x = sum_x / rp
            denom = sum_xx - (sum_x * sum_x) / rp
            inv_reg_period = 1.0 / rp
            inv_denom = 0.0
            if abs(denom) > 1e-9:
                inv_denom = 1.0 / denom

            # 1. Physics Kernel
            compute_physics_kernel[blocks_per_grid, threads_per_block](
                d_prices, d_volumes,
                d_center, d_sigma, d_slope,
                d_z, d_velocity, d_force, d_momentum,
                d_coherence, d_entropy,
                d_prob0, d_prob1, d_prob2,
                rp, mean_x, inv_reg_period, inv_denom, denom
            )

            # 2. Archetype Kernel
            detect_archetype_kernel[blocks_per_grid, threads_per_block](
                d_z, d_velocity, d_momentum, d_coherence,
                d_roche, d_drive
            )

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
        else:
            # CPU Fallback
            (center, sigma, slope, z_scores, velocity, force, momentum,
             coherence, entropy, prob0, prob1, prob2, roche_snap, structural_drive) = \
                self._calculate_physics_cpu(prices, volumes, timestamps)

        # Unified pattern detection (Geometric/Candlestick)
        if 'high' in day_data.columns:
            highs = day_data['high'].values.astype(np.float64)
            lows = day_data['low'].values.astype(np.float64)
            opens = day_data['open'].values.astype(np.float64)
        else:
            highs = prices
            lows = prices
            opens = prices

        pattern_types, candlestick_types = self._detect_patterns_unified(opens, highs, lows, prices)

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
                hurst_exponent=0.5,
                adx_strength=0.0, dmi_plus=0.0, dmi_minus=0.0,
                sigma_fractal=sigma[i],
                term_pid=0.0,
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
