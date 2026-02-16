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
        if not cuda.is_available():
            raise RuntimeError("CUDA accelerator is mandatory but not available on this system.")

        if not CUDA_PHYSICS_AVAILABLE:
             raise RuntimeError("core.cuda_physics module is missing.")

        self.use_gpu = True

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
        results = self.batch_compute_states(df_macro, use_cuda=True)
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
