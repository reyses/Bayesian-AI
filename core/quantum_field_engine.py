"""
Quantum Field Calculator
Computes three-body gravitational fields + quantum wave function
Integrates Nightmare Protocol gravity calculations
"""
import numpy as np
import pandas as pd
import math
from scipy.stats import linregress
import numba
from numba import cuda
from tqdm import tqdm

from core.three_body_state import ThreeBodyQuantumState
from core.risk_engine import QuantumRiskEngine
from core.pattern_utils import (
    PATTERN_NONE, PATTERN_COMPRESSION, PATTERN_WEDGE, PATTERN_BREAKDOWN,
    detect_geometric_pattern, detect_candlestick_pattern,
    detect_geometric_patterns_vectorized, detect_candlestick_patterns_vectorized
)

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

@cuda.jit
def calculate_z_score_kernel(prices, z_scores, means, sigmas, window):
    """
    Compute rolling mean, standard deviation, and Z-score for each element.
    z = (price - mean) / std

    Each thread computes one element.
    O(N*W) complexity - acceptable for small windows (e.g. 20-50).
    """
    i = cuda.grid(1)
    if i < prices.shape[0]:
        if i < window - 1: # Index is 0-based. i=window-1 is the first index with 'window' items (0..window-1)
            # Not enough data for full window
            z_scores[i] = 0.0
            means[i] = prices[i]
            sigmas[i] = 1.0 # Avoid div/0
        else:
            # Compute mean
            sum_val = 0.0
            for j in range(window):
                sum_val += prices[i - j]
            mean = sum_val / window
            means[i] = mean

            # Compute std
            sum_sq_diff = 0.0
            for j in range(window):
                diff = prices[i - j] - mean
                sum_sq_diff += diff * diff

            # Sample std dev (N-1)
            if window > 1:
                std = math.sqrt(sum_sq_diff / (window - 1))
            else:
                std = 0.0

            # Avoid div/0
            if std < 1e-9:
                std = 1e-9

            sigmas[i] = std
            z_scores[i] = (prices[i] - mean) / std

@cuda.jit
def calculate_force_field_kernel(z_scores, forces):
    """
    Compute tidal force field: Force = Z^2 / 9
    """
    i = cuda.grid(1)
    if i < z_scores.shape[0]:
        z = z_scores[i]
        forces[i] = (z * z) / 9.0

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
        
        Process:
        1. Calculate Attractors (Center Mass)
        2. Calculate Fractal Diffusion (Sigma)
        3. Calculate Force Vectors (Gravity, PID, Momentum)
        4. Calculate Lyapunov Stability
        5. Compute Wave Function & Tunneling
        """
        
        if len(df_macro) < self.regression_period:
            return ThreeBodyQuantumState.null_state()

        # Extract physics parameters (with defaults)
        params = params or {}
        kp = params.get('pid_kp', DEFAULT_PID_KP)
        ki = params.get('pid_ki', DEFAULT_PID_KI)
        kd = params.get('pid_kd', DEFAULT_PID_KD)
        theta = params.get('gravity_theta', DEFAULT_GRAVITY_THETA)
        
        # 1. ATTRACTORS (Center Mass)
        center, reg_sigma, slope, residuals = self._calculate_center_mass(df_macro)
        
        # 2. FRACTAL & TREND INDICATORS (Needed for Fractal Diffusion)
        if HURST_AVAILABLE and len(df_micro) >= HURST_WINDOW:
            hurst_series = df_micro['close'].iloc[-HURST_WINDOW:]
            try:
                H, c, _ = compute_Hc(hurst_series, kind='price', simplified=True)
                hurst_val = H
            except Exception:
                hurst_val = 0.5
        else:
            hurst_val = 0.5

        # 3. FRACTAL DIFFUSION (Volatility)
        # σ(v, τ) = σ_base * (v_micro / v_macro)^H
        v_micro = abs(tick_velocity)
        # Macro velocity: Mean absolute change in macro window
        macro_diffs = df_macro['close'].diff().abs()
        v_macro = macro_diffs.mean() if not macro_diffs.empty else 1.0
        v_macro = max(v_macro, 1e-6)

        # Base sigma is the regression residual std dev (historical volatility)
        sigma_base = reg_sigma

        # Apply Nightmare Formula for Fractal Sigma
        velocity_ratio = v_micro / v_macro if v_macro > 0 else 1.0
        # Cap ratio to prevent explosion
        velocity_ratio = min(max(velocity_ratio, 0.1), 10.0)

        sigma_fractal = sigma_base * (velocity_ratio ** hurst_val)

        # Use the computed fractal sigma as the system sigma
        sigma = max(sigma_fractal, 1e-6)

        # Trend Direction (15m slope)
        slope_strength = (slope * len(df_macro)) / (sigma + 1e-6)
        if slope_strength > 1.0:
            trend_direction = 'UP' if slope > 0 else 'DOWN'
        else:
            trend_direction = 'RANGE'

        # Bodies 2 & 3: Singularities (using fractal sigma)
        upper_sing = center + self.SIGMA_ROCHE_MULTIPLIER * sigma
        lower_sing = center - self.SIGMA_ROCHE_MULTIPLIER * sigma
        upper_event = center + self.SIGMA_EVENT_MULTIPLIER * sigma
        lower_event = center - self.SIGMA_EVENT_MULTIPLIER * sigma
        
        # Particle state
        z_score = (current_price - center) / sigma if sigma > 0 else 0.0
        
        # 4. FORCE VECTORS (Gravity, PID, Momentum)
        # F_gravity (OU) = theta * (mean - price)
        # F_pid = Kp*e + Ki*int(e) + Kd*de/dt
        # F_net = F_gravity + F_pid + F_momentum

        # Calculate PID components
        # Error = Price - Mean. (So restoring force is negative of error)
        # But wait, OU definition in text: F = theta * (mean - price).
        # if price > mean, mean - price is negative. Force is down. Correct.
        error = current_price - center

        # Need history for Integral and Derivative. Use df_macro residuals.
        # residuals = price - center_line (approx error history)
        pid_force = self._calculate_pid_force(residuals, kp, ki, kd)

        forces = self._calculate_force_fields(
            current_price, center, upper_sing, lower_sing,
            z_score, sigma, current_volume, tick_velocity, pid_force, theta
        )
        
        # 5. LYAPUNOV STABILITY
        # Estimate from Z-score divergence
        # Need recent Z-history. We can approximate using recent df_micro prices against current center/sigma
        # or just track it. For now, use simple heuristic: expanding vs decaying Z.
        if len(df_micro) >= 3:
            recent_prices = df_micro['close'].iloc[-3:].values
            recent_z = (recent_prices - center) / sigma
            # Divergence: |Z_t| - |Z_{t-1}|
            delta_z = np.abs(recent_z[-1]) - np.abs(recent_z[-2])
            lyapunov = delta_z # Simple proxy: > 0 expanding, < 0 decaying
        else:
            lyapunov = 0.0

        market_regime = 'CHAOTIC' if lyapunov > 0 else 'STABLE'

        # Wave function
        quantum = self._calculate_wave_function(z_score, forces['F_net'], forces['F_momentum'])
        
        # Measurements
        measurements = self._check_measurements(df_micro, z_score, tick_velocity)
        
        # Tunneling (Heuristic by default)
        tunnel_prob, escape_prob, barrier = self._calculate_tunneling(
            z_score, forces['F_momentum'], forces['F_reversion']
        )

        # MONTE CARLO REFINEMENT (QuantLib)
        # Only run if in Roche Limit (critical zone) to save compute
        if abs(z_score) > 2.0:
            try:
                # Use initialized parameters (theta, horizon)
                mc_tunnel, mc_escape = self.risk_engine.calculate_probabilities(
                    price=current_price,
                    center=center,
                    sigma=sigma
                )
                # Blend MC with Heuristic (50/50) or Replace?
                # Let's average them for robustness
                tunnel_prob = (tunnel_prob + mc_tunnel) / 2.0
                escape_prob = (escape_prob + mc_escape) / 2.0
            except Exception as e:
                # Consider logging the exception, e.g., logging.warning(f"Risk engine failed: {e}")
                pass # Fallback to heuristic
        
        # Lagrange
        lagrange_zone, stability = self._classify_lagrange(z_score, forces['F_net'])
        
        # Build context args
        context_args = {}
        if context:
            # Map dictionary keys to dataclass fields
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

        # Apply trend direction from calculation
        context_args['trend_direction_15m'] = trend_direction

        # ADX/DMI (Trend Strength) - using df_macro (15m) for robust trend
        if PANDAS_TA_AVAILABLE and len(df_macro) >= ADX_LENGTH:
            try:
                # pandas_ta requires DataFrame with high, low, close
                # df_macro might be missing high/low if just close prices passed
                if 'high' in df_macro.columns and 'low' in df_macro.columns:
                    adx_df = df_macro.ta.adx(length=ADX_LENGTH)
                    if adx_df is not None and not adx_df.empty:
                        # Columns are usually ADX_14, DMP_14, DMN_14
                        # Get last row
                        last_row = adx_df.iloc[-1]
                        adx_val = last_row.iloc[0] # ADX is usually first
                        dmp_val = last_row.iloc[1] # DMP
                        dmn_val = last_row.iloc[2] # DMN
                    else:
                        adx_val, dmp_val, dmn_val = 0.0, 0.0, 0.0
                else:
                    adx_val, dmp_val, dmn_val = 0.0, 0.0, 0.0
            except Exception:
                adx_val, dmp_val, dmn_val = 0.0, 0.0, 0.0
        else:
            adx_val, dmp_val, dmn_val = 0.0, 0.0, 0.0

        # Replace NaNs
        hurst_val = np.nan_to_num(hurst_val, nan=0.5)
        adx_val = np.nan_to_num(adx_val, nan=0.0)
        dmp_val = np.nan_to_num(dmp_val, nan=0.0)
        dmn_val = np.nan_to_num(dmn_val, nan=0.0)

        return ThreeBodyQuantumState(
            center_position=center,
            hurst_exponent=hurst_val,
            adx_strength=adx_val,
            dmi_plus=dmp_val,
            dmi_minus=dmn_val,
            upper_singularity=upper_sing,
            lower_singularity=lower_sing,
            event_horizon_upper=upper_event,
            event_horizon_lower=lower_event,
            particle_position=current_price,
            particle_velocity=tick_velocity,
            z_score=z_score,
            F_reversion=forces['F_reversion'],
            F_upper_repulsion=forces['F_upper_repulsion'],
            F_lower_repulsion=forces['F_lower_repulsion'],
            F_momentum=forces['F_momentum'],
            F_net=forces['F_net'],
            amplitude_center=quantum['a0'],
            amplitude_upper=quantum['a1'],
            amplitude_lower=quantum['a2'],
            P_at_center=quantum['P0'],
            P_near_upper=quantum['P1'],
            P_near_lower=quantum['P2'],
            entropy=quantum['entropy'],
            coherence=quantum['coherence'],
            pattern_maturity=measurements['pattern_maturity'],
            momentum_strength=np.nan_to_num(forces['F_momentum'] / (abs(forces['F_reversion']) + 1e-6)),
            structure_confirmed=measurements['structure_confirmed'],
            cascade_detected=measurements['cascade_detected'],
            spin_inverted=measurements['spin_inverted'],
            lagrange_zone=lagrange_zone,
            stability_index=stability,
            tunnel_probability=tunnel_prob,
            escape_probability=escape_prob,
            barrier_height=barrier,
            pattern_type=measurements['pattern_type'],
            candlestick_pattern=measurements['candlestick_pattern'],
            timestamp=df_macro.index[-1].timestamp() if hasattr(df_macro.index[-1], 'timestamp') else 0.0,
            sigma_fractal=sigma_fractal,
            term_pid=pid_force,
            lyapunov_exponent=lyapunov,
            market_regime=market_regime,
            **context_args
        )
    
    def _calculate_pid_force(self, residuals: np.ndarray, kp: float, ki: float, kd: float):
        """
        Calculate Algorithmic Control Force (PID)
        F_pid = Kp*e + Ki*int(e) + Kd*de/dt
        """
        if len(residuals) < 2:
            return 0.0

        e = residuals[-1] # Current error

        # Integral: Sum of recent errors (limit memory to regression window)
        e_integral = np.sum(residuals)

        # Derivative: Rate of change of error
        e_derivative = residuals[-1] - residuals[-2]

        # PID Formula (Restoring force opposes error)
        # If error > 0 (Price > Mean), F_pid should be negative
        f_pid = -(kp * e + ki * e_integral + kd * e_derivative)

        return f_pid

    def _calculate_center_mass(self, df: pd.DataFrame):
        """Linear regression for center star — uses residual std (not slope std_err)"""
        window = df.iloc[-self.regression_period:]
        y = window['close'].values
        x = np.arange(len(y))
        slope, intercept, _, _, _ = linregress(x, y)
        center = slope * x[-1] + intercept
        # Residual standard deviation (matches batch_compute_states)
        residuals = y - (slope * x + intercept)
        sigma = np.sqrt(np.sum(residuals ** 2) / (len(y) - 2))
        sigma = sigma if sigma > 0 else y.std()
        return center, sigma, slope, residuals
    
    def _calculate_force_fields(self, price, center, upper_sing, lower_sing, z_score, sigma, volume, velocity, pid_force, theta):
        """Compute competing gravitational forces (Unified Field Equation)"""
        F_gravity = -theta * (z_score * sigma)
        F_reversion = F_gravity

        dist_upper = abs(price - upper_sing) / sigma if sigma > 0 else 1.0
        dist_lower = abs(price - lower_sing) / sigma if sigma > 0 else 1.0
        F_upper_repulsion = min(1.0 / (dist_upper ** 3 + 0.01), 100.0) if z_score > 0 else 0.0
        F_lower_repulsion = min(1.0 / (dist_lower ** 3 + 0.01), 100.0) if z_score < 0 else 0.0
        
        safe_velocity = np.nan_to_num(velocity)
        safe_volume = np.nan_to_num(volume)
        F_momentum = safe_velocity * safe_volume / (sigma + 1e-6)
        F_pid = pid_force

        repulsion = -F_upper_repulsion if z_score > 0 else F_lower_repulsion
        F_net = F_gravity + F_momentum + F_pid + repulsion
        
        return {
            'F_reversion': F_reversion,
            'F_upper_repulsion': F_upper_repulsion,
            'F_lower_repulsion': F_lower_repulsion,
            'F_momentum': F_momentum,
            'F_net': F_net
        }
    
    def _calculate_wave_function(self, z_score, F_net, F_momentum):
        """Quantum superposition state — CUDA accelerated"""
        E0 = -z_score**2 / 2
        E1 = -(z_score - 2.0)**2 / 2
        E2 = -(z_score + 2.0)**2 / 2

        if self.use_gpu:
            energies = torch.tensor([E0, E1, E2], device=self.device, dtype=torch.float64)
            energies -= energies.max()
            probs = torch.exp(energies)
            probs /= probs.sum()
            eps = 1e-10
            entropy_val = -(probs * torch.log(probs + eps)).sum().item()
            P0, P1, P2 = probs[0].item(), probs[1].item(), probs[2].item()
        else:
            max_E = max(E0, E1, E2)
            P0 = np.exp(E0 - max_E)
            P1 = np.exp(E1 - max_E)
            P2 = np.exp(E2 - max_E)
            total = P0 + P1 + P2
            P0 /= total
            P1 /= total
            P2 /= total
            eps = 1e-10
            entropy_val = -(P0*np.log(P0+eps) + P1*np.log(P1+eps) + P2*np.log(P2+eps))

        coherence = entropy_val / np.log(3)
        phase = np.arctan2(F_momentum, F_net + 1e-6)
        a0 = np.sqrt(P0) * np.exp(1j * phase * 0)
        a1 = np.sqrt(P1) * np.exp(1j * phase * 1)
        a2 = np.sqrt(P2) * np.exp(1j * phase * -1)
        return {'a0': a0, 'a1': a1, 'a2': a2, 'P0': P0, 'P1': P1, 'P2': P2,
                'entropy': entropy_val, 'coherence': coherence}
    
    def _check_measurements(self, df_micro, z_score, velocity):
        """L8-L9 measurement operators"""
        if len(df_micro) < 20:
            return {
                'structure_confirmed': False,
                'cascade_detected': False,
                'spin_inverted': False,
                'pattern_maturity': 0.0,
                'pattern_type': PATTERN_NONE,
                'candlestick_pattern': 'NONE'
            }
        
        recent = df_micro.iloc[-20:]
        volume_spike = recent['volume'].iloc[-1] > recent['volume'].mean() * 1.2
        pattern_maturity = min((abs(z_score) - 2.0) / 1.0, 1.0) if abs(z_score) > 2.0 else 0.0
        structure_confirmed = volume_spike and pattern_maturity > 0.1
        velocity_cascade = abs(velocity) > VELOCITY_CASCADE_THRESHOLD
        
        if len(df_micro) >= 5:
            window = df_micro.iloc[-5:]
            h = window['high'].max() if 'high' in window.columns else window['close'].max()
            l = window['low'].min() if 'low' in window.columns else window['close'].min()
            rolling_range = h - l
            rolling_cascade = rolling_range > RANGE_CASCADE_THRESHOLD
        else:
            current_candle = df_micro.iloc[-1]
            h = current_candle.get('high', current_candle['close'])
            l = current_candle.get('low', current_candle['close'])
            rolling_cascade = (h - l) > RANGE_CASCADE_THRESHOLD

        cascade_detected = velocity_cascade or rolling_cascade
        current_candle = df_micro.iloc[-1]
        open_price = current_candle.get('open', current_candle['close'])

        if z_score > 2.0:
            spin_inverted = current_candle['close'] < open_price
        elif z_score < -2.0:
            spin_inverted = current_candle['close'] > open_price
        else:
            spin_inverted = False
        
        highs = df_micro['high'].values if 'high' in df_micro.columns else df_micro['close'].values
        lows = df_micro['low'].values if 'low' in df_micro.columns else df_micro['close'].values
        check_highs = highs[-20:]
        check_lows = lows[-20:]
        pattern_type = detect_geometric_pattern(check_highs, check_lows)

        opens = df_micro['open'].values if 'open' in df_micro.columns else df_micro['close'].values
        check_opens = opens[-20:]
        check_closes = df_micro['close'].values[-20:]
        candlestick_pattern = detect_candlestick_pattern(check_opens, check_highs, check_lows, check_closes)

        return {
            'structure_confirmed': structure_confirmed,
            'cascade_detected': cascade_detected,
            'spin_inverted': spin_inverted,
            'pattern_maturity': pattern_maturity,
            'pattern_type': pattern_type,
            'candlestick_pattern': candlestick_pattern
        }
    
    def _calculate_tunneling(self, z_score, F_momentum, F_reversion):
        """Quantum tunneling probabilities — CUDA accelerated"""
        barrier = abs(z_score) - 2.0
        if barrier < 0:
            return 0.5, 0.0, 0.0

        momentum_ratio = F_momentum / (F_reversion + 1e-6)

        if self.use_gpu:
            b = torch.tensor([barrier], device=self.device, dtype=torch.float64)
            tunnel_prob = torch.exp(-b * 2.0).item() * (1.0 - min(momentum_ratio, 0.9))
            escape_prob = momentum_ratio * torch.exp(-b * 0.5).item()
        else:
            tunnel_prob = np.exp(-barrier * 2.0) * (1.0 - min(momentum_ratio, 0.9))
            escape_prob = momentum_ratio * np.exp(-barrier * 0.5)

        total = tunnel_prob + escape_prob
        if total > 0:
            tunnel_prob /= total
            escape_prob /= total

        return tunnel_prob, escape_prob, barrier
    
    def _classify_lagrange(self, z_score, F_net):
        """Lagrange point classification"""
        if abs(z_score) < 1.0:
            return 'L1_STABLE', 1.0 - abs(z_score)
        elif 1.0 <= abs(z_score) < 2.0:
            return 'CHAOS', 0.5
        elif z_score >= 2.0:
            return 'L2_ROCHE', 0.1
        elif z_score <= -2.0:
            return 'L3_ROCHE', 0.1
        else:
            return 'UNKNOWN', 0.0

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
                # Fallback on error (e.g. CUDA context issues).
                # Consider logging 'e' if logging system available.
                pass

        # Default to CPU implementation for small datasets or if CUDA is unavailable/fails
        geo = detect_geometric_patterns_vectorized(highs, lows)
        cdl = detect_candlestick_patterns_vectorized(opens, highs, lows, closes)
        return geo, cdl


    # ═══════════════════════════════════════════════════════════════════════
    # VECTORIZED BATCH COMPUTATION (processes all bars at once)
    # ═══════════════════════════════════════════════════════════════════════

    def batch_compute_states(self, day_data: pd.DataFrame, use_cuda: bool = True, params: dict = None, progress_bar: bool = False, desc: str = "Computing States") -> list:
        """
        Compute ALL ThreeBodyQuantumState objects for a day using Numba CUDA kernels.
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

        # Output arrays
        z_scores = np.zeros(n, dtype=np.float64)
        means = np.zeros(n, dtype=np.float64)
        sigmas = np.zeros(n, dtype=np.float64)
        forces = np.zeros(n, dtype=np.float64)

        # GPU Allocation & Transfer
        d_prices = cuda.to_device(prices)
        d_z_scores = cuda.device_array(n, dtype=np.float64)
        d_means = cuda.device_array(n, dtype=np.float64)
        d_sigmas = cuda.device_array(n, dtype=np.float64)
        d_forces = cuda.device_array(n, dtype=np.float64)

        # Kernel Launch Configuration
        threads_per_block = 256
        blocks_per_grid = (n + (threads_per_block - 1)) // threads_per_block

        # Launch Z-Score Kernel
        calculate_z_score_kernel[blocks_per_grid, threads_per_block](
            d_prices, d_z_scores, d_means, d_sigmas, rp
        )

        # Launch Force Field Kernel
        calculate_force_field_kernel[blocks_per_grid, threads_per_block](
            d_z_scores, d_forces
        )

        # Copy back results
        d_z_scores.copy_to_host(z_scores)
        d_means.copy_to_host(means)
        d_sigmas.copy_to_host(sigmas)
        d_forces.copy_to_host(forces)

        # Unified pattern detection (CPU or CUDA)
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

        # Pre-compute tick velocity
        tick_velocity = np.zeros(n, dtype=np.float64)
        tick_velocity[1:] = prices[1:] - prices[:-1]

        # Loop from rp to n
        iterator = range(rp, n)
        if progress_bar:
            iterator = tqdm(iterator, desc=desc, leave=False, unit="bar")

        for i in iterator:
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

            # Simple measurements
            structure_confirmed = abs_z > 2.0 # simplified
            cascade_detected = False # simplified

            state = ThreeBodyQuantumState(
                center_position=means[i],
                upper_singularity=means[i] + 2.0 * sigmas[i],
                lower_singularity=means[i] - 2.0 * sigmas[i],
                event_horizon_upper=means[i] + 3.0 * sigmas[i],
                event_horizon_lower=means[i] - 3.0 * sigmas[i],
                particle_position=prices[i],
                particle_velocity=tick_velocity[i],
                z_score=z,
                F_reversion=-0.5 * z * sigmas[i], # Fallback gravity
                F_upper_repulsion=0.0,
                F_lower_repulsion=0.0,
                F_momentum=0.0,
                F_net=forces[i], # Using the Kernel computed force
                # Fill rest with defaults/placeholders
                amplitude_center=1.0, amplitude_upper=0.0, amplitude_lower=0.0,
                P_at_center=1.0, P_near_upper=0.0, P_near_lower=0.0,
                entropy=0.0, coherence=1.0,
                pattern_maturity=0.0, momentum_strength=0.0,
                structure_confirmed=structure_confirmed,
                cascade_detected=cascade_detected,
                spin_inverted=False,
                lagrange_zone=lz,
                stability_index=1.0,
                tunnel_probability=0.0, escape_probability=0.0,
                barrier_height=0.0,
                pattern_type=str(pattern_types[i]),
                candlestick_pattern=str(candlestick_types[i]),
                trend_direction_15m='RANGE',
                hurst_exponent=0.5,
                adx_strength=0.0, dmi_plus=0.0, dmi_minus=0.0,
                sigma_fractal=sigmas[i],
                term_pid=0.0,
                lyapunov_exponent=0.0,
                market_regime='STABLE'
            )

            results.append({
                'bar_idx': i,
                'state': state,
                'price': prices[i],
                'structure_ok': lz in ('L2_ROCHE', 'L3_ROCHE')
            })

        return results
