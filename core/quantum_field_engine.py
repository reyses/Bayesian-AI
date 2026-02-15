"""
Quantum Field Calculator
Computes three-body gravitational fields + quantum wave function
Integrates Nightmare Protocol gravity calculations
"""
import numpy as np
import pandas as pd
from scipy.stats import linregress
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
except (ImportError, ValueError):
    # ValueError can occur if matplotlib is present but spec not set correctly
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
    CUDA_AVAILABLE = torch.cuda.is_available()
except ImportError:
    TORCH_AVAILABLE = False
    CUDA_AVAILABLE = False

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
        if TORCH_AVAILABLE and CUDA_AVAILABLE:
            self.device = torch.device('cuda')
            self.use_gpu = True
        elif TORCH_AVAILABLE:
            self.device = torch.device('cpu')
            self.use_gpu = False # Default to false on CPU unless specifically testing
        else:
            self.device = None
            self.use_gpu = False

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
        # Default to True for single-state calculation to preserve precision
        use_hurst = params.get('use_hurst', True)
        if use_hurst and HURST_AVAILABLE and len(df_micro) >= HURST_WINDOW:
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

    def _rolling_regression_torch(self, prices: torch.Tensor, window_size: int):
        """
        Vectorized rolling linear regression using PyTorch convolutions.
        Returns: centers, slopes, sigmas (std dev of residuals)
        """
        # prices: (N,) tensor
        N = prices.shape[0]
        if N < window_size:
            return None, None, None

        # Prepare kernels (use float64 for precision)
        # x = [0, 1, ..., W-1]
        x = torch.arange(window_size, device=self.device, dtype=torch.float64)
        sum_x = x.sum()
        sum_x_sq = (x**2).sum()
        mean_x = x.mean()
        
        # Slope formula denominator: N * sum(x^2) - (sum(x))^2
        denom = window_size * sum_x_sq - sum_x**2
        
        # Convolutions for sums
        # prices view as (1, 1, N)
        prices_reshaped = prices.view(1, 1, -1)
        
        # Kernel for sum_y: ones
        ones_kernel = torch.ones((1, 1, window_size), device=self.device, dtype=torch.float64)
        
        # Kernel for sum_xy: x (reversed? conv1d computes correlation sum(I[t+j]*K[j]))
        # We want sum(y[t+j] * x[j]). So Kernel should be x.
        xy_kernel = x.view(1, 1, window_size)
        
        sum_y = F.conv1d(prices_reshaped, ones_kernel).view(-1)
        sum_xy = F.conv1d(prices_reshaped, xy_kernel).view(-1)
        
        # Slope
        # num = N * sum_xy - sum_x * sum_y
        num = window_size * sum_xy - sum_x * sum_y
        slopes = num / (denom + 1e-10)
        
        # Intercept
        # c = mean_y - m * mean_x = (sum_y / N) - m * mean_x
        intercepts = (sum_y / window_size) - slopes * mean_x
        
        # Centers (at end of window): y = mx + c where x is (W-1)
        centers = slopes * (window_size - 1) + intercepts
        
        # Residuals Sigma
        # sum((y - (mx+c))^2) = sum(y^2) - 2m sum(xy) - 2c sum(y) + m^2 sum(x^2) + 2mc sum(x) + N c^2
        
        prices_sq = prices ** 2
        sum_y_sq = F.conv1d(prices_sq.view(1, 1, -1), ones_kernel).view(-1)
        
        ssr = sum_y_sq - 2 * slopes * sum_xy - 2 * intercepts * sum_y + \
              (slopes**2) * sum_x_sq + 2 * slopes * intercepts * sum_x + \
              window_size * (intercepts**2)
              
        ssr = torch.clamp(ssr, min=0.0)
        sigmas = torch.sqrt(ssr / (window_size - 2))
        
        return centers, slopes, sigmas

    def _batch_compute_states_gpu(self, day_data: pd.DataFrame, params: dict) -> list:
        """
        GPU-accelerated batch state computation using Torch.
        """
        params = params or {}
        # Move data to GPU
        # Check columns
        if 'price' in day_data.columns:
            prices_np = day_data['price'].values
        else:
            prices_np = day_data['close'].values
        
        if 'volume' in day_data.columns:
            volumes_np = day_data['volume'].values
        else:
            volumes_np = np.zeros(len(prices_np))

        # We also need highs/lows for patterns and cascades
        if 'high' in day_data.columns:
            highs_np = day_data['high'].values
        else:
            highs_np = prices_np
        if 'low' in day_data.columns:
            lows_np = day_data['low'].values
        else:
            lows_np = prices_np
        if 'open' in day_data.columns:
            opens_np = day_data['open'].values
        else:
            opens_np = prices_np

        # Convert to Tensor
        prices = torch.tensor(prices_np, device=self.device, dtype=torch.float64)
        volumes = torch.tensor(volumes_np, device=self.device, dtype=torch.float64)
        highs = torch.tensor(highs_np, device=self.device, dtype=torch.float64)
        lows = torch.tensor(lows_np, device=self.device, dtype=torch.float64)
        opens = torch.tensor(opens_np, device=self.device, dtype=torch.float64)
        
        n = prices.shape[0]
        rp = self.regression_period
        num_bars = n - rp
        
        if num_bars <= 0:
            return []
            
        # 1. Rolling Regression (Centers, Slopes, Sigmas)
        centers, slopes, reg_sigmas = self._rolling_regression_torch(prices, rp)
        
        # Note: _rolling_regression_torch returns vectors of length N - rp + 1.
        # Output[0] corresponds to window prices[0..rp-1].
        # In batch_compute_states (CPU), the first bar we compute is at index rp.
        # It uses the window ending at rp (indices 1..rp inclusive, or 1..21).
        # So we need output index 1.
        if centers is not None:
            centers = centers[1:]
            slopes = slopes[1:]
            reg_sigmas = reg_sigmas[1:]
        
        # 2. Fractal Sigma & Robust Sigma
        # Current Residuals: Price - Center (at end of window)
        bar_prices = prices[rp:]
        current_residuals = bar_prices - centers
        abs_res = torch.abs(current_residuals)
        
        # Rolling Robust Sigma (Window 500, Percentile 84)
        # We need a robust sigma based on the LAST 500 residuals.
        # Use Unfold.
        if len(abs_res) >= 500:
            # Unfold creates view (N-499, 500).
            # Output[i] corresponds to window ending at i+499.
            # So robust_vals[i] corresponds to bar at i+499.
            windows = abs_res.unfold(0, 500, 1)
            robust_vals = torch.quantile(windows, 0.84, dim=1)
            
            # Pad beginning with reg_sigmas (fallback for startup)
            # Create tensor of same shape as reg_sigmas
            robust_sigmas = reg_sigmas.clone()
            robust_sigmas[499:] = robust_vals
            
            # Combine: max(robust, reg)
            sigmas_base = torch.max(robust_sigmas, reg_sigmas)
        else:
            sigmas_base = reg_sigmas

        # Need v_micro (tick velocity) and v_macro (rolling mean of abs diffs)
        # Tick velocity: diff(prices)
        # We need this aligned with the bars we are computing (rp to n-1)
        
        # Full velocity vector
        zeros = torch.zeros(1, device=self.device, dtype=torch.float64)
        price_diffs = torch.cat((zeros, prices[1:] - prices[:-1]))
        tick_velocity = price_diffs[rp:] # Aligned with computed bars
        v_micro = torch.abs(tick_velocity)
        
        # v_macro: rolling mean of abs(diff(close))
        abs_diffs = torch.abs(price_diffs)
        # Rolling mean of abs_diffs. Use conv1d with ones kernel.
        ones_kernel = torch.ones((1, 1, rp), device=self.device, dtype=torch.float64) / rp
        # Convolution gives result for window 0..rp-1 at index 0.
        # We want result for window 1..rp at index 0 (corresponding to bar rp).
        # So we slice [1:]
        v_macro_full = F.conv1d(abs_diffs.view(1, 1, -1), ones_kernel).view(-1)
        v_macro = v_macro_full[1:] # Aligned with centers
        
        # Hurst (Approximate or use default 0.5 for GPU speedup)
        # Vectorizing Hurst is hard. Let's assume 0.5 or allow CPU precalc passed in.
        # For now, default 0.5.
        hurst_val = 0.5
        
        # Fractal Sigma
        velocity_ratios = v_micro / (v_macro + 1e-9)
        velocity_ratios = torch.clamp(velocity_ratios, 0.1, 10.0)
        
        fractal_sigmas = sigmas_base * (velocity_ratios ** hurst_val)
        sigmas = torch.max(fractal_sigmas, torch.tensor(1e-6, device=self.device, dtype=torch.float64))
        
        # 3. Z-Score & Forces
        z_scores = (bar_prices - centers) / sigmas
        
        # PID Forces
        # Errors = Price - Center
        errors = bar_prices - centers
        
        # Derivative: errors[i] - errors[i-1]
        # Prepend 0 for first element
        zeros_err = torch.zeros(1, device=self.device, dtype=torch.float64)
        prev_errors = torch.cat((zeros_err, errors[:-1]))
        e_deriv = errors - prev_errors
        
        # Integral: rolling sum of errors over window rp
        # Use conv1d again
        ones_kernel_sum = torch.ones((1, 1, rp), device=self.device, dtype=torch.float64)
        # Pad errors at start to allow full window
        padding = torch.zeros(rp - 1, device=self.device, dtype=torch.float64)
        errors_padded = torch.cat((padding, errors))
        e_integ = F.conv1d(errors_padded.view(1, 1, -1), ones_kernel_sum).view(-1)
        
        kp = params.get('pid_kp', DEFAULT_PID_KP)
        ki = params.get('pid_ki', DEFAULT_PID_KI)
        kd = params.get('pid_kd', DEFAULT_PID_KD)
        theta = params.get('gravity_theta', DEFAULT_GRAVITY_THETA)
        
        F_pid = -(kp * errors + ki * e_integ + kd * e_deriv)
        
        # Gravity
        F_gravity = -theta * errors # -theta * z * sigma
        
        # Repulsion
        upper_sing = centers + self.SIGMA_ROCHE_MULTIPLIER * sigmas
        lower_sing = centers - self.SIGMA_ROCHE_MULTIPLIER * sigmas
        upper_event = centers + self.SIGMA_EVENT_MULTIPLIER * sigmas
        lower_event = centers - self.SIGMA_EVENT_MULTIPLIER * sigmas
        
        dist_upper = torch.abs(bar_prices - upper_sing) / sigmas
        dist_lower = torch.abs(bar_prices - lower_sing) / sigmas
        
        F_upper_raw = torch.clamp(1.0 / (dist_upper ** 3 + 0.01), max=100.0)
        F_lower_raw = torch.clamp(1.0 / (dist_lower ** 3 + 0.01), max=100.0)
        
        F_upper_repulsion = torch.where(z_scores > 0, F_upper_raw, torch.tensor(0.0, device=self.device, dtype=torch.float64))
        F_lower_repulsion = torch.where(z_scores < 0, F_lower_raw, torch.tensor(0.0, device=self.device, dtype=torch.float64))
        
        # Momentum
        bar_volumes = volumes[rp:]
        F_momentum = tick_velocity * bar_volumes / (sigmas + 1e-6)
        
        # Net Force
        repulsion = torch.where(z_scores > 0, -F_upper_repulsion, F_lower_repulsion)
        F_net = F_gravity + F_momentum + F_pid + repulsion
        F_reversion = F_gravity
        
        # 4. Wave Function
        E0 = -z_scores**2 / 2
        E1 = -(z_scores - 2.0)**2 / 2
        E2 = -(z_scores + 2.0)**2 / 2
        
        # LogSumExp trick for stability? Or just exp
        # max_E per bar
        max_E, _ = torch.max(torch.stack([E0, E1, E2]), dim=0)
        P0 = torch.exp(E0 - max_E)
        P1 = torch.exp(E1 - max_E)
        P2 = torch.exp(E2 - max_E)
        total_P = P0 + P1 + P2
        P0 /= total_P
        P1 /= total_P
        P2 /= total_P
        
        eps = 1e-10
        entropy = -(P0 * torch.log(P0 + eps) + P1 * torch.log(P1 + eps) + P2 * torch.log(P2 + eps))
        coherence = entropy / np.log(3)
        
        # Amplitudes (Complex numbers not supported well in all torch ops, keeping components)
        # We need numpy for complex usually, but we can compute phase and pull back
        phase = torch.atan2(F_momentum, F_net + 1e-6)
        
        # 5. Measurements (Structure, Cascade)
        # Volume Spike: bar_volume > rolling_mean * 1.2
        # Rolling mean of volume (window 20)
        vol_kernel = torch.ones((1, 1, 20), device=self.device, dtype=torch.float64) / 20.0
        # Padding for volume
        vol_padding = torch.zeros(19, device=self.device, dtype=torch.float64)
        vol_padded = torch.cat((vol_padding, bar_volumes))
        vol_rolling_mean = F.conv1d(vol_padded.view(1, 1, -1), vol_kernel).view(-1)
        
        volume_spike = bar_volumes > vol_rolling_mean * 1.2
        
        pattern_maturity = torch.where(
            torch.abs(z_scores) > 2.0,
            torch.clamp((torch.abs(z_scores) - 2.0), max=1.0),
            torch.tensor(0.0, device=self.device, dtype=torch.float64)
        )
        
        structure_confirmed = volume_spike & (pattern_maturity > 0.1)
        
        velocity_cascade = torch.abs(tick_velocity) > VELOCITY_CASCADE_THRESHOLD
        
        # Range Cascade: High - Low > Threshold
        bar_highs = highs[rp:]
        bar_lows = lows[rp:]
        range_cascade = (bar_highs - bar_lows) > RANGE_CASCADE_THRESHOLD
        cascade_detected = velocity_cascade | range_cascade
        
        # Spin Inverted
        bar_opens = opens[rp:]
        spin_inverted = torch.where(
            z_scores > 2.0,
            bar_prices < bar_opens,
            torch.where(z_scores < -2.0, bar_prices > bar_opens, torch.tensor(False, device=self.device))
        )
        
        # Tunneling
        barrier = torch.abs(z_scores) - 2.0
        barrier_pos = barrier > 0
        momentum_ratio = F_momentum / (F_reversion + 1e-6)
        
        tunnel_prob = torch.where(
            barrier_pos,
            torch.exp(-barrier * 2.0) * (1.0 - torch.clamp(momentum_ratio, max=0.9)),
            torch.tensor(0.5, device=self.device, dtype=torch.float64)
        )
        escape_prob = torch.where(
            barrier_pos,
            momentum_ratio * torch.exp(-barrier * 0.5),
            torch.tensor(0.0, device=self.device, dtype=torch.float64)
        )
        t_total = tunnel_prob + escape_prob
        # Avoid div/0
        mask_nonzero = t_total > 0
        tunnel_prob[mask_nonzero] /= t_total[mask_nonzero]
        escape_prob[mask_nonzero] /= t_total[mask_nonzero]
        barrier = torch.clamp(barrier, min=0.0)
        
        # Lagrange Zones
        abs_z = torch.abs(z_scores)
        # We need string labels. We'll do this on CPU or use integer codes.
        # Let's use integer codes and map on CPU.
        # 0: L1, 1: CHAOS, 2: L2, 3: L3
        lz_codes = torch.zeros(num_bars, device=self.device, dtype=torch.long)
        lz_codes[abs_z < 1.0] = 0
        lz_codes[(abs_z >= 1.0) & (abs_z < 2.0)] = 1
        lz_codes[z_scores >= 2.0] = 2
        lz_codes[z_scores <= -2.0] = 3
        
        stability = torch.where(
            abs_z < 1.0, 1.0 - abs_z,
            torch.where(abs_z < 2.0, torch.tensor(0.5, device=self.device, dtype=torch.float64), torch.tensor(0.1, device=self.device, dtype=torch.float64))
        )
        
        # Lyapunov Proxy
        lyapunov = torch.zeros(num_bars, device=self.device, dtype=torch.float64)
        lyapunov[1:] = abs_z[1:] - abs_z[:-1]
        
        momentum_strength = F_momentum / (torch.abs(F_reversion) + 1e-6)
        
        # === Transfer back to CPU ===
        # Use .detach().cpu().numpy()
        
        centers_np = centers.detach().cpu().numpy()
        upper_sing_np = upper_sing.detach().cpu().numpy()
        lower_sing_np = lower_sing.detach().cpu().numpy()
        upper_event_np = upper_event.detach().cpu().numpy()
        lower_event_np = lower_event.detach().cpu().numpy()
        
        prices_out = bar_prices.detach().cpu().numpy()
        tick_vel_out = tick_velocity.detach().cpu().numpy()
        z_scores_out = z_scores.detach().cpu().numpy()
        
        F_rev_out = F_reversion.detach().cpu().numpy()
        F_up_out = F_upper_repulsion.detach().cpu().numpy()
        F_low_out = F_lower_repulsion.detach().cpu().numpy()
        F_mom_out = F_momentum.detach().cpu().numpy()
        F_net_out = F_net.detach().cpu().numpy()
        F_pid_out = F_pid.detach().cpu().numpy()
        
        P0_out = P0.detach().cpu().numpy()
        P1_out = P1.detach().cpu().numpy()
        P2_out = P2.detach().cpu().numpy()
        entropy_out = entropy.detach().cpu().numpy()
        coherence_out = coherence.detach().cpu().numpy()
        phase_out = phase.detach().cpu().numpy()
        
        pat_mat_out = pattern_maturity.detach().cpu().numpy()
        mom_str_out = momentum_strength.detach().cpu().numpy()
        
        struct_conf_out = structure_confirmed.detach().cpu().numpy()
        casc_det_out = cascade_detected.detach().cpu().numpy()
        spin_inv_out = spin_inverted.detach().cpu().numpy()
        
        lz_codes_out = lz_codes.detach().cpu().numpy()
        stab_out = stability.detach().cpu().numpy()
        
        tun_prob_out = tunnel_prob.detach().cpu().numpy()
        esc_prob_out = escape_prob.detach().cpu().numpy()
        barrier_out = barrier.detach().cpu().numpy()
        
        lyap_out = lyapunov.detach().cpu().numpy()
        sigmas_out = sigmas.detach().cpu().numpy()
        slopes_out = slopes.detach().cpu().numpy()
        
        # Trend directions from slopes
        trend_dirs = np.where(slopes_out > 0, 'UP', 'DOWN') # Simplified
        
        # Pattern Detection (CPU or CUDA via detector)
        highs_full = highs_np
        lows_full = lows_np
        opens_full = opens_np
        prices_full = prices_np

        # Calls unified detector
        pattern_types_full, candlestick_types_full = self._detect_patterns_unified(opens_full, highs_full, lows_full, prices_full)

        pattern_types = pattern_types_full[rp:]
        candlestick_types = candlestick_types_full[rp:]
        
        # Hurst/ADX placeholders (since we didn't calculate them on GPU)
        hurst_vals = np.full(num_bars, 0.5)
        
        # Calculate ADX on CPU using pandas_ta (same as CPU path)
        adx_vals = np.zeros(num_bars)
        dmp_vals = np.zeros(num_bars)
        dmn_vals = np.zeros(num_bars)

        if PANDAS_TA_AVAILABLE and 'high' in day_data.columns and 'low' in day_data.columns:
            try:
                # Compute ADX for full dataset
                adx_df = day_data.ta.adx(length=ADX_LENGTH)
                if adx_df is not None:
                    # Align with computed bars (rp:)
                    # Columns usually: ADX_14, DMP_14, DMN_14
                    adx_col = adx_df.iloc[:, 0].values
                    dmp_col = adx_df.iloc[:, 1].values
                    dmn_col = adx_df.iloc[:, 2].values

                    # Slice to match rp:
                    # Ensure alignment: day_data is same length as prices
                    if len(adx_col) >= rp + num_bars:
                        adx_vals = np.nan_to_num(adx_col[rp:])
                        dmp_vals = np.nan_to_num(dmp_col[rp:])
                        dmn_vals = np.nan_to_num(dmn_col[rp:])
            except Exception as e:
                print(f"WARNING: ADX calculation failed in GPU path and was skipped: {e}")
                pass

        # Build Results
        results = []
        
        # Complex amplitudes reconstruction
        a0 = np.sqrt(P0_out) * np.exp(1j * phase_out * 0)
        a1 = np.sqrt(P1_out) * np.exp(1j * phase_out * 1)
        a2 = np.sqrt(P2_out) * np.exp(1j * phase_out * -1)
        
        lz_map = {0: 'L1_STABLE', 1: 'CHAOS', 2: 'L2_ROCHE', 3: 'L3_ROCHE'}
        
        for i in range(num_bars):
            # Map LZ
            lz_str = lz_map.get(lz_codes_out[i], 'UNKNOWN')
            market_reg = 'CHAOTIC' if lyap_out[i] > 0 else 'STABLE'
            
            state = ThreeBodyQuantumState(
                center_position=centers_np[i],
                upper_singularity=upper_sing_np[i],
                lower_singularity=lower_sing_np[i],
                event_horizon_upper=upper_event_np[i],
                event_horizon_lower=lower_event_np[i],
                particle_position=prices_out[i],
                particle_velocity=tick_vel_out[i],
                z_score=z_scores_out[i],
                F_reversion=F_rev_out[i],
                F_upper_repulsion=F_up_out[i],
                F_lower_repulsion=F_low_out[i],
                F_momentum=F_mom_out[i],
                F_net=F_net_out[i],
                amplitude_center=a0[i],
                amplitude_upper=a1[i],
                amplitude_lower=a2[i],
                P_at_center=P0_out[i],
                P_near_upper=P1_out[i],
                P_near_lower=P2_out[i],
                entropy=entropy_out[i],
                coherence=coherence_out[i],
                pattern_maturity=pat_mat_out[i],
                momentum_strength=mom_str_out[i],
                structure_confirmed=bool(struct_conf_out[i]),
                cascade_detected=bool(casc_det_out[i]),
                spin_inverted=bool(spin_inv_out[i]),
                lagrange_zone=lz_str,
                stability_index=stab_out[i],
                tunnel_probability=tun_prob_out[i],
                escape_probability=esc_prob_out[i],
                barrier_height=barrier_out[i],
                pattern_type=str(pattern_types[i]),
                trend_direction_15m=str(trend_dirs[i]),
                hurst_exponent=hurst_vals[i],
                adx_strength=adx_vals[i],
                dmi_plus=dmp_vals[i],
                dmi_minus=dmn_vals[i],
                candlestick_pattern=str(candlestick_types[i]),
                sigma_fractal=sigmas_out[i],
                term_pid=F_pid_out[i],
                lyapunov_exponent=lyap_out[i],
                market_regime=market_reg,
            )
            results.append({
                'bar_idx': rp + i,
                'state': state,
                'price': prices_out[i],
                'structure_ok': (
                    lz_str in ('L2_ROCHE', 'L3_ROCHE') and
                    bool(struct_conf_out[i]) and
                    bool(casc_det_out[i])
                ),
            })
            
        return results

    # ═══════════════════════════════════════════════════════════════════════
    # VECTORIZED BATCH COMPUTATION (processes all bars at once)
    # ═══════════════════════════════════════════════════════════════════════

    def batch_compute_states(self, day_data: pd.DataFrame, use_cuda: bool = True, params: dict = None) -> list:
        """
        Compute ALL ThreeBodyQuantumState objects for a day in one vectorized pass.

        Args:
            day_data: DataFrame with 'price'/'close', 'volume', 'timestamp' columns
            use_cuda: If True and CUDA available, use GPU for exp/log ops
            params: Optional physics parameters (pid_kp, gravity_theta, etc.)
                    use_hurst (bool): Enable Hurst calculation (slow, default: True)

        Returns:
            List of dicts: [{bar_idx, state, price, prob, conf, structure_ok}, ...]
        """
        # Dispatch to GPU implementation if requested and available
        if use_cuda and self.use_gpu and TORCH_AVAILABLE:
            try:
                return self._batch_compute_states_gpu(day_data, params)
            except Exception as e:
                # Fallback to CPU if GPU run fails
                print(f"WARNING: GPU batch computation failed: {e}. Falling back to CPU.")
                import traceback
                traceback.print_exc()
                pass
        
        # Default physics parameters if not provided
        params = params or {}
        kp = params.get('pid_kp', DEFAULT_PID_KP)
        ki = params.get('pid_ki', DEFAULT_PID_KI)
        kd = params.get('pid_kd', DEFAULT_PID_KD)
        theta = params.get('gravity_theta', DEFAULT_GRAVITY_THETA)

        n = len(day_data)
        rp = self.regression_period

        if n < rp:
            return []

        # --- Extract raw arrays (avoid DataFrame overhead in loop) ---
        prices = day_data['price'].values.astype(np.float64) if 'price' in day_data.columns else day_data['close'].values.astype(np.float64)
        close = day_data['close'].values.astype(np.float64) if 'close' in day_data.columns else prices.copy()
        volumes = day_data['volume'].values.astype(np.float64) if 'volume' in day_data.columns else np.zeros(n, dtype=np.float64)
        volumes = np.nan_to_num(volumes)

        num_bars = n - rp  # bars we'll compute (indices rp..n-1)

        # ═══ STEP 1: Rolling linear regression (center, sigma, slope) ═══
        # Vectorized using cumulative sums
        x = np.arange(rp, dtype=np.float64)
        x_mean = x.mean()
        x_var = np.sum((x - x_mean) ** 2)

        centers = np.empty(num_bars)
        # Using rolling percentile for sigma (fat-tail distribution)
        sigmas = np.empty(num_bars)
        trend_directions = np.empty(num_bars, dtype=object)

        # Pre-calculate rolling residuals for robust sigma
        # Rolling residuals history (maxlen=500)
        # We'll maintain a rolling buffer of residuals
        rolling_residuals = []

        for i in range(num_bars):
            # Window ending at current bar (inclusive)
            # FIX: Align window to match calculate_three_body_state (coincident, not predictive)
            start_idx = i + 1
            end_idx = i + 1 + rp
            if end_idx > len(close):
                # End of data logic
                centers[i:] = centers[i-1] if i > 0 else 0
                sigmas[i:] = sigmas[i-1] if i > 0 else 1.0
                trend_directions[i:] = 'RANGE'
                break

            y = close[start_idx : end_idx]
            y_mean = y.mean()
            slope = np.sum((x - x_mean) * (y - y_mean)) / x_var
            intercept = y_mean - slope * x_mean
            center = slope * x[-1] + intercept

            # Residuals for this window
            residuals = y - (slope * x + intercept)
            current_residual = residuals[-1]

            # Std Dev Sigma (Regression Sigma)
            std_err = np.sqrt(np.sum(residuals ** 2) / (rp - 2))
            reg_sigma = std_err if std_err > 0 else y.std()

            # Robust Sigma (98th percentile of history)
            rolling_residuals.append(abs(current_residual))
            if len(rolling_residuals) > 500:
                rolling_residuals.pop(0)

            if len(rolling_residuals) >= 20:
                # 84th percentile ~ 1 SD. (Previous 98th was ~2 SDs)
                robust_sigma = np.percentile(rolling_residuals, 84)
                sigma = max(robust_sigma, reg_sigma)
            else:
                sigma = reg_sigma

            centers[i] = center
            sigmas[i] = max(sigma, 1e-10)

            # Trend Direction
            slope_strength = (slope * rp) / (sigma + 1e-6)
            if slope_strength > 1.0:
                trend_directions[i] = 'UP' if slope > 0 else 'DOWN'
            else:
                trend_directions[i] = 'RANGE'

        # Prices at each computed bar
        bar_prices = prices[rp:]
        bar_volumes = volumes[rp:]

        # Extract highs/lows for pattern detection and cascade
        if 'high' in day_data.columns and 'low' in day_data.columns:
            highs_full = day_data['high'].values.astype(np.float64)
            lows_full = day_data['low'].values.astype(np.float64)
        else:
            highs_full = prices.copy()
            lows_full = prices.copy()

        # Compute patterns for the whole dataset first
        # Use unified detection
        opens_full = day_data['open'].values.astype(np.float64) if 'open' in day_data.columns else prices.copy()

        pattern_types_full, candlestick_types_full = self._detect_patterns_unified(opens_full, highs_full, lows_full, close)

        pattern_types = pattern_types_full[rp:]
        candlestick_types = candlestick_types_full[rp:]

        # Slice highs/lows to matched bars for cascade check
        bar_highs = highs_full[rp:]
        bar_lows = lows_full[rp:]

        # ═══ FRACTAL & TREND PRE-COMPUTATION ═══
        # ADX/DMI (Trend Strength)
        adx_vals = np.zeros(num_bars)
        dmp_vals = np.zeros(num_bars)
        dmn_vals = np.zeros(num_bars)

        if PANDAS_TA_AVAILABLE and 'high' in day_data.columns and 'low' in day_data.columns:
            try:
                # Compute ADX for full dataset
                adx_df = day_data.ta.adx(length=ADX_LENGTH)
                if adx_df is not None:
                    # Align with computed bars (rp:)
                    # Columns usually: ADX_14, DMP_14, DMN_14
                    adx_col = adx_df.iloc[:, 0].values
                    dmp_col = adx_df.iloc[:, 1].values
                    dmn_col = adx_df.iloc[:, 2].values

                    # Slice to match rp:
                    adx_vals = np.nan_to_num(adx_col[rp:])
                    dmp_vals = np.nan_to_num(dmp_col[rp:])
                    dmn_vals = np.nan_to_num(dmn_col[rp:])
            except Exception:
                pass

        # Hurst Exponent (Fractal Dimension)
        # Defaults to True (Calculated) to preserve legacy behavior.
        # Set params={'use_hurst': False} for performance boost (aligns with GPU 0.5).
        use_hurst = params.get('use_hurst', True)
        hurst_vals = np.full(num_bars, 0.5)

        if use_hurst and HURST_AVAILABLE:
            try:
                def get_hurst(x):
                    try:
                        H, _, _ = compute_Hc(x, kind='price', simplified=True)
                        return H
                    except Exception as e:
                        print(f"WARNING: Hurst calculation failed for a window: {e}. Defaulting to 0.5.")
                        return 0.5
                if len(close) >= HURST_WINDOW:
                    series = pd.Series(close)
                    rolling_hurst = series.rolling(HURST_WINDOW).apply(get_hurst, raw=True).values
                    hurst_vals = np.nan_to_num(rolling_hurst[rp:], nan=0.5)
            except Exception as e:
                print(f"WARNING: Rolling Hurst calculation failed: {e}. Defaulting to 0.5 for all bars.")
                pass

        # ═══ STEP 2: Vectorized z-scores & force fields ═══

        # 2a. Fractal Diffusion & PID (Requires loop or complex rolling)
        tick_velocity = np.zeros(num_bars)
        tick_velocity[1:] = bar_prices[1:] - bar_prices[:-1]

        close_series = pd.Series(close)
        v_macro_series = close_series.diff().abs().rolling(window=rp).mean().fillna(1.0)
        v_macro_vals = v_macro_series.values[rp:] # Slice to match num_bars

        v_micro_vals = np.abs(tick_velocity)

        # Calculate Fractal Sigma
        velocity_ratios = np.divide(v_micro_vals, v_macro_vals + 1e-9)
        velocity_ratios = np.clip(velocity_ratios, 0.1, 10.0)

        fractal_sigmas = sigmas * (velocity_ratios ** hurst_vals)
        sigmas = np.maximum(fractal_sigmas, 1e-6)

        # Re-calculate Z-scores with Fractal Sigma
        z_scores = (bar_prices - centers) / sigmas

        # 2b. PID Forces
        errors = bar_prices - centers
        e_deriv = np.zeros(num_bars)
        e_deriv[1:] = errors[1:] - errors[:-1]

        errors_series = pd.Series(errors)
        e_integ = errors_series.rolling(window=rp, min_periods=1).sum().values

        F_pid = -(kp * errors + ki * e_integ + kd * e_deriv)

        # 2c. Gravity & Repulsion
        upper_sing = centers + self.SIGMA_ROCHE_MULTIPLIER * sigmas
        lower_sing = centers - self.SIGMA_ROCHE_MULTIPLIER * sigmas
        upper_event = centers + self.SIGMA_EVENT_MULTIPLIER * sigmas
        lower_event = centers - self.SIGMA_EVENT_MULTIPLIER * sigmas

        F_gravity = -theta * errors
        F_reversion = F_gravity # Alias

        dist_upper = np.abs(bar_prices - upper_sing) / sigmas
        dist_lower = np.abs(bar_prices - lower_sing) / sigmas
        F_upper_raw = np.minimum(1.0 / (dist_upper ** 3 + 0.01), 100.0)
        F_lower_raw = np.minimum(1.0 / (dist_lower ** 3 + 0.01), 100.0)
        F_upper_repulsion = np.where(z_scores > 0, F_upper_raw, 0.0)
        F_lower_repulsion = np.where(z_scores < 0, F_lower_raw, 0.0)

        safe_velocity = np.nan_to_num(tick_velocity)
        safe_volumes = np.nan_to_num(bar_volumes)
        F_momentum = safe_velocity * safe_volumes / (sigmas + 1e-6)

        repulsion = np.where(z_scores > 0, -F_upper_repulsion, F_lower_repulsion)
        F_net = F_gravity + F_momentum + F_pid + repulsion

        # ═══ STEP 3: Wave function (exp/log — GPU-accelerated) ═══
        E0 = -z_scores ** 2 / 2
        E1 = -(z_scores - 2.0) ** 2 / 2
        E2 = -(z_scores + 2.0) ** 2 / 2

        if use_cuda and CUDA_AVAILABLE:
            # GPU path (partial usage for wave function in legacy/CPU mode)
            # But normally we dispatch to _batch_compute_states_gpu entirely
            # So this is just fallback acceleration for CPU loop?
            # Keeping as is.
            device = torch.device('cuda')
            E0_t = torch.tensor(E0, device=device, dtype=torch.float64)
            E1_t = torch.tensor(E1, device=device, dtype=torch.float64)
            E2_t = torch.tensor(E2, device=device, dtype=torch.float64)

            max_E = torch.max(torch.stack([E0_t, E1_t, E2_t]), dim=0)[0]
            P0_t = torch.exp(E0_t - max_E)
            P1_t = torch.exp(E1_t - max_E)
            P2_t = torch.exp(E2_t - max_E)
            total = P0_t + P1_t + P2_t
            P0_t /= total
            P1_t /= total
            P2_t /= total

            eps = 1e-10
            entropy_t = -(P0_t * torch.log(P0_t + eps) + P1_t * torch.log(P1_t + eps) + P2_t * torch.log(P2_t + eps))

            P0 = P0_t.cpu().numpy()
            P1 = P1_t.cpu().numpy()
            P2 = P2_t.cpu().numpy()
            entropy = entropy_t.cpu().numpy()
        else:
            max_E = np.maximum(np.maximum(E0, E1), E2)
            P0 = np.exp(E0 - max_E)
            P1 = np.exp(E1 - max_E)
            P2 = np.exp(E2 - max_E)
            total = P0 + P1 + P2
            P0 /= total
            P1 /= total
            P2 /= total

            eps = 1e-10
            entropy = -(P0 * np.log(P0 + eps) + P1 * np.log(P1 + eps) + P2 * np.log(P2 + eps))

        coherence = entropy / np.log(3)

        phase = np.arctan2(F_momentum, F_net + 1e-6)
        a0 = np.sqrt(P0) * np.exp(1j * phase * 0)
        a1 = np.sqrt(P1) * np.exp(1j * phase * 1)
        a2 = np.sqrt(P2) * np.exp(1j * phase * -1)

        # ═══ STEP 4: Measurements (volume spike, pattern maturity) ═══
        vol_rolling_mean = np.empty(num_bars)
        for i in range(num_bars):
            start = max(0, rp + i - 19)
            end = rp + i + 1
            vol_rolling_mean[i] = volumes[start:end].mean()

        volume_spike = bar_volumes > vol_rolling_mean * 1.2
        pattern_maturity = np.where(
            np.abs(z_scores) > 2.0,
            np.minimum((np.abs(z_scores) - 2.0) / 1.0, 1.0),
            0.0
        )
        structure_confirmed = volume_spike & (pattern_maturity > 0.1)

        velocity_cascade = np.abs(tick_velocity) > VELOCITY_CASCADE_THRESHOLD
        range_cascade = (bar_highs - bar_lows) > RANGE_CASCADE_THRESHOLD
        cascade_detected = velocity_cascade | range_cascade

        if 'open' in day_data.columns:
            opens = day_data['open'].values[rp:]
        else:
            opens = close[rp:]
        spin_inverted = np.where(
            z_scores > 2.0,
            close[rp:] < opens,
            np.where(z_scores < -2.0, close[rp:] > opens, False)
        )

        # ═══ STEP 5: Tunneling ═══
        barrier = np.abs(z_scores) - 2.0
        barrier_positive = barrier > 0
        momentum_ratio = F_momentum / (F_reversion + 1e-6)

        tunnel_prob = np.where(
            barrier_positive,
            np.exp(-barrier * 2.0) * (1.0 - np.minimum(momentum_ratio, 0.9)),
            0.5
        )
        escape_prob = np.where(
            barrier_positive,
            momentum_ratio * np.exp(-barrier * 0.5),
            0.0
        )
        tunnel_total = tunnel_prob + escape_prob
        safe_total = np.where(tunnel_total > 0, tunnel_total, 1.0)
        tunnel_prob = tunnel_prob / safe_total
        escape_prob = escape_prob / safe_total
        barrier = np.maximum(barrier, 0.0)

        # ═══ STEP 6: Lagrange classification ═══
        abs_z = np.abs(z_scores)
        lagrange_zones = np.where(
            abs_z < 1.0, 'L1_STABLE',
            np.where(abs_z < 2.0, 'CHAOS',
                np.where(z_scores >= 2.0, 'L2_ROCHE',
                    np.where(z_scores <= -2.0, 'L3_ROCHE', 'UNKNOWN')))
        )
        stability = np.where(
            abs_z < 1.0, 1.0 - abs_z,
            np.where(abs_z < 2.0, 0.5, 0.1)
        )

        lyapunov = np.zeros(num_bars)
        abs_z = np.abs(z_scores)
        lyapunov[1:] = abs_z[1:] - abs_z[:-1]
        market_regimes = np.where(lyapunov > 0, 'CHAOTIC', 'STABLE')

        momentum_strength = np.nan_to_num(F_momentum / (np.abs(F_reversion) + 1e-6))

        # ═══ STEP 7: Build state objects ═══
        results = []
        for i in range(num_bars):
            state = ThreeBodyQuantumState(
                center_position=centers[i],
                upper_singularity=upper_sing[i],
                lower_singularity=lower_sing[i],
                event_horizon_upper=upper_event[i],
                event_horizon_lower=lower_event[i],
                particle_position=bar_prices[i],
                particle_velocity=tick_velocity[i],
                z_score=z_scores[i],
                F_reversion=F_reversion[i],
                F_upper_repulsion=F_upper_repulsion[i],
                F_lower_repulsion=F_lower_repulsion[i],
                F_momentum=F_momentum[i],
                F_net=F_net[i],
                amplitude_center=a0[i],
                amplitude_upper=a1[i],
                amplitude_lower=a2[i],
                P_at_center=P0[i],
                P_near_upper=P1[i],
                P_near_lower=P2[i],
                entropy=entropy[i],
                coherence=coherence[i],
                pattern_maturity=pattern_maturity[i],
                momentum_strength=momentum_strength[i],
                structure_confirmed=bool(structure_confirmed[i]),
                cascade_detected=bool(cascade_detected[i]),
                spin_inverted=bool(spin_inverted[i]),
                lagrange_zone=str(lagrange_zones[i]),
                stability_index=stability[i],
                tunnel_probability=tunnel_prob[i],
                escape_probability=escape_prob[i],
                barrier_height=barrier[i],
                pattern_type=str(pattern_types[i]),
                trend_direction_15m=str(trend_directions[i]),
                hurst_exponent=hurst_vals[i],
                adx_strength=adx_vals[i],
                dmi_plus=dmp_vals[i],
                dmi_minus=dmn_vals[i],
                candlestick_pattern=str(candlestick_types[i]),
                sigma_fractal=sigmas[i],
                term_pid=F_pid[i],
                lyapunov_exponent=lyapunov[i],
                market_regime=str(market_regimes[i]),
            )
            results.append({
                'bar_idx': rp + i,
                'state': state,
                'price': bar_prices[i],
                'structure_ok': (
                    str(lagrange_zones[i]) in ('L2_ROCHE', 'L3_ROCHE') and
                    bool(structure_confirmed[i]) and
                    bool(cascade_detected[i]) and
                    True
                ),
            })

        return results
