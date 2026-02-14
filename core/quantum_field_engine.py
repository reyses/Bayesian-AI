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

# Optional: Fractal & Trend Libraries
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

# Optional CUDA support
try:
    import torch
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
            self.use_gpu = False
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
        context: dict = None      # Optional multi-timeframe context
    ) -> ThreeBodyQuantumState:
        """
        MASTER FUNCTION: Computes complete quantum state
        
        Process:
        1. Calculate attractors (center, upper, lower)
        2. Compute force fields
        3. Calculate wave function
        4. Check measurements
        5. Compute tunneling probabilities
        6. Classify Lagrange zone
        """
        
        if len(df_macro) < self.regression_period:
            return ThreeBodyQuantumState.null_state()
        
        # Body 1: Center star
        # Use residuals from regression to update historical distribution
        center, reg_sigma, slope, residuals = self._calculate_center_mass(df_macro)
        
        # Update residual history with latest residual
        # The latest residual is (current_price - center)
        # But _calculate_center_mass uses historical window.
        # We need to be consistent. Let's use the residual of the last point in regression.
        latest_residual = residuals[-1]
        self.residual_history.append(abs(latest_residual))
        if len(self.residual_history) > self.residual_window:
            self.residual_history.pop(0)

        # Compute Fat-Tail Sigma (84th percentile of historical residuals)
        # 84th percentile ~ 1 Standard Deviation (assuming normal distribution)
        # 98th percentile ~ 2 SDs, which made z=2 equivalent to 4 SDs (too strict)
        if len(self.residual_history) >= 20:
            sigma = np.percentile(self.residual_history, 84)
            # Fallback to regression sigma if percentile is too small (e.g. tight consolidation)
            sigma = max(sigma, reg_sigma)
        else:
            sigma = reg_sigma

        # Trend Direction (15m slope)
        slope_strength = (slope * len(df_macro)) / (sigma + 1e-6)
        if slope_strength > 1.0:
            trend_direction = 'UP' if slope > 0 else 'DOWN'
        else:
            trend_direction = 'RANGE'

        # Bodies 2 & 3: Singularities (using robust sigma)
        upper_sing = center + self.SIGMA_ROCHE_MULTIPLIER * sigma
        lower_sing = center - self.SIGMA_ROCHE_MULTIPLIER * sigma
        upper_event = center + self.SIGMA_EVENT_MULTIPLIER * sigma
        lower_event = center - self.SIGMA_EVENT_MULTIPLIER * sigma
        
        # Particle state
        z_score = (current_price - center) / sigma if sigma > 0 else 0.0
        
        # Force fields
        forces = self._calculate_force_fields(
            current_price, center, upper_sing, lower_sing,
            z_score, sigma, current_volume, tick_velocity
        )
        
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

        # FRACTAL & TREND INDICATORS
        # Hurst Exponent (Fractal Dimension) - using df_micro close prices
        if HURST_AVAILABLE and len(df_micro) >= HURST_WINDOW:
            # Use last HURST_WINDOW bars for calculation
            hurst_series = df_micro['close'].iloc[-HURST_WINDOW:]
            try:
                H, c, _ = compute_Hc(hurst_series, kind='price', simplified=True)
                hurst_val = H
            except Exception:
                hurst_val = 0.5
        else:
            hurst_val = 0.5

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
            momentum_strength=np.nan_to_num(forces['F_momentum'] / (forces['F_reversion'] + 1e-6)),
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
            **context_args
        )
    
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
    
    def _calculate_force_fields(self, price, center, upper_sing, lower_sing, z_score, sigma, volume, velocity):
        """Compute competing gravitational forces"""
        # Tidal force (mean reversion)
        F_reversion = (z_score ** self.TIDAL_FORCE_EXPONENT) / 9.0
        
        # Singularity repulsion (inverse cube law)
        dist_upper = abs(price - upper_sing) / sigma if sigma > 0 else 1.0
        dist_lower = abs(price - lower_sing) / sigma if sigma > 0 else 1.0
        F_upper_repulsion = min(1.0 / (dist_upper ** 3 + 0.01), 100.0) if z_score > 0 else 0.0
        F_lower_repulsion = min(1.0 / (dist_lower ** 3 + 0.01), 100.0) if z_score < 0 else 0.0
        
        # Momentum - sanitize inputs to prevent NaN
        safe_velocity = np.nan_to_num(velocity)
        safe_volume = np.nan_to_num(volume)

        F_momentum = abs(safe_velocity) * safe_volume / (sigma + 1e-6)
        
        # Net force
        if z_score > 0:
            F_net = F_reversion + F_upper_repulsion - F_momentum
        else:
            F_net = F_reversion + F_lower_repulsion - F_momentum
        
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
            # GPU path: batch exp/log into single kernel launches
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
        # Relaxed from > 0.5 (z > 2.5) to > 0.1 (z > 2.1)
        structure_confirmed = volume_spike and pattern_maturity > 0.1
        
        # Cascade Logic: Velocity OR Rolling Range (5s window)
        velocity_cascade = abs(velocity) > VELOCITY_CASCADE_THRESHOLD
        
        # Rolling Window Cascade: Max High - Min Low over last 5 bars (assuming 1s bars)
        # If bars are not 1s (e.g. 15s), this checks the range of the last 5 aggregate bars (75s),
        # which is safer (captures large moves) than missing them.
        if len(df_micro) >= 5:
            window = df_micro.iloc[-5:]
            h = window['high'].max() if 'high' in window.columns else window['close'].max()
            l = window['low'].min() if 'low' in window.columns else window['close'].min()
            rolling_range = h - l
            rolling_cascade = rolling_range > RANGE_CASCADE_THRESHOLD
        else:
            # Fallback to current bar range if not enough history
            current_candle = df_micro.iloc[-1]
            h = current_candle.get('high', current_candle['close'])
            l = current_candle.get('low', current_candle['close'])
            rolling_cascade = (h - l) > RANGE_CASCADE_THRESHOLD

        cascade_detected = velocity_cascade or rolling_cascade

        # Define current_candle for spin inversion check
        current_candle = df_micro.iloc[-1]

        # Use open price if available, otherwise fallback to close
        open_price = current_candle.get('open', current_candle['close'])

        if z_score > 2.0:
            spin_inverted = current_candle['close'] < open_price
        elif z_score < -2.0:
            spin_inverted = current_candle['close'] > open_price
        else:
            spin_inverted = False
        
        # Geometric Pattern Detection
        highs = df_micro['high'].values if 'high' in df_micro.columns else df_micro['close'].values
        lows = df_micro['low'].values if 'low' in df_micro.columns else df_micro['close'].values

        # We need to detect pattern for the LAST bar.
        check_highs = highs[-20:]
        check_lows = lows[-20:]
        # Use scalar version for efficiency on last bar check
        pattern_type = detect_geometric_pattern(check_highs, check_lows)

        # Candlestick Pattern Detection
        opens = df_micro['open'].values if 'open' in df_micro.columns else df_micro['close'].values
        check_opens = opens[-20:]
        check_closes = df_micro['close'].values[-20:]

        # Use scalar version for efficiency
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

    def _detect_geometric_patterns(self, highs: np.ndarray, lows: np.ndarray) -> np.ndarray:
        """
        Vectorized geometric pattern detection.
        Wraps core.pattern_utils.detect_geometric_patterns_vectorized.
        """
        return detect_geometric_patterns_vectorized(highs, lows)

    def _detect_candlestick_patterns(self, opens: np.ndarray, highs: np.ndarray,
                                     lows: np.ndarray, closes: np.ndarray) -> np.ndarray:
        """
        Vectorized candlestick pattern detection.
        Wraps core.pattern_utils.detect_candlestick_patterns_vectorized.
        """
        return detect_candlestick_patterns_vectorized(opens, highs, lows, closes)

    # ═══════════════════════════════════════════════════════════════════════
    # VECTORIZED BATCH COMPUTATION (processes all bars at once)
    # ═══════════════════════════════════════════════════════════════════════

    def batch_compute_states(self, day_data: pd.DataFrame, use_cuda: bool = True) -> list:
        """
        Compute ALL ThreeBodyQuantumState objects for a day in one vectorized pass.

        Instead of looping 35k times calling calculate_three_body_state(),
        this processes all bars simultaneously using numpy/torch arrays.

        Args:
            day_data: DataFrame with 'price'/'close', 'volume', 'timestamp' columns
            use_cuda: If True and CUDA available, use GPU for exp/log ops

        Returns:
            List of dicts: [{bar_idx, state, price, prob, conf, structure_ok}, ...]
        """
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
        pattern_types_full = self._detect_geometric_patterns(highs_full, lows_full)
        # Slice to matched bars
        pattern_types = pattern_types_full[rp:]

        # Compute candlestick patterns for whole dataset
        opens_full = day_data['open'].values.astype(np.float64) if 'open' in day_data.columns else prices.copy()
        candlestick_types_full = self._detect_candlestick_patterns(opens_full, highs_full, lows_full, close)
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
        # Computing rolling Hurst is expensive. We'll approximate or skip for batch speed if needed.
        # For training accuracy, we should compute it.
        # We can use a simple rolling window apply if dataset isn't too huge.
        hurst_vals = np.full(num_bars, 0.5)

        if HURST_AVAILABLE:
            # We need a rolling apply on 'close'
            # simplified=True is faster.
            # Using a stride can speed it up (compute every N bars, interpolate).
            # For now, let's just do it for every bar but use a smaller window? No, H needs ~100 bars.
            # Optimization: Compute only if we need it.
            # Let's try to do it via rolling apply.
            try:
                # Custom function for rolling apply
                def get_hurst(x):
                    try:
                        H, _, _ = compute_Hc(x, kind='price', simplified=True)
                        return H
                    except Exception as e:
                        # Consider logging the exception here, e.g., logging.error(f"Error computing Hurst: {e}")
                        return 0.5

                # We apply to the *original* series, then slice
                # Calculating on 1s data (35k bars) with window=100 is 35k*100 ops.
                # It might take a few seconds. Acceptable.
                # However, calculate_three_body_state uses df_micro (15s usually).
                # batch_compute_states is called with 15s data in Orchestrator.
                # So n ~ 2000 bars. This is fast enough.

                # Check if we have enough data
                if len(close) >= HURST_WINDOW:
                    series = pd.Series(close)
                    # rolling(100).apply(get_hurst)
                    # raw=True for speed (passes ndarray)
                    rolling_hurst = series.rolling(HURST_WINDOW).apply(get_hurst, raw=True).values
                    hurst_vals = np.nan_to_num(rolling_hurst[rp:], nan=0.5)
            except Exception:
                pass

        # ═══ STEP 2: Vectorized z-scores & force fields ═══
        z_scores = (bar_prices - centers) / sigmas

        upper_sing = centers + self.SIGMA_ROCHE_MULTIPLIER * sigmas
        lower_sing = centers - self.SIGMA_ROCHE_MULTIPLIER * sigmas
        upper_event = centers + self.SIGMA_EVENT_MULTIPLIER * sigmas
        lower_event = centers - self.SIGMA_EVENT_MULTIPLIER * sigmas

        F_reversion = (z_scores ** 2) / 9.0

        dist_upper = np.abs(bar_prices - upper_sing) / sigmas
        dist_lower = np.abs(bar_prices - lower_sing) / sigmas
        F_upper_raw = np.minimum(1.0 / (dist_upper ** 3 + 0.01), 100.0)
        F_lower_raw = np.minimum(1.0 / (dist_lower ** 3 + 0.01), 100.0)
        F_upper_repulsion = np.where(z_scores > 0, F_upper_raw, 0.0)
        F_lower_repulsion = np.where(z_scores < 0, F_lower_raw, 0.0)

        # Tick velocity: price change between consecutive bars
        tick_velocity = np.zeros(num_bars)
        tick_velocity[1:] = bar_prices[1:] - bar_prices[:-1]

        # Momentum: |velocity| * volume / sigma (matches per-bar _calculate_force_fields)
        safe_velocity = np.nan_to_num(tick_velocity)
        safe_volumes = np.nan_to_num(bar_volumes)
        F_momentum = np.abs(safe_velocity) * safe_volumes / (sigmas + 1e-6)

        F_net = np.where(
            z_scores > 0,
            F_reversion + F_upper_repulsion - F_momentum,
            F_reversion + F_lower_repulsion - F_momentum
        )

        # ═══ STEP 3: Wave function (exp/log — GPU-accelerated) ═══
        E0 = -z_scores ** 2 / 2
        E1 = -(z_scores - 2.0) ** 2 / 2
        E2 = -(z_scores + 2.0) ** 2 / 2

        if use_cuda and CUDA_AVAILABLE:
            # GPU path
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
            # CPU path (still vectorized — much faster than per-bar loop)
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

        # Amplitudes
        phase = np.arctan2(F_momentum, F_net + 1e-6)
        a0 = np.sqrt(P0) * np.exp(1j * phase * 0)
        a1 = np.sqrt(P1) * np.exp(1j * phase * 1)
        a2 = np.sqrt(P2) * np.exp(1j * phase * -1)

        # ═══ STEP 4: Measurements (volume spike, pattern maturity) ═══
        # Rolling mean of volume over last 20 bars
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
        # Relaxed from > 0.5 to > 0.1 (z > 2.1)
        structure_confirmed = volume_spike & (pattern_maturity > 0.1)

        # Cascade Logic: Velocity OR Range
        velocity_cascade = np.abs(tick_velocity) > VELOCITY_CASCADE_THRESHOLD
        range_cascade = (bar_highs - bar_lows) > RANGE_CASCADE_THRESHOLD
        cascade_detected = velocity_cascade | range_cascade

        # Spin inversion
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

        momentum_strength = np.nan_to_num(F_momentum / (F_reversion + 1e-6))

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
            )
            results.append({
                'bar_idx': rp + i,
                'state': state,
                'price': bar_prices[i],
                'structure_ok': (
                    str(lagrange_zones[i]) in ('L2_ROCHE', 'L3_ROCHE') and
                    bool(structure_confirmed[i]) and
                    bool(cascade_detected[i]) and
                    True  # conf >= 0.30 checked at lookup time
                ),
            })

        return results
