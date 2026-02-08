"""
Quantum Field Calculator
Computes three-body gravitational fields + quantum wave function
Integrates Nightmare Protocol gravity calculations
"""
import numpy as np
import pandas as pd
from scipy.stats import linregress
from core.three_body_state import ThreeBodyQuantumState

class QuantumFieldEngine:
    """
    Unified field calculator
    - Nightmare Protocol (gravity wells, O-U process)
    - Three-body dynamics (Lagrange points, tidal forces)
    - Quantum mechanics (superposition, tunneling)
    """
    
    def __init__(self, regression_period: int = 21):
        self.regression_period = regression_period
        self.SIGMA_ROCHE_MULTIPLIER = 2.0
        self.SIGMA_EVENT_MULTIPLIER = 3.0
        self.TIDAL_FORCE_EXPONENT = 2.0
    
    def calculate_three_body_state(
        self, 
        df_macro: pd.DataFrame,   # 15min bars
        df_micro: pd.DataFrame,   # 15sec bars
        current_price: float,
        current_volume: float,
        tick_velocity: float
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
        center, sigma, slope = self._calculate_center_mass(df_macro)
        
        # Bodies 2 & 3: Singularities
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
        
        # Tunneling
        tunnel_prob, escape_prob, barrier = self._calculate_tunneling(
            z_score, forces['F_momentum'], forces['F_reversion']
        )
        
        # Lagrange
        lagrange_zone, stability = self._classify_lagrange(z_score, forces['F_net'])
        
        return ThreeBodyQuantumState(
            center_position=center,
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
            momentum_strength=forces['F_momentum'] / (forces['F_reversion'] + 1e-6),
            structure_confirmed=measurements['structure_confirmed'],
            cascade_detected=measurements['cascade_detected'],
            spin_inverted=measurements['spin_inverted'],
            lagrange_zone=lagrange_zone,
            stability_index=stability,
            tunnel_probability=tunnel_prob,
            escape_probability=escape_prob,
            barrier_height=barrier,
            timestamp=df_macro.index[-1].timestamp() if hasattr(df_macro.index[-1], 'timestamp') else 0.0
        )
    
    def _calculate_center_mass(self, df: pd.DataFrame):
        """Linear regression for center star"""
        window = df.iloc[-self.regression_period:]
        y = window['close'].values
        x = np.arange(len(y))
        slope, intercept, _, _, std_err = linregress(x, y)
        center = slope * x[-1] + intercept
        sigma = std_err if std_err > 0 else y.std()
        return center, sigma, slope
    
    def _calculate_force_fields(self, price, center, upper_sing, lower_sing, z_score, sigma, volume, velocity):
        """Compute competing gravitational forces"""
        # Tidal force (mean reversion)
        F_reversion = (z_score ** self.TIDAL_FORCE_EXPONENT) / 9.0
        
        # Singularity repulsion (inverse cube law)
        dist_upper = abs(price - upper_sing) / sigma if sigma > 0 else 1.0
        dist_lower = abs(price - lower_sing) / sigma if sigma > 0 else 1.0
        F_upper_repulsion = min(1.0 / (dist_upper ** 3 + 0.01), 100.0) if z_score > 0 else 0.0
        F_lower_repulsion = min(1.0 / (dist_lower ** 3 + 0.01), 100.0) if z_score < 0 else 0.0
        
        # Momentum
        F_momentum = abs(velocity) * volume / (sigma + 1e-6)
        
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
        """Quantum superposition state"""
        # Gaussian probabilities
        P0 = np.exp(-z_score**2 / 2)
        P1 = np.exp(-(z_score - 2.0)**2 / 2)
        P2 = np.exp(-(z_score + 2.0)**2 / 2)
        
        # Normalize
        total = P0 + P1 + P2
        P0 /= total
        P1 /= total
        P2 /= total
        
        # Amplitudes
        phase = np.arctan2(F_momentum, F_net + 1e-6)
        a0 = np.sqrt(P0) * np.exp(1j * phase * 0)
        a1 = np.sqrt(P1) * np.exp(1j * phase * 1)
        a2 = np.sqrt(P2) * np.exp(1j * phase * -1)
        
        # Entropy
        epsilon = 1e-10
        entropy = -(P0*np.log(P0+epsilon) + P1*np.log(P1+epsilon) + P2*np.log(P2+epsilon))
        coherence = entropy / np.log(3)
        
        return {'a0': a0, 'a1': a1, 'a2': a2, 'P0': P0, 'P1': P1, 'P2': P2, 
                'entropy': entropy, 'coherence': coherence}
    
    def _check_measurements(self, df_micro, z_score, velocity):
        """L8-L9 measurement operators"""
        if len(df_micro) < 20:
            return {
                'structure_confirmed': False,
                'cascade_detected': False,
                'spin_inverted': False,
                'pattern_maturity': 0.0
            }
        
        recent = df_micro.iloc[-20:]
        volume_spike = recent['volume'].iloc[-1] > recent['volume'].mean() * 1.2
        pattern_maturity = min((abs(z_score) - 2.0) / 1.0, 1.0) if abs(z_score) > 2.0 else 0.0
        structure_confirmed = volume_spike and pattern_maturity > 0.5
        
        cascade_threshold = 10.0
        cascade_detected = abs(velocity) > cascade_threshold
        
        current_candle = df_micro.iloc[-1]
        # Use open price if available, otherwise fallback to close
        open_price = current_candle['open'] if 'open' in current_candle else current_candle['close']

        if z_score > 2.0:
            spin_inverted = current_candle['close'] < open_price
        elif z_score < -2.0:
            spin_inverted = current_candle['close'] > open_price
        else:
            spin_inverted = False
        
        return {
            'structure_confirmed': structure_confirmed,
            'cascade_detected': cascade_detected,
            'spin_inverted': spin_inverted,
            'pattern_maturity': pattern_maturity
        }
    
    def _calculate_tunneling(self, z_score, F_momentum, F_reversion):
        """Quantum tunneling probabilities"""
        barrier = abs(z_score) - 2.0
        if barrier < 0:
            return 0.5, 0.0, 0.0
        
        momentum_ratio = F_momentum / (F_reversion + 1e-6)
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
