"""
Fractal Three-Body State
Each layer is a complete three-body system nested within parent layer
Markets are self-similar across all scales
"""
from dataclasses import dataclass
from typing import List, Optional
import numpy as np
import pandas as pd
from scipy.stats import linregress

@dataclass
class FractalThreeBodyLayer:
    """Single layer in fractal hierarchy"""
    timeframe: str              # '90d', '30d', '7d', '1d', '4h', '1h', '15m', '5m', '1s'
    parent_timeframe: str
    child_timeframe: str
    
    # Three-body at THIS scale
    center_mass: float
    upper_singularity: float
    lower_singularity: float
    local_position: float
    local_z_score: float
    local_velocity: float
    
    # Forces at THIS scale
    F_reversion_local: float
    F_momentum_local: float
    
    # Classification at THIS scale
    lagrange_zone_local: str
    
    # Quantum at THIS scale
    wave_function_local: complex
    tunnel_prob_local: float
    
    # Cross-scale coupling
    parent_coupling: float = 0.0
    child_coupling: float = 0.0
    
    # Resonance
    phase: float = 0.0
    frequency: float = 0.0

class FractalMarketState:
    """Complete fractal representation: 9 nested three-body systems"""
    
    def __init__(self):
        self.timeframes = ['90d', '30d', '7d', '1d', '4h', '1h', '15m', '5m', '1s']
    
    def calculate_fractal_state(self, price_history: dict) -> List[FractalThreeBodyLayer]:
        """
        Compute three-body state at ALL 9 scales simultaneously
        
        Args:
            price_history: {'90d': DataFrame, '30d': DataFrame, ...}
        
        Returns: List of 9 FractalThreeBodyLayer objects
        """
        layers = []
        
        for i, timeframe in enumerate(self.timeframes):
            df = price_history[timeframe]
            layer = self._compute_layer_three_body(
                df,
                timeframe,
                parent_tf=self.timeframes[i-1] if i > 0 else None,
                child_tf=self.timeframes[i+1] if i < 8 else None
            )
            layers.append(layer)
        
        self._calculate_coupling(layers)
        return layers
    
    def _compute_layer_three_body(self, df, timeframe, parent_tf, child_tf):
        """SAME three-body calculation, different timeframe"""
        lookback = 21
        if len(df) < lookback:
            # Return null layer
            return FractalThreeBodyLayer(
                timeframe=timeframe,
                parent_timeframe=parent_tf or 'NONE',
                child_timeframe=child_tf or 'NONE',
                center_mass=0.0, upper_singularity=0.0, lower_singularity=0.0,
                local_position=0.0, local_z_score=0.0, local_velocity=0.0,
                F_reversion_local=0.0, F_momentum_local=0.0,
                lagrange_zone_local='L1_STABLE',
                wave_function_local=0+0j, tunnel_prob_local=0.5
            )
        
        window = df.iloc[-lookback:]
        y = window['close'].values
        x = np.arange(len(y))
        slope, intercept, _, _, std_err = linregress(x, y)
        center = slope * x[-1] + intercept
        sigma = std_err if std_err > 0 else y.std()
        
        upper_sing = center + 2.0 * sigma
        lower_sing = center - 2.0 * sigma
        current_price = df['close'].iloc[-1]
        z_score_local = (current_price - center) / sigma if sigma > 0 else 0.0
        
        velocity_local = df['close'].iloc[-1] - df['close'].iloc[-2] if len(df) >= 2 else 0.0
        
        F_reversion = (z_score_local ** 2) / 9.0
        F_momentum = abs(velocity_local) / (sigma + 1e-6)
        
        if abs(z_score_local) < 1.0:
            lagrange = 'L1_STABLE'
        elif abs(z_score_local) >= 2.0:
            lagrange = 'L2_ROCHE' if z_score_local > 0 else 'L3_ROCHE'
        else:
            lagrange = 'DRIFT'
        
        tunnel_prob = np.exp(-(abs(z_score_local) - 2.0) * 2.0) if abs(z_score_local) >= 2.0 else 0.5
        wave_func = np.exp(-z_score_local**2 / 2) * np.exp(1j * 0)
        
        phase = np.arctan2(velocity_local, current_price - center)
        if phase < 0:
            phase += 2*np.pi
        
        freq_map = {
            '90d': 2*np.pi / (90*86400), '30d': 2*np.pi / (30*86400),
            '7d': 2*np.pi / (7*86400), '1d': 2*np.pi / 86400,
            '4h': 2*np.pi / (4*3600), '1h': 2*np.pi / 3600,
            '15m': 2*np.pi / (15*60), '5m': 2*np.pi / (5*60), '1s': 2*np.pi
        }
        frequency = freq_map.get(timeframe, 0.0)
        
        return FractalThreeBodyLayer(
            timeframe=timeframe,
            parent_timeframe=parent_tf or 'NONE',
            child_timeframe=child_tf or 'NONE',
            center_mass=center,
            upper_singularity=upper_sing,
            lower_singularity=lower_sing,
            local_position=current_price,
            local_z_score=z_score_local,
            local_velocity=velocity_local,
            F_reversion_local=F_reversion,
            F_momentum_local=F_momentum,
            lagrange_zone_local=lagrange,
            wave_function_local=wave_func,
            tunnel_prob_local=tunnel_prob,
            phase=phase,
            frequency=frequency
        )
    
    def _calculate_coupling(self, layers):
        """Calculate cross-scale influence"""
        for i, layer in enumerate(layers):
            if i > 0:
                parent = layers[i-1]
                layer.parent_coupling = abs(parent.local_z_score) / 3.0
            if i < len(layers) - 1:
                child = layers[i+1]
                layer.child_coupling = abs(child.local_z_score) / 10.0

class FractalTradingLogic:
    """Trading decisions using fractal alignment"""
    
    @staticmethod
    def check_fractal_alignment(layers: List[FractalThreeBodyLayer]) -> dict:
        """
        Check if multiple scales aligned at Roche
        Highest confidence when ALL scales agree
        """
        roche_count = sum(
            1 for layer in layers 
            if layer.lagrange_zone_local in ['L2_ROCHE', 'L3_ROCHE']
        )
        
        tunnel_probs = [layer.tunnel_prob_local for layer in layers]
        avg_tunnel = np.mean(tunnel_probs)
        
        phases = [layer.phase for layer in layers]
        phase_coherence = abs(np.mean([np.exp(1j*p) for p in phases]))
        
        z_scores = [layer.local_z_score for layer in layers]
        all_positive = all(z > 0 for z in z_scores)
        all_negative = all(z < 0 for z in z_scores)
        direction_consensus = all_positive or all_negative
        
        if roche_count >= 7 and avg_tunnel > 0.80 and direction_consensus:
            confidence = 'EXTREME'  # 95%+
        elif roche_count >= 5 and avg_tunnel > 0.70:
            confidence = 'HIGH'  # 85%+
        elif roche_count >= 3 and avg_tunnel > 0.60:
            confidence = 'MEDIUM'  # 70%+
        else:
            confidence = 'LOW'
        
        return {
            'roche_alignment_count': roche_count,
            'avg_tunnel_probability': avg_tunnel,
            'phase_coherence': phase_coherence,
            'direction_consensus': direction_consensus,
            'confidence_level': confidence,
            'fractal_edge': roche_count / 9.0
        }
