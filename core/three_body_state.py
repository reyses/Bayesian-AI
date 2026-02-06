"""
Three-Body Quantum State Vector
Unified field theory for market microstructure
"""
from dataclasses import dataclass
import numpy as np
from typing import Optional

@dataclass(frozen=True)
class ThreeBodyQuantumState:
    """
    Complete quantum state representation of market as three-body system
    
    PHYSICS MODEL:
    - Body 1 (Center Star): Fair value regression - ATTRACTIVE
    - Body 2 (Upper Singularity): +2σ resistance - REPULSIVE  
    - Body 3 (Lower Singularity): -2σ support - REPULSIVE
    - Particle: Price - exists in SUPERPOSITION until measured
    
    QUANTUM MECHANICS:
    - Wave function ψ = a₀·ψ_center + a₁·ψ_upper + a₂·ψ_lower
    - Measurement (L8-L9) causes collapse to definite state
    - Tunneling probability determines mean reversion likelihood
    """
    
    # ═══ THREE ATTRACTORS ═══
    center_position: float           # Fair value (Body 1)
    upper_singularity: float         # +2σ resistance (Body 2)
    lower_singularity: float         # -2σ support (Body 3)
    event_horizon_upper: float       # +3σ point of no return
    event_horizon_lower: float       # -3σ point of no return
    
    # ═══ PARTICLE STATE ═══
    particle_position: float         # Current price
    particle_velocity: float         # Price momentum
    z_score: float                   # Normalized distance (σ units)
    
    # ═══ FORCE FIELDS ═══
    F_reversion: float              # Tidal force = z²/9
    F_upper_repulsion: float        # 1/r³ from upper
    F_lower_repulsion: float        # 1/r³ from lower
    F_momentum: float               # Kinetic energy
    F_net: float                    # Vector sum
    
    # ═══ QUANTUM WAVE FUNCTION ═══
    amplitude_center: complex       # a₀
    amplitude_upper: complex        # a₁
    amplitude_lower: complex        # a₂
    P_at_center: float             # |a₀|²
    P_near_upper: float            # |a₁|²
    P_near_lower: float            # |a₂|²
    
    # ═══ DECOHERENCE ═══
    entropy: float                  # Shannon entropy
    coherence: float                # 1.0=superposition, 0.0=collapsed
    pattern_maturity: float         # L7 development
    momentum_strength: float        # Normalized KE
    
    # ═══ MEASUREMENT OPERATORS ═══
    structure_confirmed: bool       # L8 validation
    cascade_detected: bool          # L9 velocity trigger
    spin_inverted: bool             # Micro confirms macro
    
    # ═══ LAGRANGE CLASSIFICATION ═══
    lagrange_zone: str             # L1_STABLE | L2_ROCHE | L3_ROCHE
    stability_index: float         # 0=chaos, 1=stable
    
    # ═══ QUANTUM TUNNELING ═══
    tunnel_probability: float      # P(revert to center)
    escape_probability: float      # P(break through horizon)
    barrier_height: float          # Potential energy
    
    # ═══ RESONANCE (PHASE 3 EXTENSION) ═══
    resonance_coherence: float = 0.0      # Phase alignment
    cascade_probability: float = 0.0      # P(flash move)
    amplitude_multiplier: float = 1.0     # Energy amplification
    resonance_type: str = 'NONE'          # NONE|PARTIAL|FULL|CRITICAL
    
    # ═══ FRACTAL (PHASE 2 EXTENSION) ═══
    fractal_alignment_count: int = 0      # How many scales at Roche
    fractal_confidence: str = 'LOW'       # LOW|MEDIUM|HIGH|EXTREME
    fractal_edge: float = 0.0             # 0-1 scale alignment
    
    # ═══ TIME EVOLUTION ═══
    time_at_roche: float = 0.0
    field_evolution_rate: float = 0.0
    timestamp: float = 0.0
    timeframe_macro: str = '15m'
    timeframe_micro: str = '15s'
    
    def __hash__(self):
        """Hash for Bayesian table lookups"""
        z_bin = int(self.z_score * 2) / 2
        momentum_bin = int(self.momentum_strength * 10) / 10
        
        return hash((
            z_bin,
            momentum_bin,
            self.lagrange_zone,
            self.structure_confirmed,
            self.cascade_detected,
            self.spin_inverted
        ))
        
    def __eq__(self, other):
        """
        Equality check for Bayesian table lookups.
        Must match __hash__ logic (binning) to ensure learning aggregation.
        """
        if not isinstance(other, ThreeBodyQuantumState):
            return False
            
        z_bin = int(self.z_score * 2) / 2
        other_z_bin = int(other.z_score * 2) / 2
        
        mom_bin = int(self.momentum_strength * 10) / 10
        other_mom_bin = int(other.momentum_strength * 10) / 10
        
        return (
            z_bin == other_z_bin and
            mom_bin == other_mom_bin and
            self.lagrange_zone == other.lagrange_zone and
            self.structure_confirmed == other.structure_confirmed and
            self.cascade_detected == other.cascade_detected and
            self.spin_inverted == other.spin_inverted
        )
    
    def get_trade_directive(self) -> dict:
        """Convert quantum state to trade decision"""
        # No trade if not at Roche limit
        if self.lagrange_zone not in ['L2_ROCHE', 'L3_ROCHE']:
            return {
                'action': 'WAIT',
                'reason': f'Not at Roche. Zone: {self.lagrange_zone}, Z: {self.z_score:.2f}'
            }
        
        # No trade if momentum override
        if self.F_momentum > self.F_reversion * 1.5:
            return {
                'action': 'WAIT',
                'reason': f'Momentum too strong (breakout likely)'
            }
        
        # No trade unless measured
        if not (self.structure_confirmed and self.cascade_detected):
            return {
                'action': 'WAIT',
                'reason': f'Wave function not collapsed'
            }
        
        # Trade if tunnel probability sufficient
        if self.tunnel_probability >= 0.80:
            if self.z_score > 2.0:
                return {
                    'action': 'SELL',
                    'confidence': self.tunnel_probability,
                    'target': self.center_position,
                    'stop': self.event_horizon_upper
                }
            else:
                return {
                    'action': 'BUY',
                    'confidence': self.tunnel_probability,
                    'target': self.center_position,
                    'stop': self.event_horizon_lower
                }
        
        return {'action': 'WAIT', 'reason': f'Tunnel prob too low ({self.tunnel_probability:.2%})'}

    @classmethod
    def null_state(cls):
        """Returns a null state (safe default)"""
        return cls(
            center_position=0.0, upper_singularity=0.0, lower_singularity=0.0,
            event_horizon_upper=0.0, event_horizon_lower=0.0,
            particle_position=0.0, particle_velocity=0.0, z_score=0.0,
            F_reversion=0.0, F_upper_repulsion=0.0, F_lower_repulsion=0.0, F_momentum=0.0, F_net=0.0,
            amplitude_center=0+0j, amplitude_upper=0+0j, amplitude_lower=0+0j,
            P_at_center=0.0, P_near_upper=0.0, P_near_lower=0.0,
            entropy=0.0, coherence=0.0, pattern_maturity=0.0, momentum_strength=0.0,
            structure_confirmed=False, cascade_detected=False, spin_inverted=False,
            lagrange_zone='L1_STABLE', stability_index=1.0,
            tunnel_probability=0.0, escape_probability=0.0, barrier_height=0.0
        )
