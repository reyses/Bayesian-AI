"""
Three-Body Quantum State Vector
Unified field theory for market microstructure
Multi-timeframe cascade with 8 layers (1D -> 1S)
"""
from dataclasses import dataclass, field
import numpy as np
from typing import Optional, List

@dataclass(frozen=True)
class ThreeBodyQuantumState:
    """
    Complete quantum state representation of market as three-body system

    PHYSICS MODEL:
    - Body 1 (Center Star): Fair value regression - ATTRACTIVE
    - Body 2 (Upper Singularity): +2 sigma resistance - REPULSIVE
    - Body 3 (Lower Singularity): -2 sigma support - REPULSIVE
    - Particle: Price - exists in SUPERPOSITION until measured

    QUANTUM MECHANICS:
    - Wave function psi = a0*psi_center + a1*psi_upper + a2*psi_lower
    - Measurement (L8-L9) causes collapse to definite state
    - Tunneling probability determines mean reversion likelihood

    BINNING:
    - z_score and momentum_strength are binned via DynamicBinner
      (Freedman-Diaconis histogram edges fitted from observed data).
    - All bin values remain continuous floats — no categorical conversion.
    - If no binner is attached, falls back to fixed equal-width bins.
    """

    # ---- Class-level dynamic binner (shared by all instances) ----
    # Set once at training start via ThreeBodyQuantumState.set_binner(binner)
    _binner = None

    @classmethod
    def set_binner(cls, binner):
        """Attach a fitted DynamicBinner for all future hash/eq operations."""
        cls._binner = binner

    @classmethod
    def get_binner(cls):
        return cls._binner

    # ═══ THREE ATTRACTORS ═══
    center_position: float           # Fair value (Body 1)
    upper_singularity: float         # +2 sigma resistance (Body 2)
    lower_singularity: float         # -2 sigma support (Body 3)
    event_horizon_upper: float       # +3 sigma point of no return
    event_horizon_lower: float       # -3 sigma point of no return

    # ═══ PARTICLE STATE ═══
    particle_position: float         # Current price
    particle_velocity: float         # Price momentum
    z_score: float                   # Normalized distance (sigma units)

    # ═══ FORCE FIELDS ═══
    F_reversion: float              # Tidal force = z squared / 9
    F_upper_repulsion: float        # 1/r cubed from upper
    F_lower_repulsion: float        # 1/r cubed from lower
    F_momentum: float               # Kinetic energy
    F_net: float                    # Vector sum

    # ═══ QUANTUM WAVE FUNCTION ═══
    amplitude_center: complex       # a0
    amplitude_upper: complex        # a1
    amplitude_lower: complex        # a2
    P_at_center: float             # |a0| squared
    P_near_upper: float            # |a1| squared
    P_near_lower: float            # |a2| squared

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
    pattern_type: str = 'NONE'     # NONE | COMPRESSION | WEDGE | BREAKDOWN

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

    # ═══ MACRO TREND (from 15m timeframe) ═══
    trend_direction_15m: str = 'UNKNOWN'  # UP | DOWN | RANGE | UNKNOWN

    # ═══ MULTI-TIMEFRAME CASCADE CONTEXT ═══
    # Layer 1: Daily context (available Day 22+)
    daily_trend: Optional[str] = None        # 'BULL', 'BEAR', 'RANGE'
    daily_volatility: Optional[str] = None   # 'HIGH', 'NORMAL', 'LOW'

    # Layer 2: 4-hour context (available Day 4+)
    h4_trend: Optional[str] = None           # 'UP', 'DOWN', 'RANGE'
    session: Optional[str] = None            # 'ASIA', 'EUROPE', 'US', 'OVERLAP'

    # Layer 3: 1-hour context (available Day 2+)
    h1_trend: Optional[str] = None           # 'UP', 'DOWN', 'RANGE'

    # Context tracking
    context_level: str = 'MINIMAL'           # 'MINIMAL', 'PARTIAL', 'FULL'

    def _get_hash_bins(self):
        """
        Map continuous z_score and momentum_strength to histogram bin centers.

        If a DynamicBinner is attached (fitted from data), uses Freedman-Diaconis
        bin edges.  Otherwise falls back to fixed equal-width bins.

        Returns:
            (z_bin: float, momentum_bin: float)  — both continuous
        """
        z_val = self.z_score if not np.isnan(self.z_score) else 0.0
        mom_val = self.momentum_strength if not np.isnan(self.momentum_strength) else 0.0

        binner = ThreeBodyQuantumState._binner

        if binner is not None and binner.is_fitted:
            z_bin = binner.transform('z_score', z_val)
            momentum_bin = binner.transform('momentum', mom_val)
        else:
            # Fallback: fixed equal-width bins (0.5 for z, 0.5 for momentum)
            z_bin = round(z_val * 2) / 2.0
            momentum_bin = round(mom_val * 2) / 2.0

        return z_bin, momentum_bin

    def _get_context_tuple(self):
        """Get context-dependent hash components based on available timeframes"""
        if self.daily_trend is not None:
            # Day 22+: Full context
            return (self.daily_trend, self.daily_volatility)
        elif self.h4_trend is not None:
            # Day 4-21: 4h context
            return (self.h4_trend, self.session)
        elif self.h1_trend is not None:
            # Day 2-3: 1h context
            return (self.h1_trend,)
        else:
            # Day 1: Core only
            return ()

    def __hash__(self):
        """
        Hierarchical hash for Bayesian table lookups.

        Bin count adapts to data distribution (Freedman-Diaconis).
        Context depth adapts to available timeframe data.
        """
        z_bin, momentum_bin = self._get_hash_bins()

        core = (
            z_bin,
            momentum_bin,
            self.lagrange_zone,
            self.structure_confirmed,
            self.cascade_detected,
            self.trend_direction_15m,
            self.pattern_type
        )

        return hash(core + self._get_context_tuple())

    def __eq__(self, other):
        """
        Equality check for Bayesian table lookups.
        Must match __hash__ logic exactly.
        """
        if not isinstance(other, ThreeBodyQuantumState):
            return False

        z_bin, mom_bin = self._get_hash_bins()
        other_z_bin, other_mom_bin = other._get_hash_bins()

        core_match = (
            z_bin == other_z_bin and
            mom_bin == other_mom_bin and
            self.lagrange_zone == other.lagrange_zone and
            self.structure_confirmed == other.structure_confirmed and
            self.cascade_detected == other.cascade_detected and
            self.trend_direction_15m == other.trend_direction_15m and
            self.pattern_type == other.pattern_type
        )

        if not core_match:
            return False

        return self._get_context_tuple() == other._get_context_tuple()

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

        # REGIME FILTER: Prevent fighting strong volatility expansion trends
        # If daily volatility is HIGH, respect the daily trend
        if self.daily_volatility == 'HIGH':
            # Don't short a Bullish Expansion
            if self.daily_trend == 'BULL' and self.z_score > 2.0:
                return {
                    'action': 'WAIT',
                    'reason': 'Regime Filter: Cannot short BULL expansion'
                }
            # Don't buy a Bearish Crash
            if self.daily_trend == 'BEAR' and self.z_score < -2.0:
                return {
                    'action': 'WAIT',
                    'reason': 'Regime Filter: Cannot buy BEAR crash'
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
            structure_confirmed=False, cascade_detected=False, spin_inverted=False, pattern_type='NONE',
            lagrange_zone='L1_STABLE', stability_index=1.0,
            tunnel_probability=0.0, escape_probability=0.0, barrier_height=0.0
        )
