"""
Market State Vector
Statistical representation of market microstructure.
Multi-timeframe cascade with 8 layers (1D -> 1S).
"""
from dataclasses import dataclass
import numpy as np
from typing import Optional

@dataclass(frozen=True)
class MarketState:
    """
    Complete statistical state representation of market microstructure.

    REGRESSION MODEL:
    - Center: Fair value (OLS regression center)
    - Upper band: +2 sigma resistance
    - Lower band: -2 sigma support
    - Price: Current position relative to bands

    PROBABILITY MODEL:
    - 3-class softmax: P(near center), P(near upper), P(near lower)
    - Confirmation signals (structure + cascade) validate setups
    - Reversion probability from OU first-passage analysis

    BINNING:
    - z_score and momentum_strength are binned via DynamicBinner
      (Freedman-Diaconis histogram edges fitted from observed data).
    - All bin values remain continuous floats  -- no categorical conversion.
    - If no binner is attached, falls back to fixed equal-width bins.
    """

    # ---- Class-level dynamic binner (shared by all instances) ----
    # Set once at training start via MarketState.set_binner(binner)
    _binner = None

    @classmethod
    def set_binner(cls, binner):
        """Attach a fitted DynamicBinner for all future hash/eq operations."""
        cls._binner = binner

    @classmethod
    def get_binner(cls):
        return cls._binner

    # ═══ REGRESSION BANDS ═══
    regression_center: float           # Fair value (OLS center)
    upper_band_2sigma: float         # +2 sigma resistance
    lower_band_2sigma: float         # -2 sigma support
    upper_band_3sigma: float       # +3 sigma breakout level
    lower_band_3sigma: float       # -3 sigma breakout level

    # ═══ PRICE STATE ═══
    price: float         # Current price
    velocity: float         # Price momentum
    z_score: float                   # Normalized distance (sigma units)

    # ═══ STATISTICAL FORCES ═══
    mean_reversion_force: float              # Mean reversion force = -theta * z * sigma
    F_upper_band: float             # Band pressure from upper
    F_lower_band: float             # Band pressure from lower
    F_momentum: float               # Kinetic energy
    net_force: float                    # Vector sum

    # ═══ PROBABILITY DISTRIBUTION ═══
    prob_weight_center: complex     # sqrt(P_at_center)
    prob_weight_upper: complex      # sqrt(P_near_upper)
    prob_weight_lower: complex      # sqrt(P_near_lower)
    P_at_center: float             # |a0| squared
    P_near_upper: float            # |a1| squared
    P_near_lower: float            # |a2| squared

    # ═══ SIGNAL QUALITY ═══
    entropy: float                  # Shannon entropy
    entropy_normalized: float                # 1.0=uncertain/mixed, 0.0=decisive/aligned
    pattern_maturity: float         # L7 development
    momentum_strength: float        # Normalized KE

    # ═══ CONFIRMATION SIGNALS ═══
    structure_confirmed: bool       # L8 validation
    cascade_detected: bool          # L9 velocity trigger
    reversal_confirmed: bool             # Micro confirms macro

    # ═══ BAND CLASSIFICATION ═══
    band_zone: str             # INNER | UPPER_EXTREME | LOWER_EXTREME
    stability_index: float         # 0=chaos, 1=stable

    # ═══ REVERSION STATISTICS ═══
    reversion_probability: float      # P(revert to center)
    breakout_probability: float      # P(break through horizon)
    reversion_potential: float     # OU potential energy
    pattern_type: str = 'NONE'     # NONE | COMPRESSION | WEDGE | BREAKDOWN

    # ═══ FRACTAL & TREND INDICATORS ═══
    hurst_exponent: float = 0.5     # Fractal dimension (0.0-1.0)
    adx_strength: float = 0.0       # Trend strength (0-100)
    dmi_plus: float = 0.0           # Directional Movement Plus
    dmi_minus: float = 0.0          # Directional Movement Minus
    adx_prev: float = 0.0           # Previous bar's ADX (for slope computation)
    di_plus_prev: float = 0.0       # Previous bar's DI+ (for crossover detection)
    di_minus_prev: float = 0.0      # Previous bar's DI- (for crossover detection)
    volume_delta: float = 0.0          # Volume delta: positive=buy pressure, negative=sell
    candlestick_pattern: str = 'NONE' # HAMMER | ENGULFING_BULL | ENGULFING_BEAR | DOJI | NONE

    # ═══ VOLATILITY MODEL COMPONENTS ═══
    regression_sigma: float = 0.0      # Fractal diffusion volatility
    term_pid: float = 0.0           # Algorithmic control force
    oscillation_entropy_normalized: float = 0.0 # 1=tight periodic PID oscillation, 0=noisy/trending
    lyapunov_exponent: float = 0.0  # Stability coefficient (lambda)
    market_regime: str = 'UNKNOWN'  # STABLE | CHAOTIC

    # ═══ ALIGNMENT (PHASE 3 EXTENSION) ═══
    alignment_score: float = 0.0      # Phase alignment
    cascade_probability: float = 0.0      # P(flash move)
    signal_multiplier: float = 1.0        # Signal amplification
    alignment_type: str = 'NONE'          # NONE|PARTIAL|FULL|STRONG

    # ═══ FRACTAL (PHASE 2 EXTENSION) ═══
    multi_tf_alignment_count: int = 0      # How many scales at band extreme
    fractal_confidence: str = 'LOW'       # LOW|MEDIUM|HIGH|EXTREME
    fractal_edge: float = 0.0             # 0-1 scale alignment

    # ═══ NOISE MEASUREMENT ═══
    swing_noise_ticks: float = 35.0   # Max intra-wave pullback (rolling 32-bar window, in ticks)

    # ═══ TIME EVOLUTION ═══
    time_at_band_extreme: float = 0.0
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

    # Multi-timeframe Fractal Patterns
    daily_pattern: Optional[str] = 'NONE'    # Daily fractal pattern
    h4_pattern: Optional[str] = 'NONE'       # H4 fractal pattern
    h1_pattern: Optional[str] = 'NONE'       # H1 fractal pattern

    # Context tracking
    context_level: str = 'MINIMAL'           # 'MINIMAL', 'PARTIAL', 'FULL'

    def _get_hash_bins(self):
        """
        Map continuous z_score and momentum_strength to histogram bin centers.

        If a DynamicBinner is attached (fitted from data), uses Freedman-Diaconis
        bin edges.  Otherwise falls back to fixed equal-width bins.

        Returns:
            (z_bin: float, momentum_bin: float)   -- both continuous
        """
        z_val = self.z_score if not np.isnan(self.z_score) else 0.0
        mom_val = self.momentum_strength if not np.isnan(self.momentum_strength) else 0.0

        binner = MarketState._binner

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
            self.band_zone,
            self.structure_confirmed,
            self.cascade_detected,
            self.trend_direction_15m,
            self.pattern_type,
            self.market_regime
        )

        return hash(core + self._get_context_tuple())

    def __eq__(self, other):
        """
        Equality check for Bayesian table lookups.
        Must match __hash__ logic exactly.
        """
        if not isinstance(other, MarketState):
            return False

        z_bin, mom_bin = self._get_hash_bins()
        other_z_bin, other_mom_bin = other._get_hash_bins()

        core_match = (
            z_bin == other_z_bin and
            mom_bin == other_mom_bin and
            self.band_zone == other.band_zone and
            self.structure_confirmed == other.structure_confirmed and
            self.cascade_detected == other.cascade_detected and
            self.trend_direction_15m == other.trend_direction_15m and
            self.pattern_type == other.pattern_type and
            self.market_regime == other.market_regime
        )

        if not core_match:
            return False

        return self._get_context_tuple() == other._get_context_tuple()

    @classmethod
    def null_state(cls):
        """Returns a null state (safe default)"""
        return cls(
            regression_center=0.0, upper_band_2sigma=0.0, lower_band_2sigma=0.0,
            upper_band_3sigma=0.0, lower_band_3sigma=0.0,
            price=0.0, velocity=0.0, z_score=0.0,
            mean_reversion_force=0.0, F_upper_band=0.0, F_lower_band=0.0, F_momentum=0.0, net_force=0.0,
            prob_weight_center=0+0j, prob_weight_upper=0+0j, prob_weight_lower=0+0j,
            P_at_center=0.0, P_near_upper=0.0, P_near_lower=0.0,
            entropy=0.0, entropy_normalized=0.0, pattern_maturity=0.0, momentum_strength=0.0,
            structure_confirmed=False, cascade_detected=False, reversal_confirmed=False, pattern_type='NONE',
            band_zone='INNER', stability_index=1.0,
            reversion_probability=0.0, breakout_probability=0.0, reversion_potential=0.0,
            hurst_exponent=0.5, adx_strength=0.0, dmi_plus=0.0, dmi_minus=0.0,
            candlestick_pattern='NONE',
            regression_sigma=0.0, term_pid=0.0, oscillation_entropy_normalized=0.0, lyapunov_exponent=0.0, market_regime='STABLE'
        )
