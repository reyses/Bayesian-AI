"""Fractal DMI  -- dual-timeframe directional movement gating.

Five execution states derived from macro/micro DMI relationship:
  A. Fakeout Filter       -- block trend entries when macro has no energy
  B. Momentum Ignition    -- enter on micro pullback + velocity ignition in macro trend
  C. Fractal Exhaust      -- exit when micro energy spikes and dies at macro wall
  D. Structural Reversion  -- fade 3-sigma extremes in non-trending (OU) regimes

Academic basis:
  State A: Hawkes process gating (reject self-exciting noise in low-energy macro)
  State B: Hurst H>0.5 persistent trend + micro mean-reversion pullback + ignition
  State D: Ornstein-Uhlenbeck process  -- anti-persistent (H<0.5) elastic reversion

Macro = higher TF (default 1m), Micro = lower TF (default 5s).
The micro oscillates within the macro's trend container.
"""
from dataclasses import dataclass


@dataclass
class FractalDMIState:
    """Snapshot of fractal DMI conditions at current bar."""
    # Macro (e.g. 1m)
    macro_adx: float = 0.0
    macro_di_plus: float = 0.0
    macro_di_minus: float = 0.0
    macro_z: float = 0.0           # SIGNED z-score (positive = above mean)
    macro_z_abs: float = 0.0       # absolute z-score
    macro_tf: str = ''
    macro_hurst: float = 0.5       # Hurst exponent (H>0.55=persistent, H<0.45=mean-reverting)
    macro_se: float = 0.0          # regression standard error (for tidal wave)
    # Micro (e.g. 5s)
    micro_adx: float = 0.0
    micro_di_plus: float = 0.0
    micro_di_minus: float = 0.0
    micro_z: float = 0.0           # SIGNED z-score
    micro_z_abs: float = 0.0
    micro_tf: str = ''
    micro_velocity: float = 0.0    # net_force / velocity (first derivative)
    micro_volume_delta: float = 0.0  # volume delta (positive=buy pressure, negative=sell)
    # Previous micro bar (for crossover detection)
    prev_micro_di_plus: float = 0.0
    prev_micro_di_minus: float = 0.0
    prev_micro_adx: float = 0.0
    # Derived
    macro_trend: str = 'none'     # 'long', 'short', 'none'
    macro_regime: str = 'unknown' # 'trending', 'mean_reverting', 'chop'
    state_a_block: bool = False   # fakeout filter active
    state_b_long: bool = False    # momentum ignition long signal
    state_b_short: bool = False   # momentum ignition short signal
    state_c_exit: bool = False    # fractal exhaustion signal
    state_d_reversion_long: bool = False   # structural reversion long (fade short extreme)
    state_d_reversion_short: bool = False  # structural reversion short (fade long extreme)


class FractalDMI:
    """Evaluates fractal DMI states from TBN worker data."""

    def __init__(self, config=None):
        if config is None:
            from core.trading_config import TradingConfig
            config = TradingConfig()
        self._fakeout_micro_z = config.fdmi_fakeout_micro_z
        self._fakeout_macro_adx = config.fdmi_fakeout_macro_adx
        self._trend_macro_adx = config.fdmi_trend_macro_adx
        self._exhaust_micro_adx = config.fdmi_exhaust_micro_adx
        self._exhaust_macro_z = config.fdmi_exhaust_macro_z
        self._macro_tf = config.fdmi_macro_tf
        self._micro_tf = config.fdmi_micro_tf
        # C&E Matrix entry params
        self._momentum_z_max = config.ce_momentum_micro_z_max
        self._momentum_z_min = config.ce_momentum_micro_z_min
        self._reversion_macro_adx = config.ce_reversion_macro_adx
        self._reversion_z = config.ce_reversion_micro_z
        # Quant: Hurst regime thresholds
        self._hurst_persistent = config.hurst_persistent_threshold
        self._hurst_mean_revert = config.hurst_mean_revert_threshold

    def evaluate(self, belief_network) -> FractalDMIState:
        """Read macro/micro worker states from TBN and evaluate all states.

        Args:
            belief_network: TimeframeBeliefNetwork with active workers.

        Returns:
            FractalDMIState with all conditions evaluated.
        """
        result = FractalDMIState()

        # ── Get macro and micro workers ──
        macro_worker = belief_network.workers.get(self._macro_tf)
        micro_worker = belief_network.workers.get(self._micro_tf)

        if macro_worker is None or micro_worker is None:
            return result

        # ── Extract macro state (most recently closed bar) ──
        macro_state = self._get_current_state(macro_worker)
        if macro_state is not None:
            result.macro_adx = getattr(macro_state, 'adx_strength', 0.0)
            result.macro_di_plus = getattr(macro_state, 'dmi_plus', 0.0)
            result.macro_di_minus = getattr(macro_state, 'dmi_minus', 0.0)
            _raw_z = getattr(macro_state, 'z_score', 0.0)
            result.macro_z = _raw_z              # keep sign for position-aware exits
            result.macro_z_abs = abs(_raw_z)
            result.macro_tf = belief_network._TF_LABELS.get(self._macro_tf, str(self._macro_tf))
            result.macro_hurst = getattr(macro_state, 'hurst_exponent', 0.5)
            result.macro_se = getattr(macro_state, 'regression_sigma', 0.0)

        # ── Extract micro state (current + previous bar) ──
        micro_state = self._get_current_state(micro_worker)
        micro_prev = self._get_prev_state(micro_worker)
        if micro_state is not None:
            result.micro_adx = getattr(micro_state, 'adx_strength', 0.0)
            result.micro_di_plus = getattr(micro_state, 'dmi_plus', 0.0)
            result.micro_di_minus = getattr(micro_state, 'dmi_minus', 0.0)
            _raw_z = getattr(micro_state, 'z_score', 0.0)
            result.micro_z = _raw_z
            result.micro_z_abs = abs(_raw_z)
            result.micro_tf = belief_network._TF_LABELS.get(self._micro_tf, str(self._micro_tf))
            result.micro_velocity = getattr(micro_state, 'net_force', 0.0)
            result.micro_volume_delta = getattr(micro_state, 'volume_delta', 0.0)
        if micro_prev is not None:
            result.prev_micro_di_plus = getattr(micro_prev, 'dmi_plus', 0.0)
            result.prev_micro_di_minus = getattr(micro_prev, 'dmi_minus', 0.0)
            result.prev_micro_adx = getattr(micro_prev, 'adx_strength', 0.0)

        # ── Macro regime classification ──
        # Dual-source: Hurst exponent (academic, forward-looking) + ADX (lagging confirmation)
        # Hurst is the primary signal; ADX confirms when Hurst is borderline.
        _h = result.macro_hurst
        _adx = result.macro_adx
        if _h > self._hurst_persistent or (_h >= 0.50 and _adx >= self._trend_macro_adx):
            # Persistent / trending  -- Hurst proves mathematical memory
            result.macro_regime = 'trending'
            if result.macro_di_plus > result.macro_di_minus:
                result.macro_trend = 'long'
            elif result.macro_di_minus > result.macro_di_plus:
                result.macro_trend = 'short'
        elif _h < self._hurst_mean_revert or (_h <= 0.50 and _adx < self._reversion_macro_adx):
            # Anti-persistent / mean-reverting  -- OU regime
            result.macro_regime = 'mean_reverting'
        else:
            result.macro_regime = 'chop'

        # ── STATE A: Fakeout Filter ──
        # Micro breaking out hard but macro has no energy = fake breakout
        if (result.micro_z_abs >= self._fakeout_micro_z
                and result.macro_adx < self._fakeout_macro_adx):
            result.state_a_block = True

        # ── STATE B: Momentum Ignition (enhanced Wave Rider) ──
        # Conditions (all must be true):
        #   1. Macro trending (ADX > 25, DI direction confirmed)
        #   2. Micro Z in pullback zone (pulled back to fair value / 1-sigma)
        #   3. Micro velocity aligned (first derivative flips toward macro trend)
        #   4. Micro DI crossover realigns with macro direction
        _vd = result.micro_volume_delta
        if result.macro_trend == 'long':
            _z_in_pullback = (self._momentum_z_min <= result.micro_z
                              <= self._momentum_z_max)
            _velocity_aligned = result.micro_velocity > 0
            _di_crossover = (result.prev_micro_di_minus > result.prev_micro_di_plus
                             and result.micro_di_plus > result.micro_di_minus)
            # Volume delta confirms: aggressive buy pressure absorbing limit sells
            _vol_confirmed = _vd > 0 or _vd == 0.0  # 0 = no volume data, pass through
            if _z_in_pullback and _velocity_aligned and _di_crossover and _vol_confirmed:
                result.state_b_long = True

        elif result.macro_trend == 'short':
            _z_in_pullback = (-self._momentum_z_max <= result.micro_z
                              <= -self._momentum_z_min)
            _velocity_aligned = result.micro_velocity < 0
            _di_crossover = (result.prev_micro_di_plus > result.prev_micro_di_minus
                             and result.micro_di_minus > result.micro_di_plus)
            # Volume delta confirms: sellers taking the offer
            _vol_confirmed = _vd < 0 or _vd == 0.0
            if _z_in_pullback and _velocity_aligned and _di_crossover and _vol_confirmed:
                result.state_b_short = True

        # ── STATE C: Fractal Exhaustion (exit signal) ──
        # Micro ADX overextended + hooking down + price at macro boundary
        if (result.micro_adx > self._exhaust_micro_adx
                and result.micro_adx < result.prev_micro_adx  # ADX hooking down
                and result.macro_z_abs >= self._exhaust_macro_z):  # at macro wall
            result.state_c_exit = True

        # ── STATE D: Structural Reversion (Rubber Band / OU process) ──
        # Conditions:
        #   1. Macro ADX < 20 (non-trending / mean-reverting regime)
        #   2. Micro Z at 3-sigma extreme (unsustainable statistical anomaly)
        #   3. Micro velocity flips AGAINST the extreme (don't step in front of train)
        if result.macro_regime == 'mean_reverting':
            # LONG reversion: micro Z <= -3.0 (extreme low) AND velocity turns positive
            # Volume delta: informed buyers stepping in (delta > 0) or no data (pass through)
            if (result.micro_z <= -self._reversion_z
                    and result.micro_velocity > 0
                    and (_vd > 0 or _vd == 0.0)):
                result.state_d_reversion_long = True
            # SHORT reversion: micro Z >= +3.0 (extreme high) AND velocity turns negative
            # Volume delta: informed sellers hitting bid (delta < 0) or no data
            elif (result.micro_z >= self._reversion_z
                  and result.micro_velocity < 0
                  and (_vd < 0 or _vd == 0.0)):
                result.state_d_reversion_short = True

        return result

    @staticmethod
    def _get_current_state(worker):
        """Get the most recently evaluated state from a TBN worker."""
        idx = worker._last_tf_bar_idx
        if idx < 0 or not worker._states or idx >= len(worker._states):
            return None
        raw = worker._states[idx]
        return raw['state'] if isinstance(raw, dict) and 'state' in raw else raw

    @staticmethod
    def _get_prev_state(worker):
        """Get the previous bar's state (for crossover detection)."""
        idx = worker._last_tf_bar_idx - 1
        if idx < 0 or not worker._states or idx >= len(worker._states):
            return None
        raw = worker._states[idx]
        return raw['state'] if isinstance(raw, dict) and 'state' in raw else raw
