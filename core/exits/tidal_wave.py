"""Tidal Wave Exit  -- adverse volatility expansion against position.

Academic basis: Volatility clustering (GARCH) + structural shift detection.
If standard error suddenly expands against our position, the gravitational
center has shifted  -- abort before macro gravity pulls the trade to disaster.

Uses discovery TF for SE expansion check (not hardcoded 5m).
Suppressed when adjacent higher TF still agrees with trade direction.

Logic:
  IF SE expands > 20% in last 3 bars on discovery TF
  AND price is on wrong side of micro mean
  AND higher TF DMI does NOT agree with trade
  -> FORCE_EXIT
"""
from typing import Optional

from core.exit_engine import ExitAction, ExitResult, PositionState


# TF hierarchy for adjacent-higher lookup
_TF_HIERARCHY = [1, 5, 15, 30, 60, 120, 180, 300, 900, 1800, 3600, 14400]


class TidalWaveExit:

    def __init__(self, config=None):
        if config is None:
            from core.trading_config import TradingConfig
            config = TradingConfig()
        self._se_expansion_pct = config.tidal_wave_se_expansion_pct
        self._lookback = config.tidal_wave_lookback
        self._macro_tf = config.fdmi_macro_tf    # fallback
        self._micro_tf = config.fdmi_micro_tf    # fallback

    def evaluate(self, pos: PositionState, bar_close: float, tick_size: float,
                 belief_network=None) -> Optional[ExitResult]:
        """Check for adverse volatility expansion on discovery TF."""
        if belief_network is None:
            return None

        # Use trade's discovery TF, fallback to config macro
        _disc_tf = int(getattr(pos, 'discovery_tf_seconds', self._macro_tf))
        macro_worker = belief_network.workers.get(_disc_tf)
        if macro_worker is None:
            macro_worker = belief_network.workers.get(self._macro_tf)
        if macro_worker is None:
            return None

        # Get SE history (last N+1 bars to compute expansion)
        macro_idx = macro_worker._last_tf_bar_idx
        if macro_idx < self._lookback or not macro_worker._states:
            return None

        # Current and baseline SE
        raw_curr = macro_worker._states[min(macro_idx, len(macro_worker._states) - 1)]
        state_curr = raw_curr['state'] if isinstance(raw_curr, dict) and 'state' in raw_curr else raw_curr
        se_current = getattr(state_curr, 'regression_sigma', 0.0)

        base_idx = macro_idx - self._lookback
        if base_idx < 0 or base_idx >= len(macro_worker._states):
            return None
        raw_base = macro_worker._states[base_idx]
        state_base = raw_base['state'] if isinstance(raw_base, dict) and 'state' in raw_base else raw_base
        se_baseline = getattr(state_base, 'regression_sigma', 0.0)

        if se_baseline <= 0:
            return None

        se_expansion = (se_current - se_baseline) / se_baseline
        if se_expansion < self._se_expansion_pct:
            return None  # volatility hasn't expanded enough

        # Check if price is on the WRONG side of the discovery TF mean
        micro_z = getattr(state_curr, 'z_score', 0.0)

        if pos.side == 'long' and micro_z >= 0:
            return None  # price still above mean, not adverse
        if pos.side == 'short' and micro_z <= 0:
            return None  # price still below mean, not adverse

        # Higher TF override: if macro trend still supports, this expansion
        # might be the resonance cascade building (bands SHOULD expand).
        # ADAPTIVE: never-profitable trades skip override  -- nothing to protect.
        if pos.side == 'long':
            _peak_t = (pos.peak_favorable - pos.entry_price) / tick_size
        else:
            _peak_t = (pos.entry_price - pos.peak_favorable) / tick_size
        _never_profitable = _peak_t < 2.0 and pos.bars_held >= 4

        if not _never_profitable and self._higher_tf_agrees(belief_network, _disc_tf, pos.side):
            return None  # macro supports  -- expansion may be cascade, hold

        pnl_ticks = ((bar_close - pos.entry_price) / tick_size
                     if pos.side == 'long'
                     else (pos.entry_price - bar_close) / tick_size)

        return ExitResult(
            action=ExitAction.TIDAL_WAVE,
            exit_price=bar_close,
            reason=f"Tidal wave: SE expanded {se_expansion:.0%} in {self._lookback} bars, "
                   f"z={micro_z:+.2f} (wrong side, TF={_disc_tf}s)",
            pnl_ticks=pnl_ticks,
            bars_held=pos.bars_held,
        )

    @staticmethod
    def _higher_tf_agrees(belief_network, disc_tf_sec: int, side: str) -> bool:
        """Check if adjacent higher TF DMI still agrees with trade."""
        _higher = disc_tf_sec
        for tf in _TF_HIERARCHY:
            if tf > disc_tf_sec:
                _higher = tf
                break

        w = belief_network.workers.get(_higher)
        if w is None:
            return False  # can't check = don't suppress

        mi = w._last_tf_bar_idx
        if mi < 0 or not w._states or mi >= len(w._states):
            return False

        raw = w._states[mi]
        ms = raw['state'] if isinstance(raw, dict) and 'state' in raw else raw
        dp = getattr(ms, 'dmi_plus', 0.0)
        dm = getattr(ms, 'dmi_minus', 0.0)

        if side == 'long':
            return dp > dm
        else:
            return dm > dp
