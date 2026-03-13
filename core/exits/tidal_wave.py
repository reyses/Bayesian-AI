"""Tidal Wave Exit — adverse volatility expansion against position.

Academic basis: Volatility clustering (GARCH) + structural shift detection.
If standard error (volatility) suddenly expands against our position,
the gravitational center has shifted — abort before macro gravity pulls
the trade to a disastrous level.

Logic:
  IF position is open (long)
  AND macro SE expands by > 20% in last 3 bars (violent structural shift)
  AND price is below micro mean (caught on wrong side of expansion)
  → FORCE_EXIT
"""
from typing import Optional

from core.exit_engine import ExitAction, ExitResult, PositionState


class TidalWaveExit:

    def __init__(self, config=None):
        if config is None:
            from core.trading_config import TradingConfig
            config = TradingConfig()
        self._se_expansion_pct = config.tidal_wave_se_expansion_pct
        self._lookback = config.tidal_wave_lookback
        self._macro_tf = config.fdmi_macro_tf
        self._micro_tf = config.fdmi_micro_tf

    def evaluate(self, pos: PositionState, bar_close: float, tick_size: float,
                 belief_network=None) -> Optional[ExitResult]:
        """Check for adverse volatility expansion."""
        if belief_network is None:
            return None

        macro_worker = belief_network.workers.get(self._macro_tf)
        micro_worker = belief_network.workers.get(self._micro_tf)
        if macro_worker is None or micro_worker is None:
            return None

        # Get macro SE history (last N+1 bars to compute expansion)
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

        # Check if price is on the WRONG side of the micro mean
        micro_idx = micro_worker._last_tf_bar_idx
        if micro_idx < 0 or not micro_worker._states:
            return None
        raw_micro = micro_worker._states[min(micro_idx, len(micro_worker._states) - 1)]
        micro_state = raw_micro['state'] if isinstance(raw_micro, dict) and 'state' in raw_micro else raw_micro
        micro_z = getattr(micro_state, 'z_score', 0.0)

        # Long: price below mean (z < 0) = caught on wrong side
        # Short: price above mean (z > 0) = caught on wrong side
        if pos.side == 'long' and micro_z >= 0:
            return None  # price still above mean, not adverse
        if pos.side == 'short' and micro_z <= 0:
            return None  # price still below mean, not adverse

        pnl_ticks = ((bar_close - pos.entry_price) / tick_size
                     if pos.side == 'long'
                     else (pos.entry_price - bar_close) / tick_size)

        return ExitResult(
            action=ExitAction.REGIME_DECAY,  # reuse closest action type
            exit_price=bar_close,
            reason=f"Tidal wave: SE expanded {se_expansion:.0%} in {self._lookback} bars, "
                   f"micro_z={micro_z:+.2f} (wrong side)",
            pnl_ticks=pnl_ticks,
            bars_held=pos.bars_held,
        )
