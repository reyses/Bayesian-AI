"""Death Hook Exit (formerly Fractal Exhaust)  -- Liquidity Absorption at macro wall.

Academic basis: Order Book Imbalance / Toxic Flow.
A sudden drop in directional energy at a known mathematical boundary means
limit orders are absorbing the market orders.

Logic:
  IF position is open
  AND price at macro 2-sigma band IN THE FAVORABLE DIRECTION
  AND micro ADX > 40 (energy was extreme)
  AND micro ADX[current] < micro ADX[previous] (ADX vector rolls over  -- energy dying)
  -> FORCE_MARKET_EXIT (exit at the wall, don't wait for reversal)

Position-aware: for longs, checks upper band (macro_z > 0); for shorts, lower band.
"""
from typing import Optional

from core.exit_engine import ExitAction, ExitResult, PositionState


class FractalExhaustExit:

    def __init__(self, config=None):
        if config is None:
            from core.trading_config import TradingConfig
            config = TradingConfig()
        self._micro_adx_threshold = config.ce_death_hook_micro_adx
        self._macro_z_threshold = config.ce_death_hook_macro_z
        self._micro_tf = config.fdmi_micro_tf
        self._macro_tf = config.fdmi_macro_tf

    def evaluate(self, pos: PositionState, bar_close: float, tick_size: float,
                 belief_network=None) -> Optional[ExitResult]:
        """Check Death Hook / liquidity absorption condition."""
        if belief_network is None:
            return None

        macro_worker = belief_network.workers.get(self._macro_tf)
        micro_worker = belief_network.workers.get(self._micro_tf)
        if macro_worker is None or micro_worker is None:
            return None

        # Current micro state
        micro_idx = micro_worker._last_tf_bar_idx
        if micro_idx < 1 or not micro_worker._states:
            return None
        micro_raw = micro_worker._states[min(micro_idx, len(micro_worker._states) - 1)]
        micro_state = micro_raw['state'] if isinstance(micro_raw, dict) and 'state' in micro_raw else micro_raw
        micro_adx = getattr(micro_state, 'adx_strength', 0.0)

        # Previous micro state (for hook detection)
        prev_idx = micro_idx - 1
        if prev_idx < 0 or prev_idx >= len(micro_worker._states):
            return None
        prev_raw = micro_worker._states[prev_idx]
        prev_state = prev_raw['state'] if isinstance(prev_raw, dict) and 'state' in prev_raw else prev_raw
        prev_micro_adx = getattr(prev_state, 'adx_strength', 0.0)

        # Macro state  -- SIGNED z-score (positive = above mean, negative = below)
        macro_idx = macro_worker._last_tf_bar_idx
        if macro_idx < 0 or not macro_worker._states:
            return None
        macro_raw = macro_worker._states[min(macro_idx, len(macro_worker._states) - 1)]
        macro_state = macro_raw['state'] if isinstance(macro_raw, dict) and 'state' in macro_raw else macro_raw
        macro_z = getattr(macro_state, 'z_score', 0.0)  # SIGNED

        # Position-aware wall check:
        # Long: price at UPPER band (macro_z positive) -> favorable extreme, wall ahead
        # Short: price at LOWER band (macro_z negative) -> favorable extreme, floor ahead
        if pos.side == 'long':
            at_wall = macro_z >= self._macro_z_threshold
        else:
            at_wall = macro_z <= -self._macro_z_threshold

        # Death Hook: micro ADX was extreme and is now hooking down AT the macro wall
        if (micro_adx > self._micro_adx_threshold
                and micro_adx < prev_micro_adx    # ADX hooking down
                and at_wall):                      # at position-specific wall
            pnl_ticks = ((bar_close - pos.entry_price) / tick_size
                         if pos.side == 'long'
                         else (pos.entry_price - bar_close) / tick_size)
            return ExitResult(
                action=ExitAction.DEATH_HOOK,
                exit_price=bar_close,
                reason=f"Death hook: micro_adx={micro_adx:.1f}v macro_z={macro_z:+.1f} wall",
                pnl_ticks=pnl_ticks,
                bars_held=pos.bars_held,
                band_action='urgent',
            )

        return None
