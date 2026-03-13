"""Regime Decay Exit (Sand Trap) — exit when macro trend collapses or DI reverses.

Academic basis: Markov Regime Switching.
A strategy designed for State A (trending) has negative expectancy in State B (chop).
When the macro gravitational field collapses, cut immediately.

Also detects DI crossover against the trade — a direct trend reversal signal.
If DI+ crosses below DI- during a long (or vice versa), the trend has flipped.

Logic:
  Regime collapse:
    IF macro ADX drops below 20 → thesis invalidated, FORCE_EXIT
  DI trend reversal:
    IF DI crosses against trade direction → trend reversed, FORCE_EXIT
"""
from typing import Optional

from core.exit_engine import ExitAction, ExitResult, PositionState


class RegimeDecayExit:

    def __init__(self, config=None):
        if config is None:
            from core.trading_config import TradingConfig
            config = TradingConfig()
        self._adx_threshold = config.ce_regime_decay_adx
        self._di_cross_enabled = config.ce_regime_decay_di_cross
        self._macro_tf = config.fdmi_macro_tf
        self._hurst_exit = config.hurst_regime_exit

    def evaluate(self, pos: PositionState, bar_close: float, tick_size: float,
                 belief_network=None) -> Optional[ExitResult]:
        """Check regime decay / DI trend reversal."""
        if belief_network is None:
            return None

        macro_worker = belief_network.workers.get(self._macro_tf)
        if macro_worker is None:
            return None

        # Get current and previous macro state
        macro_idx = macro_worker._last_tf_bar_idx
        if macro_idx < 1 or not macro_worker._states:
            return None

        raw = macro_worker._states[min(macro_idx, len(macro_worker._states) - 1)]
        macro_state = raw['state'] if isinstance(raw, dict) and 'state' in raw else raw
        macro_adx = getattr(macro_state, 'adx_strength', 0.0)
        macro_di_plus = getattr(macro_state, 'dmi_plus', 0.0)
        macro_di_minus = getattr(macro_state, 'dmi_minus', 0.0)

        prev_idx = macro_idx - 1
        if prev_idx < 0 or prev_idx >= len(macro_worker._states):
            return None
        prev_raw = macro_worker._states[prev_idx]
        prev_state = prev_raw['state'] if isinstance(prev_raw, dict) and 'state' in prev_raw else prev_raw
        prev_di_plus = getattr(prev_state, 'dmi_plus', 0.0)
        prev_di_minus = getattr(prev_state, 'dmi_minus', 0.0)
        prev_adx = getattr(prev_state, 'adx_strength', 0.0)

        pnl_ticks = ((bar_close - pos.entry_price) / tick_size
                     if pos.side == 'long'
                     else (pos.entry_price - bar_close) / tick_size)

        # ── Check 0: Hurst regime shift (academic primary) ──
        # Hurst dropping below 0.50 = trend memory has died mathematically
        macro_hurst = getattr(macro_state, 'hurst_exponent', 0.5)
        prev_hurst = getattr(prev_state, 'hurst_exponent', 0.5)
        if macro_hurst < self._hurst_exit and prev_hurst >= self._hurst_exit:
            return ExitResult(
                action=ExitAction.REGIME_DECAY,
                exit_price=bar_close,
                reason=f"Hurst regime shift: H={macro_hurst:.3f} collapsed from {prev_hurst:.3f}",
                pnl_ticks=pnl_ticks,
                bars_held=pos.bars_held,
            )

        # ── Check 1: Regime collapse (ADX drops below threshold — lagging confirmation) ──
        # Only trigger if ADX was previously above threshold (actual collapse, not always-low)
        if macro_adx < self._adx_threshold and prev_adx >= self._adx_threshold:
            return ExitResult(
                action=ExitAction.REGIME_DECAY,
                exit_price=bar_close,
                reason=f"Regime decay: macro_adx={macro_adx:.1f} collapsed from {prev_adx:.1f}",
                pnl_ticks=pnl_ticks,
                bars_held=pos.bars_held,
            )

        # ── Check 2: DI crossover against trade (trend reversal) ──
        if self._di_cross_enabled and pos.bars_held >= 3:
            if pos.side == 'long':
                # Was bullish (DI+ > DI-), now bearish (DI- >= DI+)
                crossed_against = (prev_di_plus > prev_di_minus
                                   and macro_di_minus >= macro_di_plus)
            else:
                # Was bearish (DI- > DI+), now bullish (DI+ >= DI-)
                crossed_against = (prev_di_minus > prev_di_plus
                                   and macro_di_plus >= macro_di_minus)

            if crossed_against:
                return ExitResult(
                    action=ExitAction.REGIME_DECAY,
                    exit_price=bar_close,
                    reason=f"DI reversal: macro DI crossed against {pos.side}",
                    pnl_ticks=pnl_ticks,
                    bars_held=pos.bars_held,
                )

        return None
