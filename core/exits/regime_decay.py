"""Regime Decay Exit — exit when discovery TF's trend collapses.

Uses the trade's discovery TF for exit signal (not hardcoded 5m).
Adjacent higher TF provides hold override: if macro trend still alive,
suppress regime_decay (the chop is noise from the demi-gods' PID).

Logic:
  1. Hurst regime shift: H drops below threshold on discovery TF
  2. ADX collapse: ADX drops below 20 on discovery TF
  3. DI crossover: DI crosses against trade on discovery TF

  ALL checks suppressed if higher TF DMI still agrees with trade direction.
  Research: depth 7-9 regime_decay was -$5.22/trade (37% WR) because 5m ADX
  oscillations killed trades while 30m/1h trend was intact.
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
        self._macro_tf = config.fdmi_macro_tf  # fallback if no discovery TF
        self._hurst_exit = config.hurst_regime_exit

    def evaluate(self, pos: PositionState, bar_close: float, tick_size: float,
                 belief_network=None) -> Optional[ExitResult]:
        """Check regime decay using discovery TF, with higher TF override."""
        if belief_network is None:
            return None

        # Use trade's discovery TF worker, fallback to config macro TF
        _disc_tf = int(getattr(pos, 'discovery_tf_seconds', self._macro_tf))
        macro_worker = belief_network.workers.get(_disc_tf)
        if macro_worker is None:
            # Fallback to nearest available worker
            macro_worker = belief_network.workers.get(self._macro_tf)
        if macro_worker is None:
            return None

        # Get current and previous state from discovery TF
        macro_idx = macro_worker._last_tf_bar_idx
        if macro_idx < 1 or not macro_worker._states:
            return None

        raw = macro_worker._states[min(macro_idx, len(macro_worker._states) - 1)]
        macro_state = raw['state'] if isinstance(raw, dict) and 'state' in raw else raw
        macro_adx = getattr(macro_state, 'adx_strength', 0.0)
        macro_di_plus = getattr(macro_state, 'dmi_plus', 0.0)
        macro_di_minus = getattr(macro_state, 'dmi_minus', 0.0)
        macro_hurst = getattr(macro_state, 'hurst_exponent', 0.5)

        prev_idx = macro_idx - 1
        if prev_idx < 0 or prev_idx >= len(macro_worker._states):
            return None
        prev_raw = macro_worker._states[prev_idx]
        prev_state = prev_raw['state'] if isinstance(prev_raw, dict) and 'state' in prev_raw else prev_raw
        prev_di_plus = getattr(prev_state, 'dmi_plus', 0.0)
        prev_di_minus = getattr(prev_state, 'dmi_minus', 0.0)
        prev_adx = getattr(prev_state, 'adx_strength', 0.0)
        prev_hurst = getattr(prev_state, 'hurst_exponent', 0.5)

        pnl_ticks = ((bar_close - pos.entry_price) / tick_size
                     if pos.side == 'long'
                     else (pos.entry_price - bar_close) / tick_size)

        # ── Higher TF override: check adjacent higher TF ──
        # If the macro trend (next TF up) still agrees with trade direction,
        # suppress regime_decay. The chop on the discovery TF is micro noise.
        # ADAPTIVE: never-profitable trades (peak < 2t after 4+ bars) skip this
        # override — there's nothing to protect, let regime_decay fire.
        if pos.side == 'long':
            _peak_t = (pos.peak_favorable - pos.entry_price) / tick_size
        else:
            _peak_t = (pos.entry_price - pos.peak_favorable) / tick_size
        _never_profitable = _peak_t < 2.0 and pos.bars_held >= 4

        _higher_tf_agrees = False
        if not _never_profitable:
            _higher_tf_agrees = self._check_higher_tf(
                belief_network, _disc_tf, pos.side)
        if _higher_tf_agrees:
            return None  # macro trend alive — ride the chop

        # ── Check 0: Hurst regime shift ──
        if macro_hurst < self._hurst_exit and prev_hurst >= self._hurst_exit:
            return ExitResult(
                action=ExitAction.REGIME_DECAY,
                exit_price=bar_close,
                reason=f"Hurst regime shift: H={macro_hurst:.3f} "
                       f"collapsed from {prev_hurst:.3f} (TF={_disc_tf}s)",
                pnl_ticks=pnl_ticks,
                bars_held=pos.bars_held,
            )

        # ── Check 1: ADX collapse ──
        if macro_adx < self._adx_threshold and prev_adx >= self._adx_threshold:
            return ExitResult(
                action=ExitAction.REGIME_DECAY,
                exit_price=bar_close,
                reason=f"Regime decay: ADX={macro_adx:.1f} collapsed "
                       f"from {prev_adx:.1f} (TF={_disc_tf}s)",
                pnl_ticks=pnl_ticks,
                bars_held=pos.bars_held,
            )

        # ── Check 2: DI crossover against trade ──
        # Research: DMI crossover is unreliable below 3m (MFE/MAE = 1.0x).
        # Only check DI cross for discovery TFs >= 180s (3m).
        # Below that, ADX collapse and Hurst shift (checks 0+1) handle exits.
        _di_cross_min_tf = 180  # 3m — below this, DI cross is noise
        if (self._di_cross_enabled and pos.bars_held >= 3
                and _disc_tf >= _di_cross_min_tf):
            if pos.side == 'long':
                crossed_against = (prev_di_plus > prev_di_minus
                                   and macro_di_minus >= macro_di_plus)
            else:
                crossed_against = (prev_di_minus > prev_di_plus
                                   and macro_di_plus >= macro_di_minus)

            if crossed_against:
                return ExitResult(
                    action=ExitAction.REGIME_DECAY,
                    exit_price=bar_close,
                    reason=f"DI reversal: DI crossed against "
                           f"{pos.side} (TF={_disc_tf}s)",
                    pnl_ticks=pnl_ticks,
                    bars_held=pos.bars_held,
                )

        return None

    @staticmethod
    def _check_higher_tf(belief_network, disc_tf_sec: int, side: str) -> bool:
        """Check if the adjacent higher TF's DMI still agrees with trade."""
        _hierarchy = [1, 5, 15, 30, 60, 120, 180, 300, 900, 1800, 3600, 14400]
        _higher = disc_tf_sec  # fallback
        for tf in _hierarchy:
            if tf > disc_tf_sec:
                _higher = tf
                break

        w = belief_network.workers.get(_higher)
        if w is None:
            return False  # can't check → don't suppress

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
