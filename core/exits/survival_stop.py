"""Survival Probability Exit (Time-Stop) — exit when alpha has decayed.

Academic basis: Survival Analysis (Kaplan-Meier) + Bayesian Expected PnL.

Two exit modes (first match wins):
1. Bayesian ePnL: Brain computes expected PnL of holding for this (template, direction).
   When ePnL drops to 0 or below, the mathematical edge is gone — exit immediately.
   This replaces arbitrary trailing stops with continuous Bayesian inference.

2. Structural flatline: bars_held > N, PnL < 50% of target, Z variance < 0.2 (flatlining).
   This catches trades where the brain has insufficient data (< 3 obs) to compute ePnL.
"""
import numpy as np
from typing import Optional

from core.exit_engine import ExitAction, ExitResult, PositionState


class SurvivalStopExit:

    def __init__(self, config=None):
        if config is None:
            from core.trading_config import TradingConfig
            config = TradingConfig()
        self._min_bars = config.ce_survival_min_bars
        self._target_pct = config.ce_survival_target_pct
        self._z_var_max = config.ce_survival_z_var_max
        self._lookback = config.ce_survival_lookback
        self._micro_tf = config.fdmi_micro_tf
        self._epnl_min_obs = config.epnl_exit_min_obs
        self._epnl_threshold = config.epnl_exit_threshold

    def set_brain(self, brain):
        """Attach Bayesian brain for ePnL computation (called by ExitEngine)."""
        self._brain = brain

    def evaluate(self, pos: PositionState, bar_close: float, tick_size: float,
                 belief_network=None) -> Optional[ExitResult]:
        """Check survival: Bayesian ePnL first, then structural flatline."""
        # Must be in trade long enough for this to matter
        if pos.bars_held < self._min_bars:
            return None

        if pos.side == 'long':
            current_ticks = (bar_close - pos.entry_price) / tick_size
        else:
            current_ticks = (pos.entry_price - bar_close) / tick_size

        # ── Breakeven gate: only exit if trade is at or above breakeven ──
        # If underwater, let stop_loss handle it (tighter loss cap).
        if current_ticks < 0:
            return None

        # ── Mode 1: Bayesian ePnL (primary — continuous inference) ──
        # Brain's historical ePnL modulated by CURRENT conviction.
        # High conviction = workers agree with trade = hold longer (scale up ePnL).
        # Low conviction = workers disagree = exit sooner (scale down ePnL).
        # This prevents the brain from overriding a live signal that says "hold."
        brain = getattr(self, '_brain', None)
        if brain is not None:
            epnl = brain.get_expected_pnl(pos.template_id, pos.side.upper())
            if epnl is not None:
                # Get current conviction from belief network
                _conviction = 0.5
                if belief_network is not None:
                    _belief = belief_network.get_belief()
                    if _belief is not None:
                        _conviction = _belief.conviction
                        # Direction alignment bonus: if belief agrees with trade, boost
                        if _belief.direction == pos.side:
                            _conviction = min(1.0, _conviction * 1.2)
                        else:
                            _conviction = _conviction * 0.6  # penalize disagreement

                # Scale ePnL by conviction: high conviction = harder to trigger exit
                _conv_scale = 0.5 + _conviction  # range: 0.5 (low conv) to 1.5 (high conv)
                _effective_epnl = epnl * _conv_scale

                if _effective_epnl <= self._epnl_threshold:
                    return ExitResult(
                        action=ExitAction.SURVIVAL_STOP,
                        exit_price=bar_close,
                        reason=f"Bayesian ePnL exit: ePnL={epnl:.2f}*conv={_conv_scale:.2f}"
                               f"={_effective_epnl:.2f} <= {self._epnl_threshold:.2f} "
                               f"for tid={pos.template_id} {pos.side}",
                        pnl_ticks=current_ticks,
                        bars_held=pos.bars_held,
                    )

        # ── Mode 2: Structural flatline (fallback) ──
        # Only fires if there's evidence of imminent reversal.
        # A flat trade with macro support = accumulation, not danger.
        target_ticks = pos.tp_ticks if pos.tp_ticks > 0 else pos.anchor_mfe_ticks
        if target_ticks <= 0:
            return None

        if current_ticks >= target_ticks * self._target_pct:
            return None  # on track

        z_var = self._compute_z_variance(belief_network)
        if z_var is None or z_var >= self._z_var_max:
            return None  # still moving

        # Check if higher TF still supports the trade.
        # If macro agrees, this flat period is accumulation — hold.
        # Only exit flat trades when macro support is fading.
        _disc_tf = int(getattr(pos, 'discovery_tf_seconds', 300))
        _danger = self._check_flip_imminent(belief_network, _disc_tf, pos.side)
        if not _danger:
            return None  # macro still supports — flat is OK, hold

        return ExitResult(
            action=ExitAction.SURVIVAL_FLATLINE,
            exit_price=bar_close,
            reason=f"Survival flatline: {pos.bars_held}bars, "
                   f"pnl={current_ticks:.1f}t < {target_ticks*self._target_pct:.1f}t target, "
                   f"z_var={z_var:.3f}, flip_imminent",
            pnl_ticks=current_ticks,
            bars_held=pos.bars_held,
        )

    @staticmethod
    def _check_flip_imminent(belief_network, disc_tf_sec: int, side: str) -> bool:
        """Check if the adjacent higher TF is turning against the trade."""
        if belief_network is None:
            return True  # no data = assume danger

        _hierarchy = [1, 5, 15, 30, 60, 120, 180, 300, 900, 1800, 3600, 14400]
        _higher = disc_tf_sec
        for tf in _hierarchy:
            if tf > disc_tf_sec:
                _higher = tf
                break

        w = belief_network.workers.get(_higher)
        if w is None:
            return True  # can't check = assume danger

        mi = w._last_tf_bar_idx
        if mi < 1 or not w._states or mi >= len(w._states):
            return True

        # Current and previous state
        raw = w._states[mi]
        ms = raw['state'] if isinstance(raw, dict) and 'state' in raw else raw
        dp = getattr(ms, 'dmi_plus', 0.0)
        dm = getattr(ms, 'dmi_minus', 0.0)

        prev_raw = w._states[mi - 1]
        prev_ms = prev_raw['state'] if isinstance(prev_raw, dict) and 'state' in prev_raw else prev_raw
        prev_dp = getattr(prev_ms, 'dmi_plus', 0.0)
        prev_dm = getattr(prev_ms, 'dmi_minus', 0.0)

        # Danger = higher TF DMI is crossing or has crossed against
        if side == 'long':
            # Was bullish, now bearish or closing gap
            agrees = dp > dm
            was_agree = prev_dp > prev_dm
            gap_shrinking = (dp - dm) < (prev_dp - prev_dm) * 0.5
        else:
            agrees = dm > dp
            was_agree = prev_dm > prev_dp
            gap_shrinking = (dm - dp) < (prev_dm - prev_dp) * 0.5

        # Flip imminent: either already flipped or gap halved
        if not agrees:
            return True  # already flipped
        if was_agree and gap_shrinking:
            return True  # gap closing fast

        return False  # macro still solid

    def _compute_z_variance(self, belief_network) -> Optional[float]:
        """Compute Z-score variance over last N micro bars."""
        if belief_network is None:
            return None

        micro_worker = belief_network.workers.get(self._micro_tf)
        if micro_worker is None:
            return None

        idx = micro_worker._last_tf_bar_idx
        if idx < self._lookback or not micro_worker._states:
            return None

        z_scores = []
        for i in range(max(0, idx - self._lookback + 1), idx + 1):
            if i >= len(micro_worker._states):
                continue
            raw = micro_worker._states[i]
            state = raw['state'] if isinstance(raw, dict) and 'state' in raw else raw
            z_scores.append(getattr(state, 'z_score', 0.0))

        if len(z_scores) < 3:
            return None

        return float(np.var(z_scores))
