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
        # If brain has enough observations for this template+direction,
        # check if expected PnL of holding has dropped to zero or below.
        brain = getattr(self, '_brain', None)
        if brain is not None:
            epnl = brain.get_expected_pnl(pos.template_id, pos.side.upper())
            if epnl is not None and epnl <= self._epnl_threshold:
                return ExitResult(
                    action=ExitAction.SURVIVAL_STOP,
                    exit_price=bar_close,
                    reason=f"Bayesian ePnL exit: ePnL={epnl:.2f} <= {self._epnl_threshold:.2f} "
                           f"for tid={pos.template_id} {pos.side}",
                    pnl_ticks=current_ticks,
                    bars_held=pos.bars_held,
                )

        # ── Mode 2: Structural flatline (fallback) ──
        target_ticks = pos.tp_ticks if pos.tp_ticks > 0 else pos.anchor_mfe_ticks
        if target_ticks <= 0:
            return None

        if current_ticks >= target_ticks * self._target_pct:
            return None  # on track

        z_var = self._compute_z_variance(belief_network)
        if z_var is None or z_var >= self._z_var_max:
            return None  # still moving

        return ExitResult(
            action=ExitAction.SURVIVAL_FLATLINE,
            exit_price=bar_close,
            reason=f"Survival flatline: {pos.bars_held}bars, "
                   f"pnl={current_ticks:.1f}t < {target_ticks*self._target_pct:.1f}t target, "
                   f"z_var={z_var:.3f}",
            pnl_ticks=current_ticks,
            bars_held=pos.bars_held,
        )

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
