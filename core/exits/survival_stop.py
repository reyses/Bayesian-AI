"""Survival Probability Exit (Time-Stop)  -- exit when alpha has decayed.

Academic basis: Survival Analysis (Kaplan-Meier) + Bayesian Expected PnL.

Two exit modes (first match wins):
1. Bayesian ePnL: Brain computes expected PnL of holding for this (template, direction).
   When ePnL drops to 0 or below, the mathematical edge is gone  -- exit immediately.
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

        # ── Mode 1: State-based expected profit ──
        # Estimate remaining profit from CURRENT market conditions, not historical avg.
        # Uses: z-score (band position), Hurst (trend strength), conviction (TF agreement),
        # momentum alignment (forces with or against trade).
        #
        # Logic: P(remaining profit) estimated from current state.
        # If all signals say "more room" -> hold. If signals say "exhausted" -> exit.
        if belief_network is not None:
            _belief = belief_network.get_belief()
            _conviction = _belief.conviction if _belief else 0.5
            _aligned = (_belief.direction == pos.side) if _belief else True

            # Get current bar state from the discovery TF worker
            _disc_tf = int(getattr(pos, 'discovery_tf_seconds', 300))
            _worker = belief_network.workers.get(_disc_tf)
            _z = 0.0
            _hurst = 0.5
            _mom_with = True
            if _worker is not None and _worker._states:
                _mi = _worker._last_tf_bar_idx
                if 0 <= _mi < len(_worker._states):
                    _raw = _worker._states[_mi]
                    _ms = _raw['state'] if isinstance(_raw, dict) and 'state' in _raw else _raw
                    _z = getattr(_ms, 'z_score', 0.0)
                    _hurst = getattr(_ms, 'hurst_exponent', 0.5)
                    _f_mom = getattr(_ms, 'F_momentum', 0.0)
                    # Momentum aligned with trade?
                    _mom_with = (_f_mom > 0 and pos.side == 'long') or \
                                (_f_mom < 0 and pos.side == 'short')

            # Estimate remaining room:
            # For LONG: z approaching +2σ = near Roche limit = little room left
            # For SHORT: z approaching -2σ = near Roche limit
            if pos.side == 'long':
                _room = max(0, 2.0 - _z)  # how far to upper band (0 = at limit, 2+ = plenty)
            else:
                _room = max(0, 2.0 + _z)  # how far to lower band

            # Score: combine room + trend + conviction + alignment
            _trend_bonus = 1.0 if _hurst > 0.55 else (0.5 if _hurst < 0.45 else 0.75)
            _conv_bonus = _conviction  # 0.0 to 1.0
            _align_bonus = 1.0 if _aligned else 0.3
            _mom_bonus = 1.0 if _mom_with else 0.5

            _remain_score = (_room / 2.0) * _trend_bonus * _conv_bonus * _align_bonus * _mom_bonus

            # Exit when remaining score drops below threshold
            # Score < 0.1 means: near band limit + mean-reverting + low conviction + misaligned
            if _remain_score < 0.10:
                return ExitResult(
                    action=ExitAction.SURVIVAL_STOP,
                    exit_price=bar_close,
                    reason=f"State ePnL exit: remain={_remain_score:.3f} "
                           f"(room={_room:.2f} H={_hurst:.2f} conv={_conviction:.2f} "
                           f"align={'Y' if _aligned else 'N'} mom={'W' if _mom_with else 'A'})",
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
        # If macro agrees, this flat period is accumulation  -- hold.
        # Only exit flat trades when macro support is fading.
        _disc_tf = int(getattr(pos, 'discovery_tf_seconds', 300))
        _danger = self._check_flip_imminent(belief_network, _disc_tf, pos.side)
        if not _danger:
            return None  # macro still supports  -- flat is OK, hold

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
