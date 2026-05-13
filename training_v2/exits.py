"""Exit signal evaluators.

Each ExitRule reads (BarState, Position) and returns an exit reason string
or None. Engine evaluates rules in declaration order; first match closes
the position.

V2-native — all reads via state.get('L3_<tf>_z_se_<N>') etc., no V1 indices.

Adaptive thresholds:
    HardStop / TakeProfit / Giveback / TimeStop FIRST consult
    `position.extras['thresholds']` (a dict possibly populated by the engine
    via threshold_optimizer.lookup_thresholds(regime, tier)). If absent or a
    field is missing, fall back to the rule's __init__ default. Rules that
    aren't tied to per-cell thresholds (ZSeReversal, RegimeFlip,
    SwingNoiseSpike) keep their constructor configuration.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, List

from training_v2.state import BarState
from training_v2.ledger import Position
from training_v2.v2_cols import (z_se_w, swing_noise_w, price_velocity_w,
                                          reversion_prob_w)


# ── Tunables — Phase 2 baseline; recalibrate in Phase 3 ──────────────────

# Hard SL/TP in $ (per contract). MNQ: 1pt = $2; 1tick=$0.50.
HARD_STOP_USD = -25.0       # -$25 per contract
TAKE_PROFIT_USD = 60.0      # +$60 per contract
GIVEBACK_MIN_PEAK = 30.0    # arm giveback only if peak > $30
GIVEBACK_KEEP = 0.5         # keep 50% of peak; trigger if pnl < peak*0.5

# Time stop — 5s bars
MAX_HOLD_BARS = 360         # 30 min  (360 * 5s)

# z_se reversal — fade thesis dies when 1m z_se crosses 0
Z_REVERSAL_THRESHOLD = 0.0   # cross past 0 in opposite direction

# swing_noise spike — chop just started
SWING_NOISE_SPIKE_FACTOR = 2.0  # current > 2x entry value


class ExitRule(ABC):
    name: str = 'BASE'

    @abstractmethod
    def evaluate(self, state: BarState, position: Position) -> Optional[str]:
        """Return exit reason str if rule fires, else None."""


# Risk-management floor: optimizer is allowed to pick `sl_pts=999` on small
# samples (no path went deep red). Always enforce a hard cap regardless.
SL_PTS_FLOOR = 25.0     # ≈ -$50 — backstop for tail events not seen in IS


def _thr(position, key: str, default):
    thr = position.extras.get('thresholds') if position.extras else None
    if thr is None:
        return default
    return thr.get(key, default)


class HardStop(ExitRule):
    """Reads `sl_pts` (POINTS) from position.extras['thresholds']; converts
    to $; clamps to SL_PTS_FLOOR. Falls back to the constructor default $.
    """
    name = 'hard_stop'

    def __init__(self, usd: float = HARD_STOP_USD):
        self.usd = usd

    def evaluate(self, state, position):
        sl_pts = _thr(position, 'sl_pts', None)
        if sl_pts is not None:
            sl_pts = min(float(sl_pts), SL_PTS_FLOOR)
            cap_usd = -abs(sl_pts) * 2.0
        else:
            cap_usd = self.usd
        if position.pnl(state.price) <= cap_usd:
            return self.name
        return None


class TakeProfit(ExitRule):
    """Reads `tp_pts` (POINTS) from position.extras['thresholds']."""
    name = 'take_profit'

    def __init__(self, usd: float = TAKE_PROFIT_USD):
        self.usd = usd

    def evaluate(self, state, position):
        tp_pts = _thr(position, 'tp_pts', None)
        target_usd = (tp_pts * 2.0) if tp_pts is not None else self.usd
        if position.pnl(state.price) >= target_usd:
            return self.name
        return None


class Giveback(ExitRule):
    """Reads `gb_min` ($) and `gb_keep` (ratio) from position.extras['thresholds']."""
    name = 'giveback'

    def __init__(self, min_peak: float = GIVEBACK_MIN_PEAK,
                 keep: float = GIVEBACK_KEEP):
        self.min_peak = min_peak
        self.keep = keep

    def evaluate(self, state, position):
        min_peak = _thr(position, 'gb_min', self.min_peak)
        keep = _thr(position, 'gb_keep', self.keep)
        if position.peak_pnl < min_peak:
            return None
        if position.pnl(state.price) < position.peak_pnl * keep:
            return self.name
        return None


class TimeStop(ExitRule):
    """Reads `time_stop_bars` from position.extras['thresholds']."""
    name = 'time_stop'

    def __init__(self, max_bars: int = MAX_HOLD_BARS):
        self.max_bars = max_bars

    def evaluate(self, state, position):
        max_bars = _thr(position, 'time_stop_bars', self.max_bars)
        if position.bars_held >= int(max_bars):
            return self.name
        return None


class ZSeReversal(ExitRule):
    """1m mean-reversion thesis dies when z_se_w flips sign past 0
    in the direction opposite to the entry direction.

    For longs (we bet bounce up because z was negative), exit when z >= 0.
    For shorts (we bet drop because z was positive), exit when z <= 0.

    Skipped for trades flagged as RIDE (trend-follow thesis) — those entered
    in the SAME direction as the regime trend, so z is already on the
    "wrong side" at entry and this rule would fire on bar 1.

    Detection of RIDE: position.entry_tier in {'NMP_FLIP', 'MA_ALIGN'}
    or position.extras['flipped_from'] is present.
    """
    name = 'z_se_reversal'
    RIDE_TIERS = {'NMP_FLIP', 'MA_ALIGN', 'NMP_RIDE'}

    def __init__(self, tf: str = '1m'):
        self.col = z_se_w(tf)

    def evaluate(self, state, position):
        # Skip for ride/trend-follow trades — fade-thesis exit doesn't apply
        if position.entry_tier in self.RIDE_TIERS:
            return None
        if position.extras and position.extras.get('flipped_from'):
            return None
        z = state.get(self.col, 0.0)
        if position.direction == 'long' and z >= 0:
            return self.name
        if position.direction == 'short' and z <= 0:
            return self.name
        return None


class SwingNoiseSpike(ExitRule):
    """Chop just kicked in. Entry was on a smooth read; current bar's
    swing_noise is N-fold higher → exit before chop eats the trade.
    """
    name = 'swing_noise_spike'

    def __init__(self, tf: str = '1m', factor: float = SWING_NOISE_SPIKE_FACTOR):
        self.col = swing_noise_w(tf)
        self.factor = factor

    def evaluate(self, state, position):
        entry_sn = position.extras.get('entry_swing_noise')
        if entry_sn is None or entry_sn <= 0:
            return None
        current = state.get(self.col, 0.0)
        if current >= entry_sn * self.factor:
            return self.name
        return None


class RegimeFlip(ExitRule):
    """Macro-TF velocity flipped against the position's direction."""
    name = 'regime_flip'

    def __init__(self, tf: str = '4h'):
        self.col = price_velocity_w(tf)

    def evaluate(self, state, position):
        v = state.get(self.col, 0.0)
        if position.direction == 'long' and v < 0:
            return self.name
        if position.direction == 'short' and v > 0:
            return self.name
        return None


def default_exit_suite() -> List[ExitRule]:
    """Phase 3 default — order matters; first match wins."""
    return [
        HardStop(),
        TakeProfit(),
        Giveback(),
        ZSeReversal(tf='1m'),
        SwingNoiseSpike(tf='1m'),
        # RegimeFlip is opt-in; can be too aggressive on smooth-trend days
        TimeStop(),
    ]
