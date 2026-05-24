"""COMPRESSION_BOUNCE_LONG — trend-side entry on volatility crush.

Source: 2026-05-10 full 90-feature scan finding.
Validated cell: L2_15m_vol_sigma_12 below -3sigma from native RM.
    P_up_OOS = 0.639 (n_oos = 95)
    Mean fwd return OOS = +$6.82
    PF_WR positive, IS+OOS sign-stable

Rule:
    At each 5s bar, track L2_15m_vol_sigma_12's native-cadence rolling mean
    over the last 12 native (15m) bars. When the current value is 3 std
    below this mean (sustained for >= min_dwell), emit a LONG signal.

Fires once per compression event (re-arms after a non-compression bar).
"""
from __future__ import annotations

from typing import Optional

from training.utils.state import BarState
from training.strategies.base import EntrySignal, Strategy
from training.filters.bayes_filters import CompressionBounce


class CompressionBounceLong(Strategy):
    """LONG when 15m vol_sigma is crushed below its 3-hr rolling RM by 3 sigma."""
    name = 'COMPRESSION_BOUNCE_LONG'

    def __init__(self, min_dwell_bars: int = 12):
        """min_dwell_bars: require LONG_BIAS state for at least this many
        5s bars before firing (= 60s @ 12 bars). Filters out transient dips."""
        self.detector = CompressionBounce()
        self.min_dwell_bars = min_dwell_bars
        self._dwell_count = 0
        self._fired_this_event = False
        self._current_day = None

    def reset(self):
        self.detector = CompressionBounce()
        self._dwell_count = 0
        self._fired_this_event = False
        self._current_day = None

    def evaluate(self, state: BarState) -> Optional[EntrySignal]:
        if state.day != self._current_day:
            self.detector.reset(state.day)
            self._dwell_count = 0
            self._fired_this_event = False
            self._current_day = state.day

        vol_sigma_15m = state.get('L2_15m_vol_sigma_12', float('nan'))
        comp_state = self.detector.update(vol_sigma_15m)

        if comp_state != 'LONG_BIAS':
            self._dwell_count = 0
            self._fired_this_event = False
            return None

        self._dwell_count += 1
        if self._fired_this_event:
            return None
        if self._dwell_count < self.min_dwell_bars:
            return None

        # Fire LONG once per compression event
        self._fired_this_event = True
        return EntrySignal(
            direction='long', tier=self.name,
            extras={'vol_sigma_15m': float(vol_sigma_15m),
                    'dwell_bars': self._dwell_count})
