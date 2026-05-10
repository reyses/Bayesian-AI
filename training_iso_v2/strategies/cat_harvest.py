"""CAT_HARVEST_RIDE — pre-position SHORT during known danger windows.

Source: 2026-05-10 TOD/DOW analysis.
Catastrophic events (|max_z|>=6 at 1h HL band) are 96% CRASHES.
P(cat in next 60m) peaks at known UTC hours per DOW:
    Tue UTC 1:  40% (Tier-1)
    Wed UTC 1:  32%
    Thu UTC 1:  24%
    + UTC 2, 13, 14 in various DOWs

Rule:
    During a CAT_HARVEST window (per CAT_HARVEST_WINDOWS in bayes_filters),
    emit a SHORT signal ONCE per window. Skip if a SHORT is already open.

Fires at the start of each window. Exits handled by the engine's
standard exit suite (BayesConditionalExit + time stop).
"""
from __future__ import annotations

from typing import Optional

from training_iso_v2.state import BarState
from training_iso_v2.strategies.base import EntrySignal, Strategy
from training_iso_v2.filters.bayes_filters import cat_harvest_signal


class CatHarvestRide(Strategy):
    """Pre-position SHORT during known cat-harvest windows.

    Fires once per window-entry. Re-arms when out of window."""
    name = 'CAT_HARVEST'

    def __init__(self):
        self._in_window = False
        self._fired_this_window = False

    def reset(self):
        self._in_window = False
        self._fired_this_window = False

    def evaluate(self, state: BarState) -> Optional[EntrySignal]:
        sig = cat_harvest_signal(state.timestamp)
        now_in = (sig is not None)

        if not now_in:
            # Exited window — re-arm
            self._in_window = False
            self._fired_this_window = False
            return None

        # In window — fire once
        if self._fired_this_window:
            self._in_window = True
            return None

        self._fired_this_window = True
        self._in_window = True
        return EntrySignal(
            direction='short', tier=self.name,
            extras={'window_signal': sig,
                    'utc_hour': int((state.timestamp // 3600) % 24)})
