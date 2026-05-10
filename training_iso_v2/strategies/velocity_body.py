"""Velocity-body chord strategy.

Source: 2026-05-03 EDA chord finding (memory: project_v2_features_eda_stack.md).
On 5m bars, when `price_velocity_1b` and `body` agree on sign and both exceed
small thresholds, the next-bar direction tracks. WR per quantile bin: 42-75%.
6-bar mono velocity → 70% WR / +38 ticks.

Rule:
    On a 5m close, if sign(price_velocity_1b_5m) == sign(body_5m) and both
    have non-trivial magnitude, enter in that direction.
"""
from __future__ import annotations

from typing import Optional

from training_iso_v2.state import BarState
from training_iso_v2.strategies.base import EntrySignal, Strategy
from training_iso_v2.v2_cols import price_velocity_1b, body


VEL_MIN_DEFAULT = 0.5     # ticks/sec proxy — recalibrate after first run
BODY_MIN_DEFAULT = 1.0    # price units


class VelocityBodyChord(Strategy):
    name = 'VEL_BODY_CHORD'

    def __init__(self, tf: str = '5m',
                 vel_min: float = VEL_MIN_DEFAULT,
                 body_min: float = BODY_MIN_DEFAULT,
                 fire_on: str = '5m'):
        self.tf = tf
        self.vel_min = vel_min
        self.body_min = body_min
        self.fire_on = fire_on
        self._vel_col = price_velocity_1b(tf)
        self._body_col = body(tf)

    def _ready(self, state: BarState) -> bool:
        return {
            '1m': state.is_1m_close,
            '5m': state.is_5m_close,
            '15m': state.is_15m_close,
        }.get(self.fire_on, False)

    def evaluate(self, state: BarState) -> Optional[EntrySignal]:
        if not self._ready(state):
            return None
        v = state.get(self._vel_col, 0.0)
        b = state.get(self._body_col, 0.0)

        if abs(v) < self.vel_min or abs(b) < self.body_min:
            return None

        if (v > 0) == (b > 0):
            direction = 'long' if v > 0 else 'short'
            return EntrySignal(direction=direction, tier=self.name,
                                  extras={'velocity_1b': v, 'body': b,
                                              'tf': self.tf})
        return None
