"""Reversion-from-extreme strategy.

Source: V2 EDA — `z_se_w` + `reversion_prob_w` are the V2-native bands signal.
Logic mirrors the legacy NMP intent (z extreme → bet bounce), but reads V2
features directly instead of the V1-derived 91D vector.

Rule:
    If 1m |z_se_w| >= Z_THRESHOLD AND 1m reversion_prob_w >= R_THRESHOLD:
        z > 0 → SHORT (price stretched above; bet drop)
        z < 0 → LONG  (price stretched below; bet bounce)

Fires on 1m closes (1m-cadence signal).
"""
from __future__ import annotations

from typing import Optional

from training.utils.state import BarState
from training.strategies.base import EntrySignal, Strategy
from training.utils.v2_cols import z_se_w, reversion_prob_w


Z_THRESHOLD_DEFAULT = 1.8
R_THRESHOLD_DEFAULT = 0.55


class ReversionFromExtreme(Strategy):
    name = 'REVERSION'

    def __init__(self, tf: str = '1m',
                 z_threshold: float = Z_THRESHOLD_DEFAULT,
                 r_threshold: float = R_THRESHOLD_DEFAULT,
                 fire_on: str = '1m'):
        self.tf = tf
        self.z_threshold = z_threshold
        self.r_threshold = r_threshold
        self.fire_on = fire_on
        self._z_col = z_se_w(tf)
        self._r_col = reversion_prob_w(tf)

    def _ready(self, state: BarState) -> bool:
        return {
            '1m': state.is_1m_close,
            '5m': state.is_5m_close,
            '15m': state.is_15m_close,
        }.get(self.fire_on, False)

    def evaluate(self, state: BarState) -> Optional[EntrySignal]:
        if not self._ready(state):
            return None
        z = state.get(self._z_col, 0.0)
        r = state.get(self._r_col, 0.0)
        if abs(z) < self.z_threshold or r < self.r_threshold:
            return None
        direction = 'short' if z > 0 else 'long'
        return EntrySignal(direction=direction, tier=self.name,
                              extras={'z_se': z, 'reversion_prob': r,
                                          'tf': self.tf})
