"""FREIGHT_TRAIN — V2 port of legacy "extreme velocity in compressed regime".

Legacy (2026-04-09): originally `|vel|>100 + accel + vr<0.5` interpreted as
"ride the velocity" (5 trades, 20% WR). User caught the inversion: extreme
velocity in compressed (low-noise) regime is EXHAUSTION, not continuation —
fade it. Direction was flipped; verdict still negative with V1 data.

V2 trigger (no NMP seed required — this is a STANDALONE entry):
    |L2_1m_price_velocity_w| >= EXTREME_VELOCITY
    AND L3_1m_swing_noise_15 <= LOW_NOISE_THRESHOLD
    AND L3_1m_hurst_15 <= HURST_REVERTING (regime is mean-reverting per Hurst)

Direction: FADE the velocity (sign-flip), per the 2026-04-18 inversion.
"""
from __future__ import annotations
from typing import Optional

from training_iso_v2.state import BarState
from training_iso_v2.strategies.base import EntrySignal, Strategy
from training_iso_v2.v2_cols import (price_velocity_w, swing_noise_w, hurst_w)


EXTREME_VELOCITY_DEFAULT = 10.0     # V2 velocity_w units; recalibrate
LOW_NOISE_DEFAULT = 100.0           # below this = compressed regime
HURST_REVERT_DEFAULT = 0.50         # below 0.5 = mean-reverting per Hurst


class FreightTrain(Strategy):
    name = 'FREIGHT_TRAIN'

    def __init__(self,
                 extreme_velocity: float = EXTREME_VELOCITY_DEFAULT,
                 low_noise: float = LOW_NOISE_DEFAULT,
                 hurst_revert: float = HURST_REVERT_DEFAULT,
                 fire_on: str = '1m'):
        self.extreme_velocity = extreme_velocity
        self.low_noise = low_noise
        self.hurst_revert = hurst_revert
        self.fire_on = fire_on
        self._vel_col = price_velocity_w('1m')
        self._noise_col = swing_noise_w('1m')
        self._hurst_col = hurst_w('1m')

    def _ready(self, state: BarState) -> bool:
        return {
            '1m': state.is_1m_close,
            '5m': state.is_5m_close,
        }.get(self.fire_on, False)

    def evaluate(self, state: BarState) -> Optional[EntrySignal]:
        if not self._ready(state):
            return None
        v = state.get(self._vel_col, 0.0)
        if abs(v) < self.extreme_velocity:
            return None
        sn = state.get(self._noise_col, 1e9)
        if sn > self.low_noise:
            return None
        h = state.get(self._hurst_col, 1.0)
        if h > self.hurst_revert:
            return None
        # FADE the velocity: if going up fast, short; if going down fast, long
        direction = 'short' if v > 0 else 'long'
        return EntrySignal(direction=direction, tier=self.name,
                              extras={'velocity_1m': v, 'swing_noise': sn,
                                          'hurst': h})
