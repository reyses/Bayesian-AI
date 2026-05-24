"""RIDE_MOMENTUM — V2 port of legacy "CNN-flipped fade with high velocity".

Legacy: regime says flip, momentum is high → ride the trend that's
already moving fast.

V2 trigger:
    NMP seed
    AND (regime_idx, direction) in flip cells
    AND |L2_1m_price_velocity_w| >= MOMENTUM_VELOCITY

Direction: opposite of NMP fade (RIDE).
"""
from __future__ import annotations
from typing import Optional

from training.utils.state import BarState
from training.strategies.base import EntrySignal
from training.strategies._nmp_base import NMPBaseStrategy, NMPSeed
from training.strategies.regime_aware import DEFAULT_FLIP_CELLS
from training.utils.v2_cols import price_velocity_w


MOMENTUM_VELOCITY_DEFAULT = 5.0


class RideMomentum(NMPBaseStrategy):
    name = 'RIDE_MOMENTUM'

    def __init__(self, momentum_velocity: float = MOMENTUM_VELOCITY_DEFAULT, **kwargs):
        super().__init__(**kwargs)
        self.momentum_velocity = momentum_velocity
        self._vel_col = price_velocity_w('1m')

    def _qualify(self, state: BarState, seed: NMPSeed) -> Optional[EntrySignal]:
        if (int(state.regime_idx), seed.direction) not in DEFAULT_FLIP_CELLS:
            return None
        v = state.get(self._vel_col, 0.0)
        if abs(v) < self.momentum_velocity:
            return None
        new_dir = 'short' if seed.direction == 'long' else 'long'
        return EntrySignal(direction=new_dir, tier=self.name,
                              extras={'z_se': seed.z, 'velocity_1m': v,
                                          'flipped_from': seed.direction,
                                          'regime_2d': state.regime_2d})
