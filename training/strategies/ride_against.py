"""RIDE_AGAINST — V2 port of legacy "CNN-flipped + 1h opposing the fade".

Legacy (2026-04-09): SURPRISE WINNER. -$0.5/trade fade became +$24/trade
when flipped to ride the 1h-opposing direction. The 1h opposing is the
SIGNAL that the fade is wrong — flip and ride the 1h.

V2 trigger:
    NMP seed
    AND sign(L2_1h_price_velocity_w) opposes the fade direction
    (No regime cell filter — the 1h-opposing IS the regime signal here.)

Direction: opposite of fade — RIDE the 1h.
"""
from __future__ import annotations
from typing import Optional

from training.utils.state import BarState
from training.strategies.base import EntrySignal
from training.strategies._nmp_base import NMPBaseStrategy, NMPSeed
from training.utils.v2_cols import price_velocity_w


H1_VEL_THRESHOLD_DEFAULT = 0.0


class RideAgainst(NMPBaseStrategy):
    name = 'RIDE_AGAINST'

    def __init__(self, h1_vel_threshold: float = H1_VEL_THRESHOLD_DEFAULT, **kwargs):
        super().__init__(**kwargs)
        self.h1_vel_threshold = h1_vel_threshold
        self._h1_vel_col = price_velocity_w('1h')

    def _qualify(self, state: BarState, seed: NMPSeed) -> Optional[EntrySignal]:
        h1_v = state.get(self._h1_vel_col, 0.0)
        opposing = ((seed.direction == 'long' and h1_v < -self.h1_vel_threshold)
                          or (seed.direction == 'short' and h1_v > self.h1_vel_threshold))
        if not opposing:
            return None
        # Flip to ride 1h
        new_dir = 'short' if seed.direction == 'long' else 'long'
        return EntrySignal(direction=new_dir, tier=self.name,
                              extras={'z_se': seed.z, 'h1_velocity': h1_v,
                                          'flipped_from': seed.direction})
