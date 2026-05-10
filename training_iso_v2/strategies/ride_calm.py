"""RIDE_CALM — V2 port of legacy "CNN-flipped fade with low velocity".

Legacy: NMP entry, CNN flip predictor said COUNTER, low velocity at entry.
The "ride a slow trend" version of NMP-flip.

V2 substitute for CNN flip: regime-direction flip rule (the categorical
splitter we already validated). Flip when (regime, direction) is a known
flip cell (UP_*, DOWN_SMOOTH respective directions).

V2 trigger:
    NMP seed
    AND (regime_idx, direction) in DEFAULT_FLIP_CELLS
    AND |L2_1m_price_velocity_w| < CALM_VELOCITY

Direction: opposite of NMP fade (RIDE).
"""
from __future__ import annotations
from typing import Optional

from training_iso_v2.state import BarState
from training_iso_v2.strategies.base import EntrySignal
from training_iso_v2.strategies._nmp_base import NMPBaseStrategy, NMPSeed
from training_iso_v2.strategies.regime_aware import DEFAULT_FLIP_CELLS
from training_iso_v2.v2_cols import price_velocity_w


CALM_VELOCITY_DEFAULT = 5.0


class RideCalm(NMPBaseStrategy):
    name = 'RIDE_CALM'

    def __init__(self, calm_velocity: float = CALM_VELOCITY_DEFAULT, **kwargs):
        super().__init__(**kwargs)
        self.calm_velocity = calm_velocity
        self._vel_col = price_velocity_w('1m')

    def _qualify(self, state: BarState, seed: NMPSeed) -> Optional[EntrySignal]:
        # Only fire if (regime, original_direction) is a flip cell — the regime
        # supports flipping THIS direction.
        if (int(state.regime_idx), seed.direction) not in DEFAULT_FLIP_CELLS:
            return None
        v = state.get(self._vel_col, 0.0)
        if abs(v) >= self.calm_velocity:
            return None
        # Flip direction
        new_dir = 'short' if seed.direction == 'long' else 'long'
        return EntrySignal(direction=new_dir, tier=self.name,
                              extras={'z_se': seed.z, 'velocity_1m': v,
                                          'flipped_from': seed.direction,
                                          'regime_2d': state.regime_2d})
