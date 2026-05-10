"""FADE_AGAINST — V2 port of legacy "NMP fade with 1h opposing".

Legacy (2026-04-09 journal): "1h direction overrides the fade" — flagged
POISON in iso form (-$14K IS / -$5K OOS) because following 1h against
the fade kills the trade. Kept here as a canonical port; expected to lose.

V2 trigger:
    NMP seed
    AND sign(L2_1h_price_velocity_w) is OPPOSITE the fade direction
        (i.e., for LONG fade: 1h velocity < 0; for SHORT fade: 1h velocity > 0)

Direction: NMP fade (we expect it to fail, per legacy verdict).
"""
from __future__ import annotations
from typing import Optional

from training_iso_v2.state import BarState
from training_iso_v2.strategies.base import EntrySignal
from training_iso_v2.strategies._nmp_base import NMPBaseStrategy, NMPSeed
from training_iso_v2.v2_cols import price_velocity_w


H1_VEL_THRESHOLD_DEFAULT = 0.0  # any opposing 1h velocity counts


class FadeAgainst(NMPBaseStrategy):
    name = 'FADE_AGAINST'

    def __init__(self, h1_vel_threshold: float = H1_VEL_THRESHOLD_DEFAULT, **kwargs):
        super().__init__(**kwargs)
        self.h1_vel_threshold = h1_vel_threshold
        self._h1_vel_col = price_velocity_w('1h')

    def _qualify(self, state: BarState, seed: NMPSeed) -> Optional[EntrySignal]:
        h1_v = state.get(self._h1_vel_col, 0.0)
        # OPPOSING: long fade with 1h going down, OR short fade with 1h going up
        opposing = ((seed.direction == 'long' and h1_v < -self.h1_vel_threshold)
                          or (seed.direction == 'short' and h1_v > self.h1_vel_threshold))
        if not opposing:
            return None
        return EntrySignal(direction=seed.direction, tier=self.name,
                              extras={'z_se': seed.z, 'h1_velocity': h1_v})
