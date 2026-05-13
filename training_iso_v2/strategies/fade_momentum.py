"""FADE_MOMENTUM — V2 port of legacy "Fade z, high velocity" (freight-train flavor).

Legacy (2026-04-08): 112 trades, $16.7/trade. Distinct from FADE_CALM by
velocity at entry — strong momentum suggests imminent exhaustion.

V2 trigger:
    NMP seed
    AND |L2_1m_price_velocity_w| >= MOMENTUM_VELOCITY

Direction: NMP fade.
"""
from __future__ import annotations
from typing import Optional

from training_iso_v2.state import BarState
from training_iso_v2.strategies.base import EntrySignal
from training_iso_v2.strategies._nmp_base import NMPBaseStrategy, NMPSeed
from training_iso_v2.v2_cols import price_velocity_w


MOMENTUM_VELOCITY_DEFAULT = 5.0


class FadeMomentum(NMPBaseStrategy):
    name = 'FADE_MOMENTUM'

    # ── 2026-05-10 RETUNE (validated 12,766 IS+OOS trades) ─────────────
    # E1 z-band: |z| in [1.5, 1.8]  → PF_WR 0.038 -> 0.153
    # No VETO cells found (no structural-loser direction-regime combo)
    # OOS uplift: +$128
    RETUNE_Z_LO = 1.5
    RETUNE_Z_HI = 1.8

    def __init__(self, momentum_velocity: float = MOMENTUM_VELOCITY_DEFAULT,
                  retune: bool = True, **kwargs):
        if retune:
            kwargs.setdefault('z_band_lo', self.RETUNE_Z_LO)
            kwargs.setdefault('z_band_hi', self.RETUNE_Z_HI)
            kwargs.setdefault('z_threshold', self.RETUNE_Z_LO)
        super().__init__(**kwargs)
        self.momentum_velocity = momentum_velocity
        self._vel_col = price_velocity_w('1m')

    def _qualify(self, state: BarState, seed: NMPSeed) -> Optional[EntrySignal]:
        v = state.get(self._vel_col, 0.0)
        if abs(v) < self.momentum_velocity:
            return None
        return EntrySignal(direction=seed.direction, tier=self.name,
                              extras={'z_se': seed.z, 'velocity_1m': v})
