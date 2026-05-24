"""FADE_CALM — V2-native port of the legacy base-NMP fade with low velocity.

Legacy (2026-04-08): "Fade z, low velocity" — 8,868 trades, $0.3/trade.
The catch-all NMP fade — fires when no other tier fires.

V2 trigger:
    NMP seed (|z_se_w| >= 1.8 + reversion_prob_w >= 0.55)
    AND |L2_1m_price_velocity_w| < CALM_VELOCITY

Direction: standard NMP fade.
"""
from __future__ import annotations
from typing import Optional

from training.utils.state import BarState
from training.strategies.base import EntrySignal
from training.strategies._nmp_base import NMPBaseStrategy, NMPSeed
from training.utils.v2_cols import price_velocity_w


CALM_VELOCITY_DEFAULT = 5.0  # V2 velocity_w units (recalibrated from V1 50)


class FadeCalm(NMPBaseStrategy):
    name = 'FADE_CALM'

    # ── 2026-05-10 RETUNE (validated 21,371 IS+OOS trades) ─────────────
    # E1 z-band: |z| in [1.5, 1.8]  → PF_WR 0.016 -> 0.096
    # E2 VETO  short × neutral (1h_z_se in [-0.3, +0.3])
    # Combined OOS uplift: +$280
    RETUNE_Z_LO = 1.5
    RETUNE_Z_HI = 1.8
    RETUNE_VETO = [('short', 'neutral')]

    def __init__(self, calm_velocity: float = CALM_VELOCITY_DEFAULT,
                  retune: bool = True, **kwargs):
        if retune:
            kwargs.setdefault('z_band_lo', self.RETUNE_Z_LO)
            kwargs.setdefault('z_band_hi', self.RETUNE_Z_HI)
            kwargs.setdefault('veto_cells', self.RETUNE_VETO)
            # Lower seed threshold to retune floor — otherwise the seed gates
            # at |z|>=1.8 and the retune ceiling at <=1.8 intersect at exactly
            # 1.8, firing ~0 trades. See docs/RETUNE_RESULTS_2026-05-10.md.
            kwargs.setdefault('z_threshold', self.RETUNE_Z_LO)
        super().__init__(**kwargs)
        self.calm_velocity = calm_velocity
        self._vel_col = price_velocity_w('1m')

    def _qualify(self, state: BarState, seed: NMPSeed) -> Optional[EntrySignal]:
        v = state.get(self._vel_col, 0.0)
        if abs(v) >= self.calm_velocity:
            return None
        return EntrySignal(direction=seed.direction, tier=self.name,
                              extras={'z_se': seed.z, 'velocity_1m': v})
