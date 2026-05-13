"""Diagnostic baseline tiers — used to test whether downstream filters help.

NMP_FADE_RAW : pure NMP seed, fade direction, no velocity/wick filter
NMP_RIDE_RAW : NMP seed + regime-flip-cell membership ONLY (ride direction),
                no velocity filter

Compare these against the 9 filtered tiers in the iso pipeline:
    - If NMP_FADE_RAW $/day >> sum(FADE_*, KILL_SHOT, CASCADE, FADE_AGAINST)
      $/day, the velocity / wick filters are subtractive.
    - If NMP_RIDE_RAW $/day >> sum(RIDE_*) $/day, the velocity calm/momentum
      split is subtractive.

This is a DIAGNOSTIC, not a deploy candidate.
"""
from __future__ import annotations
from typing import Optional

from training_iso_v2.state import BarState
from training_iso_v2.strategies.base import EntrySignal
from training_iso_v2.strategies._nmp_base import NMPBaseStrategy, NMPSeed
from training_iso_v2.strategies.regime_aware import DEFAULT_FLIP_CELLS


class NMPFadeRaw(NMPBaseStrategy):
    """Pure NMP fade — no filter beyond the seed itself.

    ── 2026-05-10 RETUNE (validated 67,105 IS+OOS trades) ────────────
    E1 z-band: |z| in [1.5, 1.8]  → PF_WR 0.030 -> 0.109
    E2 VETO  short × aligned (1h_z_se >= +0.3)
    Combined OOS uplift: +$250 (90 -> 340)
    Total IS$ drops (5584 -> 3738) but OOS improves and PF_WR jumps.
    """
    name = 'NMP_FADE_RAW'

    RETUNE_Z_LO = 1.5
    RETUNE_Z_HI = 1.8
    RETUNE_VETO = [('short', 'aligned')]

    def __init__(self, retune: bool = True, **kwargs):
        if retune:
            kwargs.setdefault('z_band_lo', self.RETUNE_Z_LO)
            kwargs.setdefault('z_band_hi', self.RETUNE_Z_HI)
            kwargs.setdefault('veto_cells', self.RETUNE_VETO)
            kwargs.setdefault('z_threshold', self.RETUNE_Z_LO)
        super().__init__(**kwargs)

    def _qualify(self, state: BarState, seed: NMPSeed) -> Optional[EntrySignal]:
        return EntrySignal(direction=seed.direction, tier=self.name,
                              extras={'z_se': seed.z,
                                          'reversion_prob': seed.rprob})


class NMPRideRaw(NMPBaseStrategy):
    """NMP seed + flip-cell membership only. No velocity gate."""
    name = 'NMP_RIDE_RAW'

    def _qualify(self, state: BarState, seed: NMPSeed) -> Optional[EntrySignal]:
        if (int(state.regime_idx), seed.direction) not in DEFAULT_FLIP_CELLS:
            return None
        new_dir = 'short' if seed.direction == 'long' else 'long'
        return EntrySignal(direction=new_dir, tier=self.name,
                              extras={'z_se': seed.z,
                                          'reversion_prob': seed.rprob,
                                          'flipped_from': seed.direction,
                                          'regime_2d': state.regime_2d})
