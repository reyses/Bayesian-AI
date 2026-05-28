"""KILL_SHOT — V2 port of legacy "z extreme + multi-TF wick rejection".

Legacy (2026-04-06): NMP entry plus 5m wick > 0.83 + 15m wick > 0.77 →
96% WR, $42/day. The wick rejection IS the quality filter for an NMP fade.

V2 trigger:
    NMP seed (|z_se_w| >= 1.8 + reversion_prob_w >= 0.55)
    AND directional_wick(5m bar) >= WICK_5M_MIN
    AND directional_wick(15m bar) >= WICK_15M_MIN

For LONG entry (price stretched DOWN), need LOWER wicks at 5m and 15m
(price tried lower but rejected — bullish rejection).
For SHORT entry, need UPPER wicks at 5m and 15m.

Pure OHLCV math (V2-native because OHLCV isn't V1-specific).
"""
from __future__ import annotations
from typing import Optional

from training.utils.state import BarState
from training.strategies.base import EntrySignal
from training.strategies._nmp_base import NMPBaseStrategy, NMPSeed
from training.utils.wicks import wick_ratio_from_bar


WICK_5M_MIN_DEFAULT = 0.50    # legacy 0.83 was on V1 data; recalibrate as needed
WICK_15M_MIN_DEFAULT = 0.45   # legacy 0.77; recalibrate


class KillShot(NMPBaseStrategy):
    name = 'KILL_SHOT'

    def __init__(self,
                 wick_5m_min: float = WICK_5M_MIN_DEFAULT,
                 wick_15m_min: float = WICK_15M_MIN_DEFAULT,
                 **kwargs):
        super().__init__(**kwargs)
        self.wick_5m_min = wick_5m_min
        self.wick_15m_min = wick_15m_min

    def _qualify(self, state: BarState, seed: NMPSeed) -> Optional[EntrySignal]:
        w5 = wick_ratio_from_bar(state.ohlcv_5m, seed.direction)
        w15 = wick_ratio_from_bar(state.ohlcv_15m, seed.direction)
        if w5 < self.wick_5m_min or w15 < self.wick_15m_min:
            return None
        return EntrySignal(direction=seed.direction, tier=self.name,
                              extras={'z_se': seed.z,
                                          'wick_5m': w5, 'wick_15m': w15})
