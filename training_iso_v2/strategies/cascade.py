"""CASCADE — V2 port of legacy "wick rejection + 1h aligned with fade".

Legacy (2026-04-06): same wick-rejection signal as KILL_SHOT, but with
1h velocity ALIGNED with the fade direction (1h going same way as the
mean-reversion target). +$36/trade, 54% WR, vs KILL_SHOT's -$15/trade
when 1h was NOT aligned. The 1h alignment is the difference between
trend-supported reversion and fading into a trend.

V2 trigger:
    NMP seed
    AND directional_wick(5m bar) >= WICK_5M_MIN
    AND directional_wick(15m bar) >= WICK_15M_MIN
    AND sign(L2_1h_price_velocity_w) ALIGNS with fade direction
        (LONG fade → 1h velocity > 0, SHORT fade → 1h velocity < 0)

Direction: NMP fade.
"""
from __future__ import annotations
from typing import Optional

from training_iso_v2.state import BarState
from training_iso_v2.strategies.base import EntrySignal
from training_iso_v2.strategies._nmp_base import NMPBaseStrategy, NMPSeed
from training_iso_v2.wicks import wick_ratio_from_bar
from training_iso_v2.v2_cols import price_velocity_w


WICK_5M_MIN_DEFAULT = 0.50
WICK_15M_MIN_DEFAULT = 0.45
H1_VEL_ALIGN_MIN_DEFAULT = 0.0  # any same-sign h1 velocity counts


class Cascade(NMPBaseStrategy):
    name = 'CASCADE'

    def __init__(self,
                 wick_5m_min: float = WICK_5M_MIN_DEFAULT,
                 wick_15m_min: float = WICK_15M_MIN_DEFAULT,
                 h1_vel_align_min: float = H1_VEL_ALIGN_MIN_DEFAULT,
                 **kwargs):
        super().__init__(**kwargs)
        self.wick_5m_min = wick_5m_min
        self.wick_15m_min = wick_15m_min
        self.h1_vel_align_min = h1_vel_align_min
        self._h1_vel_col = price_velocity_w('1h')

    def _qualify(self, state: BarState, seed: NMPSeed) -> Optional[EntrySignal]:
        w5 = wick_ratio_from_bar(state.ohlcv_5m, seed.direction)
        w15 = wick_ratio_from_bar(state.ohlcv_15m, seed.direction)
        if w5 < self.wick_5m_min or w15 < self.wick_15m_min:
            return None
        h1_v = state.get(self._h1_vel_col, 0.0)
        # ALIGNED: long fade with h1 going up, OR short fade with h1 going down
        aligned = ((seed.direction == 'long' and h1_v > self.h1_vel_align_min)
                         or (seed.direction == 'short' and h1_v < -self.h1_vel_align_min))
        if not aligned:
            return None
        return EntrySignal(direction=seed.direction, tier=self.name,
                              extras={'z_se': seed.z, 'wick_5m': w5,
                                          'wick_15m': w15, 'h1_velocity': h1_v})
