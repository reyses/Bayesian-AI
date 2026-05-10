"""MA-align trend-follow strategy.

Source: 2026-05-01 EDA finding (memory: project_v2_ma_alignment_directional.md).
Result on IS:
    7-of-8 TF vwap_w alignment → 70.5% direction acc on 20% of 5m bars
    (+17.6% lift over 52.9% baseline). Deterministic, no fit, walk-forward stable.
    15m and 1h vwap windows carry the signal; 5s-15s noise; 4h-1D too coarse.

Rule:
    For each TF in V2_TFS, compute alignment = sign(price - vwap_w[tf]).
    If at least N_ALIGN of the 8 TFs agree (e.g. 7 or 8), enter in that
    direction. Fires on 5m bar closes (matches the EDA cadence).

Entries are skipped when the alignment is mid-spectrum (4-of-8 split = chop).

CNN filter (Phase 6) gates whether to take the signal; this just emits it.
"""
from __future__ import annotations

from typing import Optional

from training_iso_v2.state import BarState
from training_iso_v2.strategies.base import EntrySignal, Strategy
from training_iso_v2.v2_cols import vwap_w, TF_ORDER


N_ALIGN_DEFAULT = 7   # 7-of-8 carries the EDA result; 8-of-8 = highest precision


class MAAlignTrendFollow(Strategy):
    """7-of-8 TF vwap_w alignment trend-follow."""

    name = 'MA_ALIGN'

    def __init__(self, n_align: int = N_ALIGN_DEFAULT,
                 fire_on: str = '5m'):
        self.n_align = n_align
        self.fire_on = fire_on  # '5m' | '15m' | '1m'

    def _ready_to_fire(self, state: BarState) -> bool:
        return {
            '1m': state.is_1m_close,
            '5m': state.is_5m_close,
            '15m': state.is_15m_close,
            '1h': state.is_1h_close,
        }.get(self.fire_on, False)

    def evaluate(self, state: BarState) -> Optional[EntrySignal]:
        if not self._ready_to_fire(state):
            return None

        price = state.price
        if price <= 0:
            return None

        above = 0
        below = 0
        per_tf = {}
        for tf in TF_ORDER:
            v = state.get(vwap_w(tf), default=float('nan'))
            if v != v or v <= 0:  # NaN or zero (warmup)
                continue
            if price > v:
                above += 1
                per_tf[tf] = +1
            elif price < v:
                below += 1
                per_tf[tf] = -1

        if above >= self.n_align:
            return EntrySignal(direction='long', tier=self.name,
                                  extras={'above': above, 'below': below,
                                              'per_tf': per_tf})
        if below >= self.n_align:
            return EntrySignal(direction='short', tier=self.name,
                                  extras={'above': above, 'below': below,
                                              'per_tf': per_tf})
        return None
