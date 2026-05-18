"""DMI-smoothed trend3 strategy — fires at confirmed regime transitions.

Reads the smoothed trend3 cache (built by tools/precompute_trend3_smoothed.py)
which adds:
    p_long_ema / p_short_ema  : EMA-smoothed raw probabilities
    dx / adx                   : DMI-style directional strength
    regime_dir                 : state-machine output (LONG/SHORT/NEUTRAL)
    regime_change              : True on bars where regime flipped

Fire rule:
  Default (mode='on_flip'): fire ENTRY on the bar where regime_change is True
                            (a confirmed direction flip just happened).
  Mode='regime':            fire on every bar where regime_dir matches a
                            directional state. Engine's flat-only logic will
                            still hold the position until exit.

Optional ADX gate:
  Skip the fire if `adx < adx_floor` at the moment of fire (filter weak regimes).
"""
from __future__ import annotations

import os
from typing import Optional

from training_iso_v2.state import BarState
from training_iso_v2.strategies.base import EntrySignal, Strategy


class Trend3SmoothedStrategy(Strategy):
    """Fires on confirmed regime transitions from the smoothed trend3 cache."""

    name = 'TREND3_SMOOTHED'

    def __init__(self, smoothed_cache: str,
                 mode: str = 'on_flip',          # 'on_flip' | 'regime'
                 adx_floor: float = 0.0,          # extra gate on ADX
                 fire_cadence: str = '1m'):
        import pandas as pd
        if not os.path.exists(smoothed_cache):
            raise FileNotFoundError(smoothed_cache)
        df = pd.read_parquet(smoothed_cache)
        # Lookup by timestamp → (regime, regime_change, adx)
        self._lookup = {
            int(ts): (str(rd), bool(rc), float(ad))
            for ts, rd, rc, ad in zip(
                df['timestamp'], df['regime_dir'],
                df['regime_change'], df['adx']
            )
        }
        self.mode = mode
        self.adx_floor = float(adx_floor)
        self.fire_cadence = fire_cadence

    def _ready(self, state: BarState) -> bool:
        return {
            '5s': True,
            '15s': state.bar_idx % 3 == 0,
            '1m': state.is_1m_close,
            '5m': state.is_5m_close,
            '15m': state.is_15m_close,
        }.get(self.fire_cadence, False)

    def evaluate(self, state: BarState) -> Optional[EntrySignal]:
        if not self._ready(state):
            return None
        rec = self._lookup.get(int(state.timestamp))
        if rec is None:
            return None
        regime, changed, adx = rec
        if adx < self.adx_floor:
            return None
        if regime not in ('LONG', 'SHORT'):
            return None
        if self.mode == 'on_flip' and not changed:
            return None
        direction = 'long' if regime == 'LONG' else 'short'
        return EntrySignal(direction=direction, tier=self.name,
                            extras={'regime': regime, 'adx': adx,
                                    'regime_change': changed})
