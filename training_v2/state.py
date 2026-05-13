"""Per-bar trading state — the single object passed to strategies + exits.

V2-native: everything addressable by canonical V2 column name. No 91D vector,
no `_1M_OFFSET` indices, no V1 layout assumptions.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Any

import numpy as np


# Regime vocabulary — index 0 reserved for UNKNOWN so the embedding is total
REGIME_VOCAB = ('UNKNOWN', 'UP_SMOOTH', 'UP_CHOPPY',
                    'DOWN_SMOOTH', 'DOWN_CHOPPY',
                    'FLAT_SMOOTH', 'FLAT_CHOPPY')


def regime_to_idx(regime_2d: str) -> int:
    try:
        return REGIME_VOCAB.index(regime_2d)
    except ValueError:
        return 0


@dataclass
class BarState:
    """One 5s bar's worth of state. Strategies + exits read this; nothing
    else is allowed to be passed in.

    Fields:
        timestamp     : 5s bar start (Unix seconds)
        bar_idx       : monotonic index within the day
        day           : 'YYYY_MM_DD'
        price         : current close (from 1m OHLCV row covering this 5s)
        ohlcv_5s      : current 5s OHLCV row {open, high, low, close, volume}
        ohlcv_1m      : current 1m OHLCV row (the 1m bar containing this 5s)
        v2            : dict {col_name: float} for all 185 V2 features.
                          Use named lookups: v2['L3_1m_z_se_15'].
        v2_vector     : np.ndarray (185,) in canonical FEATURE_NAMES order.
                          For CNN consumption.
        regime_2d     : 'UP_SMOOTH' | 'UP_CHOPPY' | 'DOWN_SMOOTH' | ...
        regime_idx    : int 0..len(REGIME_VOCAB)-1
        is_1m_close   : True on the 5s bar that closes a 1m bar (entry trigger)
        is_5m_close   : True on the 5s bar that closes a 5m bar
        is_15m_close  : True on the 5s bar that closes a 15m bar
        is_1h_close   : True on the 5s bar that closes a 1h bar
    """
    timestamp: float
    bar_idx: int
    day: str
    price: float
    ohlcv_5s: Dict[str, float]
    ohlcv_1m: Optional[Dict[str, float]]
    v2: Dict[str, float]
    v2_vector: np.ndarray
    regime_2d: str
    regime_idx: int
    is_1m_close: bool = False
    is_5m_close: bool = False
    is_15m_close: bool = False
    is_1h_close: bool = False

    def get(self, col: str, default: float = 0.0) -> float:
        """Safe V2 column lookup."""
        return self.v2.get(col, default)
