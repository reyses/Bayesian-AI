"""Per-bar trading state — the single object passed to strategies + exits.

V2-native: everything addressable by canonical V2 column name. No 91D vector,
no `_1M_OFFSET` indices, no V1 layout assumptions.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Any

import numpy as np


# Regime vocabulary — index 0 reserved for UNKNOWN so the embedding is total.
# WARMUP added for hourly regime mode — bars before any completed hour have
# no regime signal yet; strategies/exits that gate on regime should treat
# WARMUP as "unknown" and skip or use permissive defaults.
REGIME_VOCAB = ('UNKNOWN', 'UP_SMOOTH', 'UP_CHOPPY',
                    'DOWN_SMOOTH', 'DOWN_CHOPPY',
                    'FLAT_SMOOTH', 'FLAT_CHOPPY',
                    'WARMUP')


def regime_to_idx(regime_2d: str) -> int:
    try:
        return REGIME_VOCAB.index(regime_2d)
    except ValueError:
        return 0


# ── Velocity-based regime classifier (forward-pass-honest, per-bar) ─────
# Reads L2 velocity + L3 swing_noise from V2 features at the CURRENT bar.
# These are rolling-window stats, by construction using only past data —
# so no lookahead. Per-bar classification means regime transitions are
# smooth (vs the hourly batch labeler's stepwise jumps).
#
# Thresholds default to MNQ-tuned values; override at runtime if needed.

def is_trend_too_fast(vel: float, direction: str,
                              fast_thr: float = 20.0,
                              mode: str = 'symmetric') -> bool:
    """Velocity-aware entry gate. Returns True when the entry should be SKIPPED
    because the macro velocity is too extreme.

    mode='symmetric' (default, data-validated):
        Skip if |vel| > fast_thr regardless of direction. Both counter-trend
        (fade against fast macro) AND pro-trend chasing (riding after the move
        already happened) are blocked. Per the 2026_02_12 audit:
            - moderate vel (+7) was the sweet spot ($147/trade)
            - fast vel (-25 to -49) tanked quality ($20/trade)

    mode='counter_only':
        Skip only counter-trend entries (LONG in fast down, SHORT in fast up).
        Permits pro-trend rides at any velocity.

    mode='pro_only':
        Skip only pro-trend entries (SHORT in fast down, LONG in fast up).
        Permits counter-trend fades at any velocity.
    """
    if vel != vel:        # NaN guard
        return False
    av = abs(vel)
    if mode == 'symmetric':
        return av > fast_thr
    if mode == 'counter_only':
        if direction == 'long' and vel <= -fast_thr:
            return True
        if direction == 'short' and vel >= fast_thr:
            return True
        return False
    if mode == 'pro_only':
        if direction == 'short' and vel <= -fast_thr:
            return True
        if direction == 'long' and vel >= fast_thr:
            return True
        return False
    return False


def classify_regime_from_velocity(vel: float, sn: float,
                                                  vel_thr: float = 1.0,
                                                  sn_thr: float = 100.0) -> str:
    """Classify (direction × variation) from one TF's velocity + swing_noise.

    direction: |vel| above vel_thr → UP/DOWN by sign; else FLAT
    variation: sn below sn_thr → SMOOTH; else CHOPPY

    `vel_thr` and `sn_thr` should be calibrated per TF on IS data —
    use tools/calibrate_velocity_regime.py to set them.
    """
    if vel != vel:        # NaN guard (warmup)
        return 'WARMUP'
    if vel >= vel_thr:
        direction = 'UP'
    elif vel <= -vel_thr:
        direction = 'DOWN'
    else:
        direction = 'FLAT'
    variation = 'SMOOTH' if (sn != sn and sn != 0) or sn < sn_thr else 'CHOPPY'
    if sn != sn:           # sn NaN
        variation = 'SMOOTH'    # default permissive
    return f'{direction}_{variation}'


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
    # Higher-TF most-recent CLOSED bar (lookahead-free) for wick-rejection tiers.
    # None when the day hasn't seen a close at that TF yet, OR the parquet is
    # absent for the day.
    ohlcv_5m: Optional[Dict[str, float]] = None
    ohlcv_15m: Optional[Dict[str, float]] = None
    ohlcv_1h: Optional[Dict[str, float]] = None

    def get(self, col: str, default: float = 0.0) -> float:
        """Safe V2 column lookup."""
        return self.v2.get(col, default)
