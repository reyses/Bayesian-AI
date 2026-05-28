"""V2-native directional wick math — pure OHLCV, no V1 dependency.

A bar's wick measures the rejection of a price move. Definitions:
    body_high = max(open, close)
    body_low  = min(open, close)
    upper_wick = high - body_high            # rejection of upward move
    lower_wick = body_low - low              # rejection of downward move
    bar_range  = high - low
    body_size  = body_high - body_low

Ratios (in [0, 1]):
    upper_wick_ratio = upper_wick / bar_range
    lower_wick_ratio = lower_wick / bar_range

For mean-reversion entries (NMP fade):
  - LONG entry (price stretched DOWN — z << 0): we want a LOWER WICK at the
    fade bar. Price tried to go lower but rejected back up — bullish rejection.
  - SHORT entry (price stretched UP — z >> 0): we want an UPPER WICK at the
    fade bar. Price tried to go higher but rejected back down — bearish rejection.

For trend-continuation entries (RIDE):
  - Same wicks but interpreted as failed COUNTER pressure → trend continues.

KILL_SHOT historical threshold (2026-04-06): 5m wick > 0.83 + 15m wick > 0.77.
CASCADE = same wick + 1h velocity aligned with fade direction.

This module is V2-native because it depends ONLY on OHLCV data, which is
the underlying market data, not a V1 feature. The math is identical in V1
and V2 — just the container/feature-extraction code differs.
"""
from __future__ import annotations

from typing import Dict, Optional


def upper_wick_ratio(o: float, h: float, l: float, c: float) -> float:
    """Fraction of bar range above the body (rejection of upward move)."""
    rng = h - l
    if rng <= 0:
        return 0.0
    body_high = max(o, c)
    return max(0.0, (h - body_high) / rng)


def lower_wick_ratio(o: float, h: float, l: float, c: float) -> float:
    """Fraction of bar range below the body (rejection of downward move)."""
    rng = h - l
    if rng <= 0:
        return 0.0
    body_low = min(o, c)
    return max(0.0, (body_low - l) / rng)


def directional_wick(o: float, h: float, l: float, c: float,
                            direction: str) -> float:
    """Wick ratio in the direction supporting `direction` as a fade entry.

    direction='long'  -> LONG fade entry needs LOWER WICK (rejection of low)
    direction='short' -> SHORT fade entry needs UPPER WICK (rejection of high)
    """
    if direction == 'long':
        return lower_wick_ratio(o, h, l, c)
    if direction == 'short':
        return upper_wick_ratio(o, h, l, c)
    return 0.0


def wick_ratio_from_bar(bar: Optional[Dict[str, float]],
                                direction: str) -> float:
    """Convenience wrapper — None bar → 0.0 (warmup-safe)."""
    if bar is None:
        return 0.0
    return directional_wick(bar.get('open', 0.0), bar.get('high', 0.0),
                                       bar.get('low', 0.0), bar.get('close', 0.0),
                                       direction)
