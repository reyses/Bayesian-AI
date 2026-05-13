"""Position state + closed-trade list.

Single-position-at-a-time for now. Chains can be added later under a flag
without changing the strategy/exit contracts.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any

import numpy as np


# MNQ contract economics
TICK = 0.25
TICK_VALUE = 0.50      # $/tick/contract
DOLLAR_PER_POINT = 2.0  # $/pt/contract


@dataclass
class Position:
    """One open position. Owned by the ledger; strategies + exits read it."""
    direction: str               # 'long' | 'short'
    entry_price: float
    entry_ts: float
    entry_tier: str              # which strategy fired the entry
    entry_v2: np.ndarray         # 185D V2 snapshot at entry (for CNN training)
    entry_regime_idx: int
    entry_day: str

    bars_held: int = 0
    peak_pnl: float = 0.0
    trough_pnl: float = 0.0
    last_price: float = 0.0
    cnn_filtered: bool = False   # was a deterministic entry kept by CNN filter
    cnn_generated: bool = False  # was the entry CNN-originated (no rule fired)
    extras: Dict[str, Any] = field(default_factory=dict)

    def pnl(self, price: float) -> float:
        if self.direction == 'long':
            return (price - self.entry_price) / TICK * TICK_VALUE
        return (self.entry_price - price) / TICK * TICK_VALUE


@dataclass
class ClosedTrade:
    direction: str
    entry_price: float
    exit_price: float
    entry_ts: float
    exit_ts: float
    bars_held: int
    pnl: float
    peak_pnl: float
    entry_tier: str
    exit_reason: str
    entry_day: str
    entry_v2: np.ndarray
    entry_regime_idx: int
    cnn_filtered: bool = False
    cnn_generated: bool = False
    extras: Dict[str, Any] = field(default_factory=dict)


class Ledger:
    """Single-position ledger. Add later: chains, sizing, equity tracking."""

    def __init__(self):
        self.position: Optional[Position] = None
        self.closed: List[ClosedTrade] = []

    @property
    def is_flat(self) -> bool:
        return self.position is None

    def open(self, direction: str, price: float, ts: float, tier: str,
             v2_vector: np.ndarray, regime_idx: int, day: str,
             cnn_filtered: bool = False, cnn_generated: bool = False,
             extras: Optional[Dict[str, Any]] = None) -> None:
        if self.position is not None:
            raise RuntimeError('open() called while in position')
        self.position = Position(
            direction=direction, entry_price=price, entry_ts=ts,
            entry_tier=tier, entry_v2=v2_vector.copy(),
            entry_regime_idx=regime_idx, entry_day=day,
            last_price=price, cnn_filtered=cnn_filtered,
            cnn_generated=cnn_generated,
            extras=dict(extras or {}),
        )

    def update(self, price: float) -> None:
        if self.position is None:
            return
        self.position.bars_held += 1
        self.position.last_price = price
        pnl = self.position.pnl(price)
        if pnl > self.position.peak_pnl:
            self.position.peak_pnl = pnl
        if pnl < self.position.trough_pnl:
            self.position.trough_pnl = pnl

    def close(self, price: float, ts: float, reason: str) -> Optional[ClosedTrade]:
        if self.position is None:
            return None
        p = self.position
        trade = ClosedTrade(
            direction=p.direction, entry_price=p.entry_price, exit_price=price,
            entry_ts=p.entry_ts, exit_ts=ts, bars_held=p.bars_held,
            pnl=p.pnl(price), peak_pnl=p.peak_pnl, entry_tier=p.entry_tier,
            exit_reason=reason, entry_day=p.entry_day, entry_v2=p.entry_v2,
            entry_regime_idx=p.entry_regime_idx,
            cnn_filtered=p.cnn_filtered, cnn_generated=p.cnn_generated,
            extras=dict(p.extras),
        )
        self.closed.append(trade)
        self.position = None
        return trade
