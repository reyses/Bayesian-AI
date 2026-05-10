"""Strategy ABC.

Engine evaluates strategies in order; first one to return a non-None
EntrySignal wins. This implements the "first signal happens" rule the
user specified — no priority arbitration, no scoring.

Each strategy holds zero state across bars unless it explicitly opts in.
The bar state (BarState) carries everything needed.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Dict, Any

from training_iso_v2.state import BarState


@dataclass
class EntrySignal:
    direction: str           # 'long' | 'short'
    tier: str                # name of the firing strategy
    extras: Dict[str, Any] = None  # strategy-specific debug info


class Strategy(ABC):
    """Pure entry-rule. Reads BarState, returns EntrySignal or None."""

    name: str = 'BASE'

    @abstractmethod
    def evaluate(self, state: BarState) -> Optional[EntrySignal]:
        """Return an EntrySignal to open, or None to pass."""

    def reset(self) -> None:
        """Hook for strategies that carry state across bars. Default: no-op."""
        pass
