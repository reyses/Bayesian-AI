"""
LiveBarAggregator — accumulates inbound 15s bars into a growing DataFrame
and recomputes quantum states via the QuantumFieldEngine.

In live mode there is one day of bars that grows bar-by-bar.  At session
reset (detected via timestamp gap) the buffer is flushed.
"""

import logging
import numpy as np
import pandas as pd
from typing import List, Optional

from core.quantum_field_engine import QuantumFieldEngine
from live.config import LiveConfig

logger = logging.getLogger(__name__)


class LiveBarAggregator:
    """Accumulate bars and recompute quantum states incrementally."""

    def __init__(self, engine: QuantumFieldEngine, config: LiveConfig):
        self._engine = engine
        self._cfg = config
        self._rows: list = []
        self._states: list = []
        self._warmed_up = False

    # ── Public API ────────────────────────────────────────────────────

    @property
    def bar_count(self) -> int:
        return len(self._rows)

    @property
    def is_warmed_up(self) -> bool:
        return self._warmed_up

    @property
    def states(self) -> list:
        return self._states

    @property
    def df(self) -> pd.DataFrame:
        """Current bar DataFrame (empty if no bars yet)."""
        if not self._rows:
            return pd.DataFrame()
        return pd.DataFrame(self._rows)

    def add_bar(self, msg: dict) -> Optional[list]:
        """
        Append an inbound BAR message and recompute states.

        Returns the states list once warmed up, else None.
        """
        row = {
            'timestamp': float(msg['timestamp']),
            'open':      float(msg['open']),
            'high':      float(msg['high']),
            'low':       float(msg['low']),
            'close':     float(msg['close']),
            'volume':    float(msg.get('volume', 0)),
        }

        # Session reset detection: >2h gap between bars
        if self._rows:
            gap = row['timestamp'] - self._rows[-1]['timestamp']
            if gap > 7200:
                logger.info(f"Session gap detected ({gap:.0f}s) — resetting aggregator")
                self.reset()

        self._rows.append(row)

        if self.bar_count < self._cfg.warmup_bars:
            if self.bar_count % 60 == 0:
                logger.info(f"Warmup: {self.bar_count}/{self._cfg.warmup_bars} bars")
            return None

        self._warmed_up = True
        return self._recompute()

    def reset(self):
        """Flush all bars and states (session boundary)."""
        self._rows.clear()
        self._states.clear()
        self._warmed_up = False
        logger.info("Aggregator reset")

    # ── Internal ──────────────────────────────────────────────────────

    def _recompute(self) -> list:
        """Recompute quantum states on the full bar buffer."""
        df = self.df
        try:
            self._states = self._engine.batch_compute_states(df, use_cuda=True)
        except Exception as e:
            logger.error(f"State recompute failed: {e}")
            self._states = []
        return self._states
