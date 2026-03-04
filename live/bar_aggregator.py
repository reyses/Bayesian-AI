"""
LiveBarAggregator — aggregates inbound 1s bars into 15s bars, accumulates
them into a growing DataFrame, and recomputes quantum states via the
QuantumFieldEngine.

NT8 sends 1-second bars.  The aggregator buffers 15 of them, builds one
OHLCV 15s bar, and appends it to the state buffer.  At session reset
(detected via timestamp gap) the buffer is flushed.
"""

import logging
import numpy as np
import pandas as pd
from typing import List, Optional

from core.quantum_field_engine import QuantumFieldEngine
from live.config import LiveConfig

logger = logging.getLogger(__name__)

TARGET_PERIOD = 15   # default; overridden by constructor


class LiveBarAggregator:
    """Aggregate 1s bars into anchor-TF bars, accumulate, recompute quantum states."""

    def __init__(self, engine: QuantumFieldEngine, config: LiveConfig,
                 target_period: int = TARGET_PERIOD):
        self._engine = engine
        self._cfg = config
        self._target_period = target_period
        self._rows: list = []       # completed anchor-TF bars
        self._rows_1s: list = []    # raw 1s bars (for TBN sub-resolution workers)
        self._states: list = []
        self._warmed_up = False
        self._sub_bars: list = []   # buffered 1s bars for current window
        self._history_mode = True   # True until HISTORY_DONE received

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

    @property
    def df_1s(self) -> pd.DataFrame:
        """Raw 1s bar DataFrame (for TBN sub-resolution workers)."""
        if not self._rows_1s:
            return pd.DataFrame()
        return pd.DataFrame(self._rows_1s)

    def add_bar(self, msg: dict) -> Optional[list]:
        """
        Append an inbound 1s BAR message.  Every 15 bars, emit one 15s
        OHLCV bar and recompute states.

        Returns the states list once warmed up, else None.
        """
        bar_period = int(msg.get('bar_period_s', 1))

        row_1s = {
            'timestamp': float(msg['timestamp']),
            'open':      float(msg['open']),
            'high':      float(msg['high']),
            'low':       float(msg['low']),
            'close':     float(msg['close']),
            'volume':    float(msg.get('volume', 0)),
        }

        # Session reset detection: >2h gap between bars (disabled during history)
        if self._sub_bars and not self._history_mode:
            gap = row_1s['timestamp'] - self._sub_bars[-1]['timestamp']
            if gap > 7200:
                logger.info(f"Session gap ({gap:.0f}s) — resetting aggregator")
                self.reset()

        # If source is already at anchor period (or larger), pass through
        if bar_period >= self._target_period:
            return self._append_bar(row_1s)

        # Keep raw 1s bars for TBN sub-resolution workers
        self._rows_1s.append(row_1s)

        # Buffer the 1s bar
        self._sub_bars.append(row_1s)

        # Check if we've completed an anchor-TF window
        if len(self._sub_bars) >= self._target_period:
            agg = self._aggregate_sub_bars()
            self._sub_bars.clear()
            return self._append_bar(agg)

        return None

    def finish_history(self):
        """Called when HISTORY_DONE received — recompute and go live."""
        total = self.bar_count
        logger.info(f"History ingestion complete: {total} bars retained")
        # One bulk recompute
        if len(self._rows) >= self._cfg.warmup_bars:
            self._warmed_up = True
            self._recompute()
            logger.info(f"Post-history recompute: {len(self._states)} states ready")
        self._history_mode = False

    def reset(self):
        """Flush all bars and states (session boundary)."""
        self._rows.clear()
        self._rows_1s.clear()
        self._states.clear()
        self._sub_bars.clear()
        self._warmed_up = False
        logger.info("Aggregator reset")

    # ── Internal ──────────────────────────────────────────────────────

    def _aggregate_sub_bars(self) -> dict:
        """Combine buffered 1s bars into a single 15s OHLCV bar."""
        bars = self._sub_bars
        return {
            'timestamp': bars[0]['timestamp'],
            'open':      bars[0]['open'],
            'high':      max(b['high'] for b in bars),
            'low':       min(b['low'] for b in bars),
            'close':     bars[-1]['close'],
            'volume':    sum(b['volume'] for b in bars),
        }

    def _append_bar(self, row: dict) -> Optional[list]:
        """Append a completed 15s bar and recompute if warmed up."""
        # Session gap check against last 15s bar (disabled during history)
        if self._rows and not self._history_mode:
            gap = row['timestamp'] - self._rows[-1]['timestamp']
            if gap > 7200:
                logger.info(f"Session gap ({gap:.0f}s) — resetting aggregator")
                self.reset()

        self._rows.append(row)

        # During history ingestion, just accumulate — no per-bar recompute
        if self._history_mode:
            if self.bar_count % 500 == 0:
                logger.info(f"History ingestion: {self.bar_count} bars buffered")
            return None

        if self.bar_count < self._cfg.warmup_bars:
            if self.bar_count % 60 == 0:
                _elapsed_m = self.bar_count * self._target_period // 60
                logger.info(f"Warmup: {self.bar_count}/{self._cfg.warmup_bars} "
                            f"{self._target_period}s bars ({_elapsed_m}m)")
            return None

        self._warmed_up = True
        return self._recompute()

    def _recompute(self) -> list:
        """Recompute quantum states on the full bar buffer."""
        df = self.df
        try:
            self._states = self._engine.batch_compute_states(df, use_cuda=True)
        except Exception as e:
            logger.error(f"State recompute failed: {e}")
            self._states = []
        return self._states
