"""
Data Feeder  -- source-agnostic bar provider for the trading engine.

The engine doesn't know or care where bars come from.
ParquetFeeder reads ATLAS files. NT8Feeder (future) receives socket messages.
Same engine.feed_bar() call regardless of source.
"""
import os
from pathlib import Path
from typing import Iterator, Dict

import pandas as pd


class BarFeeder:
    """Abstract bar feeder. Subclass and implement iter_bars()."""

    def iter_bars(self) -> Iterator[Dict]:
        """Yields dicts with: timestamp, open, high, low, close, volume."""
        raise NotImplementedError

    def file_count(self) -> int:
        """Number of source files (for progress tracking)."""
        return 0

    def label(self) -> str:
        """Human-readable label for this feeder."""
        return 'unknown'


class ParquetFeeder(BarFeeder):
    """Reads ATLAS parquet files and yields bars sequentially.

    Usage:
        feeder = ParquetFeeder('DATA/ATLAS', tf='15s')
        for bar in feeder.iter_bars():
            engine.feed_bar(bar)
    """

    def __init__(self, atlas_root: str, tf: str = '15s'):
        self._root = atlas_root
        self._tf = tf
        tf_dir = os.path.join(atlas_root, tf)
        self._files = sorted(Path(tf_dir).glob('*.parquet'))
        if not self._files:
            raise FileNotFoundError(f"No parquet files in {tf_dir}")

    def file_count(self) -> int:
        return len(self._files)

    def label(self) -> str:
        return os.path.basename(self._root)

    def get_files(self):
        """Return list of parquet file paths (for TBN warmup, state precompute)."""
        return self._files

    def iter_bars(self) -> Iterator[Dict]:
        """Yields bars from all parquet files in order."""
        for f in self._files:
            df = pd.read_parquet(f)
            df = df.sort_values('timestamp').reset_index(drop=True)
            for _, row in df.iterrows():
                yield {
                    'timestamp': row['timestamp'],
                    'open': row['open'],
                    'high': row['high'],
                    'low': row['low'],
                    'close': row['close'],
                    'volume': row.get('volume', 0),
                    '_file': str(f),
                }

    def iter_dataframes(self) -> Iterator[pd.DataFrame]:
        """Yields entire DataFrames (for batch processing / state precompute)."""
        for f in self._files:
            df = pd.read_parquet(f).sort_values('timestamp').reset_index(drop=True)
            yield df, f


class ContinuousFeeder(BarFeeder):
    """Chains multiple feeders into one continuous stream.

    Usage:
        is_feeder = ParquetFeeder('DATA/ATLAS')
        oos_feeder = ParquetFeeder('DATA/ATLAS_OOS')
        continuous = ContinuousFeeder([
            ('IS', is_feeder),
            ('OOS', oos_feeder),
        ])
        for bar in continuous.iter_bars():
            engine.feed_bar(bar)
            # bar['_phase'] tells you which phase you're in
    """

    def __init__(self, feeders: list):
        """feeders: list of (phase_name, BarFeeder) tuples."""
        self._feeders = feeders
        self._phase_callbacks = {}  # phase_name -> callable (checkpoint hooks)

    def on_phase_complete(self, phase_name: str, callback):
        """Register a callback to run when a phase completes.
        Used for checkpoints between IS and OOS."""
        self._phase_callbacks[phase_name] = callback

    def file_count(self) -> int:
        return sum(f.file_count() for _, f in self._feeders)

    def label(self) -> str:
        return ' -> '.join(name for name, _ in self._feeders)

    def iter_bars(self) -> Iterator[Dict]:
        for phase_name, feeder in self._feeders:
            for bar in feeder.iter_bars():
                bar['_phase'] = phase_name
                yield bar

            # Phase complete  -- run checkpoint callback if registered
            cb = self._phase_callbacks.get(phase_name)
            if cb:
                cb(phase_name)
