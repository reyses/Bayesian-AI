"""
Ticker — dumb pipe that feeds 1s bars one at a time.

This module has ONE job: deliver bars sequentially.
It does NOT:
  - Aggregate timeframes
  - Compute features
  - Run SFE
  - Make any decisions

Usage (backtest from file):
    ticker = FileTicker('DATA/ATLAS/1s/2026_01_06.parquet')
    for bar in ticker:
        # bar = {'timestamp', 'open', 'high', 'low', 'close', 'volume'}
        ai.on_bar(bar)

Usage (feed bars manually):
    ticker = ManualTicker()
    ticker.feed(bar_dict)  # returns the bar back (pass-through)
"""
import pandas as pd
from typing import Iterator, Dict


class FileTicker:
    """Reads 1s bars from a parquet file and yields them one at a time."""

    def __init__(self, filepath: str):
        self._df = pd.read_parquet(filepath).sort_values('timestamp').reset_index(drop=True)
        self._n = len(self._df)
        self._idx = 0

    def __iter__(self) -> Iterator[Dict]:
        for i in range(self._n):
            row = self._df.iloc[i]
            yield {
                'timestamp': float(row['timestamp']),
                'open': float(row['open']),
                'high': float(row['high']),
                'low': float(row['low']),
                'close': float(row['close']),
                'volume': float(row['volume']),
            }

    def __len__(self) -> int:
        return self._n


class MultiDayTicker:
    """Reads 1s bars from multiple daily parquet files sequentially."""

    def __init__(self, filepaths: list):
        self._files = filepaths
        self._total = 0

    def __iter__(self) -> Iterator[Dict]:
        for fp in self._files:
            df = pd.read_parquet(fp).sort_values('timestamp').reset_index(drop=True)
            self._total += len(df)
            for i in range(len(df)):
                row = df.iloc[i]
                yield {
                    'timestamp': float(row['timestamp']),
                    'open': float(row['open']),
                    'high': float(row['high']),
                    'low': float(row['low']),
                    'close': float(row['close']),
                    'volume': float(row['volume']),
                }


class ManualTicker:
    """Pass-through for live/manual bar feeding."""

    def feed(self, bar: Dict) -> Dict:
        """Pass a bar through. Returns the same bar."""
        return bar
