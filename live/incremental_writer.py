"""Crash-safe incremental parquet writer.

Writes small chunk files on periodic save. Merges at shutdown.
Recovery: load all chunks + merge.

DATA/ATLAS_LIVE/
  1s/
    2026_04_10.parquet          <- final merged file (at shutdown)
    _chunks/
      2026_04_10_001.parquet    <- chunk 1 (bars 0-300)
      2026_04_10_002.parquet    <- chunk 2 (bars 300-600)
"""
import os
import logging
import pandas as pd
from pathlib import Path

logger = logging.getLogger(__name__)


class IncrementalWriter:

    def __init__(self, base_dir: str, session_date: str):
        self._base_dir = base_dir
        self._date = session_date
        self._chunk_idx = 0
        self._last_saved_count = {}

    def _chunk_dir(self, tf: str) -> str:
        d = os.path.join(self._base_dir, tf, '_chunks')
        os.makedirs(d, exist_ok=True)
        return d

    def _final_path(self, tf: str) -> str:
        d = os.path.join(self._base_dir, tf)
        os.makedirs(d, exist_ok=True)
        return os.path.join(d, f'{self._date}.parquet')

    def save_chunk(self, tf: str, all_bars: list):
        """Save only NEW bars since last chunk."""
        prev_count = self._last_saved_count.get(tf, 0)
        if len(all_bars) <= prev_count:
            return

        new_bars = all_bars[prev_count:]
        df = pd.DataFrame(new_bars)

        chunk_dir = self._chunk_dir(tf)
        chunk_name = f'{self._date}_{self._chunk_idx:04d}_{tf}.parquet'
        chunk_path = os.path.join(chunk_dir, chunk_name)

        tmp_path = chunk_path + '.tmp'
        df.to_parquet(tmp_path, index=False)
        os.replace(tmp_path, chunk_path)

        self._last_saved_count[tf] = len(all_bars)

    def save_all_chunks(self, bars_by_tf: dict):
        """Save chunks for all TFs. Called every 5 min."""
        for tf, bars in bars_by_tf.items():
            if bars:
                self.save_chunk(tf, bars)
        self._chunk_idx += 1

    def merge_final(self, tf: str, all_bars: list = None):
        """Merge all chunks into final day file. Called at shutdown."""
        final_path = self._final_path(tf)

        if all_bars:
            df = pd.DataFrame(all_bars)
        else:
            chunk_dir = self._chunk_dir(tf)
            if not os.path.exists(chunk_dir):
                return
            chunks = sorted(Path(chunk_dir).glob(f'{self._date}_*_{tf}.parquet'))
            if not chunks:
                return
            dfs = [pd.read_parquet(c) for c in chunks]
            df = pd.concat(dfs, ignore_index=True)

        df = (df.sort_values('timestamp')
                .drop_duplicates(subset='timestamp', keep='last')
                .reset_index(drop=True))

        tmp_path = final_path + '.tmp'
        df.to_parquet(tmp_path, index=False)
        os.replace(tmp_path, final_path)

        logger.info(
            f'ATLAS_LIVE {tf}: {len(df)} bars -> {final_path}')

        chunk_dir = self._chunk_dir(tf)
        if os.path.exists(chunk_dir):
            for c in Path(chunk_dir).glob(f'{self._date}_*_{tf}.parquet'):
                c.unlink()

    def merge_all_final(self, bars_by_tf: dict):
        """Merge all TFs at shutdown."""
        for tf, bars in bars_by_tf.items():
            if bars:
                self.merge_final(tf, bars)

    @staticmethod
    def recover(base_dir: str, session_date: str) -> dict:
        """Crash recovery: reconstruct day data from chunks."""
        recovered = {}
        base = Path(base_dir)

        for tf_dir in base.iterdir():
            if not tf_dir.is_dir():
                continue
            tf = tf_dir.name
            chunk_dir = tf_dir / '_chunks'
            if not chunk_dir.exists():
                continue

            chunks = sorted(chunk_dir.glob(f'{session_date}_*.parquet'))
            if not chunks:
                continue

            dfs = [pd.read_parquet(c) for c in chunks]
            df = (pd.concat(dfs, ignore_index=True)
                    .sort_values('timestamp')
                    .drop_duplicates(subset='timestamp', keep='last')
                    .reset_index(drop=True))

            recovered[tf] = df
            logger.info(f'Recovered {tf}: {len(df)} bars from {len(chunks)} chunks')

        return recovered
