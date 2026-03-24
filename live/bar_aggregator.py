"""
LiveBarAggregator  -- aggregates inbound 1s bars into 15s bars, accumulates
them into a growing DataFrame, and recomputes market states via the
StatisticalFieldEngine.

NT8 sends 1-second bars.  The aggregator buffers 15 of them, builds one
OHLCV 15s bar, and appends it to the state buffer.  At session reset
(detected via timestamp gap) the buffer is flushed.
"""

import logging
import os
import pandas as pd
from typing import Optional

from core.statistical_field_engine import StatisticalFieldEngine
from live.config import LiveConfig

logger = logging.getLogger(__name__)

TARGET_PERIOD = 15   # default; overridden by constructor
MAX_MEMORY_BARS = 10000   # keep last N bars in RAM (~42h of 15s bars)
PERSIST_DIR = os.path.join('checkpoints', 'live')
FLUSH_INTERVAL = 1000     # auto-flush to parquet every N new bars


class LiveBarAggregator:
    """Aggregate 1s bars into anchor-TF bars, accumulate, recompute market states."""

    def __init__(self, engine: StatisticalFieldEngine, config: LiveConfig,
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
        self._bars_since_flush = 0  # counter for auto-flush

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

        # Session gap detection: >2h gap = daily maintenance break
        if self._sub_bars and not self._history_mode:
            gap = row_1s['timestamp'] - self._sub_bars[-1]['timestamp']
            if gap > 7200:
                logger.info(f"Session gap ({gap:.0f}s)  -- running daily maintenance")
                self.daily_maintenance()

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
        """Called when HISTORY_DONE received  -- recompute and go live."""
        total = self.bar_count
        logger.info(f"History ingestion complete: {total} bars retained")
        # One bulk recompute
        if len(self._rows) >= self._cfg.warmup_bars:
            self._warmed_up = True
            self._recompute()
            logger.info(f"Post-history recompute: {len(self._states)} states ready")
        self._history_mode = False

    @property
    def last_timestamp(self) -> float:
        """Timestamp of most recent bar (0.0 if empty)."""
        return self._rows[-1]['timestamp'] if self._rows else 0.0

    def reset(self):
        """Flush all bars and states (session boundary)."""
        self._rows.clear()
        self._rows_1s.clear()
        self._states.clear()
        self._sub_bars.clear()
        self._warmed_up = False
        self._bars_since_flush = 0
        logger.info("Aggregator reset")

    def seed_from_replay(self, df: 'pd.DataFrame', states: list):
        """Accept pre-computed state from history replay.

        Skips history ingestion entirely  -- aggregator starts warm.
        """
        self._rows = df.to_dict('records')
        self._states = states
        self._warmed_up = True
        self._history_mode = False
        logger.info(f"Seeded from replay: {len(self._rows):,} bars, "
                    f"{len(self._states)} states")

    # ── Persistence ──────────────────────────────────────────────────

    def _parquet_path(self) -> str:
        """Path to the persisted bar file."""
        os.makedirs(PERSIST_DIR, exist_ok=True)
        return os.path.join(PERSIST_DIR,
                            f'bars_{self._cfg.asset_ticker}_{self._target_period}s.parquet')

    def save_to_parquet(self):
        """Flush current bars to parquet (append-safe  -- overwrites with full buffer)."""
        if not self._rows:
            return
        path = self._parquet_path()
        df = pd.DataFrame(self._rows)
        # Merge with existing on-disk data (dedup by timestamp)
        if os.path.exists(path):
            try:
                old = pd.read_parquet(path)
                df = pd.concat([old, df]).drop_duplicates(
                    subset='timestamp', keep='last').sort_values('timestamp')
            except Exception as e:
                logger.warning(f"Could not merge with existing parquet: {e}")
        # Trim to 60 days max (60 * 24 * 3600 / target_period)
        max_bars = 60 * 24 * 3600 // max(self._target_period, 1)
        if len(df) > max_bars:
            df = df.iloc[-max_bars:]
        df.to_parquet(path, index=False)
        self._bars_since_flush = 0
        logger.info(f"Bars saved: {len(df):,} rows -> {path}")

    def update_atlas(self, atlas_root: str = 'DATA/ATLAS_LIVE'):
        """Append current bars to ATLAS parquet files (keeps OOS data current).

        NT8 sends 10k bars on connect + live bars during trading.
        This merges them into ATLAS format: {atlas_root}/{tf}/YYYY_MM.parquet
        so the next validation run includes the most recent market data.
        """
        from datetime import datetime, timezone
        if not self._rows:
            return

        tf_label = f'{self._target_period}s'
        tf_dir = os.path.join(atlas_root, tf_label)
        os.makedirs(tf_dir, exist_ok=True)

        df = pd.DataFrame(self._rows)
        df = df.sort_values('timestamp').drop_duplicates(subset='timestamp', keep='last')

        # Group bars by YYYY_MM for ATLAS file structure
        df['_month'] = df['timestamp'].apply(
            lambda ts: datetime.fromtimestamp(ts, tz=timezone.utc).strftime('%Y_%m'))
        n_written = 0
        for month, group in df.groupby('_month'):
            month_path = os.path.join(tf_dir, f'{month}.parquet')
            out = group.drop(columns=['_month'])
            # Merge with existing month file
            if os.path.exists(month_path):
                try:
                    old = pd.read_parquet(month_path)
                    out = pd.concat([old, out]).drop_duplicates(
                        subset='timestamp', keep='last').sort_values('timestamp')
                except Exception as e:
                    logger.warning(f"Could not merge with {month_path}: {e}")
            out.to_parquet(month_path, index=False)
            n_written += len(group)
            logger.info(f"ATLAS updated: {month_path} ({len(out):,} bars)")

        # Also update 1s bars if we have them
        if self._rows_1s:
            tf_1s_dir = os.path.join(atlas_root, '1s')
            os.makedirs(tf_1s_dir, exist_ok=True)
            df_1s = pd.DataFrame(self._rows_1s)
            df_1s = df_1s.sort_values('timestamp').drop_duplicates(
                subset='timestamp', keep='last')
            df_1s['_month'] = df_1s['timestamp'].apply(
                lambda ts: datetime.fromtimestamp(ts, tz=timezone.utc).strftime('%Y_%m'))
            for month, group in df_1s.groupby('_month'):
                month_path = os.path.join(tf_1s_dir, f'{month}.parquet')
                out = group.drop(columns=['_month'])
                if os.path.exists(month_path):
                    try:
                        old = pd.read_parquet(month_path)
                        out = pd.concat([old, out]).drop_duplicates(
                            subset='timestamp', keep='last').sort_values('timestamp')
                    except Exception:
                        pass
                out.to_parquet(month_path, index=False)
            logger.info(f"ATLAS 1s updated: {len(df_1s):,} bars")

        logger.info(f"ATLAS update complete: {n_written:,} {tf_label} bars -> {atlas_root}")

    def load_from_parquet(self) -> float:
        """Load persisted bars into the buffer. Returns last timestamp (0 if none)."""
        path = self._parquet_path()
        if not os.path.exists(path):
            logger.info("No persisted bars found  -- fresh start")
            return 0.0
        try:
            df = pd.read_parquet(path)
            if df.empty:
                return 0.0
            # Only keep last MAX_MEMORY_BARS for RAM
            if len(df) > MAX_MEMORY_BARS:
                df = df.iloc[-MAX_MEMORY_BARS:]
            self._rows = df.to_dict('records')
            last_ts = self._rows[-1]['timestamp']
            logger.info(f"Loaded {len(self._rows):,} persisted bars "
                        f"(last ts={last_ts:.0f})")
            return last_ts
        except Exception as e:
            logger.warning(f"Failed to load persisted bars: {e}")
            return 0.0

    def daily_maintenance(self):
        """Run during daily break (CME 5PM-6PM ET gap).

        Saves bars to parquet, trims 1s buffer, resets for fresh session.
        Called automatically when a >2h gap is detected between bars.
        """
        logger.info("=" * 50)
        logger.info("DAILY MAINTENANCE  -- session gap detected")

        # 1. Save current bars to disk + update ATLAS
        n_bars = self.bar_count
        n_1s = len(self._rows_1s)
        self.save_to_parquet()
        self.update_atlas()

        # 2. Trim 1s bars (no need to persist these  -- only for intra-session TBN)
        self._rows_1s.clear()

        # 3. Memory cap: keep only last MAX_MEMORY_BARS in RAM
        if len(self._rows) > MAX_MEMORY_BARS:
            trimmed = len(self._rows) - MAX_MEMORY_BARS
            self._rows = self._rows[-MAX_MEMORY_BARS:]
            logger.info(f"  Trimmed {trimmed:,} old bars from RAM "
                        f"(kept {MAX_MEMORY_BARS:,})")

        # 4. Clear states + sub-bars for fresh session recompute
        self._states.clear()
        self._sub_bars.clear()
        self._warmed_up = False
        self._bars_since_flush = 0

        logger.info(f"  Saved {n_bars:,} bars, cleared {n_1s:,} 1s bars")
        logger.info(f"  RAM buffer: {len(self._rows):,} bars retained")
        logger.info("=" * 50)

    def _maybe_auto_flush(self):
        """Auto-flush to parquet every FLUSH_INTERVAL bars."""
        self._bars_since_flush += 1
        if self._bars_since_flush >= FLUSH_INTERVAL:
            self.save_to_parquet()

    # ── Internal ──────────────────────────────────────────────────────

    def _aggregate_sub_bars(self) -> dict:
        """Combine buffered 1s bars into a single 15s OHLCV bar."""
        bars = self._sub_bars
        agg = {
            'timestamp': bars[0]['timestamp'],
            'open':      bars[0]['open'],
            'high':      max(b['high'] for b in bars),
            'low':       min(b['low'] for b in bars),
            'close':     bars[-1]['close'],
            'volume':    sum(b['volume'] for b in bars),
        }
        # Carry NT8 native indicators from last sub-bar (if present)
        last = bars[-1]
        for key in ('dmi_plus', 'dmi_minus', 'adx', 'dmi'):
            if key in last:
                agg[key] = last[key]
        return agg

    def _append_bar(self, row: dict) -> Optional[list]:
        """Append a completed 15s bar and recompute if warmed up."""
        # Session gap check against last 15s bar (disabled during history)
        if self._rows and not self._history_mode:
            gap = row['timestamp'] - self._rows[-1]['timestamp']
            if gap > 7200:
                logger.info(f"Session gap ({gap:.0f}s)  -- daily maintenance")
                self.daily_maintenance()

        self._rows.append(row)

        # During history ingestion, just accumulate  -- no per-bar recompute
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
        self._maybe_auto_flush()
        return self._recompute()

    def _recompute(self) -> list:
        """Recompute market states on the full bar buffer."""
        df = self.df
        try:
            self._states = self._engine.batch_compute_states(df, use_cuda=True)
        except Exception as e:
            logger.error(f"State recompute failed: {e}")
            self._states = []
        return self._states
