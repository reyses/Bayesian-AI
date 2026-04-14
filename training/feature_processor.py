"""
Feature Processor — shared incremental path for training + live.

One function: feed 5s bars through aggregator → compute_79d → save features.
Both build_dataset.py and engine_v2.py use this same path.
Checkpoint carries aggregator state between sessions.

Usage:
    # Build features for new days (from checkpoint)
    from training.feature_processor import FeatureProcessor
    fp = FeatureProcessor(atlas_root='DATA/ATLAS_NT8')
    fp.process_new_days()   # builds features for days after checkpoint

    # Process a single bar (live)
    feat = fp.process_bar(bar)  # returns 91D or None
"""
import os
import glob
import numpy as np
import pandas as pd
from datetime import datetime, timezone

from training.aggregator import Aggregator
from core.statistical_field_engine import StatisticalFieldEngine
from training.compute_79d import compute_79d_from_aggregator, SFE_MIN_BARS
from core.features_79d import FEATURE_NAMES_79D, TF_ORDER

HISTORY_LIMIT = 2000  # max bars per TF in aggregator


class FeatureProcessor:
    """Shared feature computation — same path for training and live."""

    def __init__(self, atlas_root='DATA/ATLAS_NT8', output_dir=None,
                 checkpoint_path=None):
        self._atlas_root = atlas_root
        self._output_dir = output_dir or self._derive_output_dir()
        self._checkpoint_path = checkpoint_path or os.path.join(atlas_root, 'checkpoint.json')

        self._agg = Aggregator(history_limit=HISTORY_LIMIT)
        self._sfe = StatisticalFieldEngine()
        self._prev_vel = {}
        self._last_ts = 0.0
        self._feat_buffer = []  # accumulated features for current day
        self._current_day = ''

    def _derive_output_dir(self):
        atlas_name = os.path.basename(self._atlas_root.rstrip('/'))
        feat_name = atlas_name.replace('ATLAS', 'FEATURES')
        return os.path.join('DATA', f'{feat_name}_5s')

    # ══════════════════════════════════════════════════════════════════
    # CHECKPOINT — load/save aggregator state
    # ══════════════════════════════════════════════════════════════════

    def load_checkpoint(self):
        """Load aggregator state from checkpoint. Returns True if loaded."""
        if not os.path.exists(self._checkpoint_path):
            return False
        self._last_ts, self._prev_vel, _ = self._agg.load_checkpoint(
            self._checkpoint_path)
        return True

    def save_checkpoint(self, trade_state=None):
        """Save current aggregator state to checkpoint."""
        self._agg.save_checkpoint(
            self._checkpoint_path,
            velocities=self._prev_vel,
            trade_state=trade_state)

    # ══════════════════════════════════════════════════════════════════
    # PROCESS BAR — single 5s bar (used by live engine)
    # ══════════════════════════════════════════════════════════════════

    def process_bar(self, bar: dict):
        """Feed one 5s bar, compute features if 5s boundary.

        Args:
            bar: dict with timestamp, open, high, low, close, volume

        Returns:
            91D numpy array if features computed, None otherwise.
        """
        self._agg.feed(bar)
        self._last_ts = bar['timestamp']

        # Only compute features on 5s boundaries (aggregator fires on_bar_close)
        if self._agg.get_bar_count('1m') < SFE_MIN_BARS:
            return None

        feat, self._prev_vel, _, _ = compute_79d_from_aggregator(
            self._agg, self._sfe, self._prev_vel, bar['timestamp'])

        if feat is not None:
            self._feat_buffer.append({
                'timestamp': bar['timestamp'],
                'features': feat.copy(),
            })

        return feat

    # ══════════════════════════════════════════════════════════════════
    # PROCESS NEW DAYS — batch mode (used by build_dataset)
    # ══════════════════════════════════════════════════════════════════

    def process_new_days(self, force_rebuild=False):
        """Find days with 5s bars but no features, process them.

        Loads checkpoint, feeds new 5s bars through aggregator,
        computes features, saves parquets + updated checkpoint.
        """
        os.makedirs(self._output_dir, exist_ok=True)

        # Load checkpoint for warm start
        if not force_rebuild and self.load_checkpoint():
            print(f'  Checkpoint loaded: last_ts={self._last_ts:.0f}')
        else:
            print(f'  Cold start (no checkpoint)')

        # Find days to process
        atlas_5s = os.path.join(self._atlas_root, '5s')
        if not os.path.exists(atlas_5s):
            print(f'  No 5s data in {atlas_5s}/')
            return

        all_days = sorted(f.replace('.parquet', '') for f in os.listdir(atlas_5s)
                          if f.endswith('.parquet'))

        to_process = []
        for day in all_days:
            feat_path = os.path.join(self._output_dir, f'{day}.parquet')
            if force_rebuild or not os.path.exists(feat_path):
                to_process.append(day)

        if not to_process:
            print(f'  Features up to date ({len(all_days)} days)')
            return

        print(f'  Processing: {len(to_process)} days '
              f'({to_process[0]} -> {to_process[-1]})')

        # Feed context days (before first day to process) for aggregator warmup
        first_process = to_process[0]
        context_days = [d for d in all_days if d < first_process]

        # Only feed context if aggregator is cold (no checkpoint)
        if self._last_ts == 0 and context_days:
            print(f'  Warming from {len(context_days)} context days...')
            for day in context_days:
                self._feed_day_5s(day)

        # Also feed higher TF context (1m, 5m, 15m, 1h, 1D)
        # These may extend further back than 5s
        if self._last_ts == 0:
            self._feed_higher_tf_context(first_process)

        # Process each day
        from tqdm import tqdm
        total_rows = 0

        for day in tqdm(to_process, desc='Days', unit='day'):
            rows = self._process_day(day)
            if rows:
                df = self._rows_to_df(rows)
                df.to_parquet(
                    os.path.join(self._output_dir, f'{day}.parquet'),
                    index=False)
                total_rows += len(df)

        print(f'  Done: {total_rows:,} rows across {len(to_process)} days')

        # Save checkpoint
        self.save_checkpoint()
        bar_counts = {tf: len(b) for tf, b in self._agg.history.items() if b}
        print(f'  Checkpoint saved: {bar_counts}')

    def _process_day(self, day_name):
        """Feed one day's 5s bars, collect feature rows."""
        fpath = os.path.join(self._atlas_root, '5s', f'{day_name}.parquet')
        if not os.path.exists(fpath):
            return []

        df = pd.read_parquet(fpath).sort_values('timestamp').reset_index(drop=True)
        rows = []

        for _, row in df.iterrows():
            bar = {
                'timestamp': row['timestamp'],
                'open': row['open'], 'high': row['high'],
                'low': row['low'], 'close': row['close'],
                'volume': row.get('volume', 0),
            }
            feat = self.process_bar(bar)
            if feat is not None:
                rows.append({'timestamp': bar['timestamp'], 'features': feat})

        return rows

    def _feed_day_5s(self, day_name):
        """Feed a day's 5s bars into aggregator (warmup, no features)."""
        fpath = os.path.join(self._atlas_root, '5s', f'{day_name}.parquet')
        if not os.path.exists(fpath):
            return
        df = pd.read_parquet(fpath).sort_values('timestamp').reset_index(drop=True)
        for _, row in df.iterrows():
            self._agg.feed({
                'timestamp': row['timestamp'],
                'open': row['open'], 'high': row['high'],
                'low': row['low'], 'close': row['close'],
                'volume': row.get('volume', 0),
            })
            self._last_ts = row['timestamp']

    def _feed_higher_tf_context(self, before_day):
        """Feed higher TF bars (1m, 5m, etc.) that exist before the first 5s day.

        This provides 1h/1D context from Databento data that predates NT8.
        """
        for tf in ['1m', '5m', '15m', '1h', '1D']:
            tf_dir = os.path.join(self._atlas_root, tf)
            if not os.path.exists(tf_dir):
                continue
            for f in sorted(os.listdir(tf_dir)):
                if not f.endswith('.parquet'):
                    continue
                day = f.replace('.parquet', '')
                if day >= before_day:
                    break
                df = pd.read_parquet(os.path.join(tf_dir, f))
                for _, row in df.iterrows():
                    # Feed directly into the TF's history (bypass aggregator)
                    self._agg.history[tf].append({
                        'timestamp': row['timestamp'],
                        'open': row['open'], 'high': row['high'],
                        'low': row['low'], 'close': row['close'],
                        'volume': row.get('volume', 0),
                    })
                # Trim to history limit
                if len(self._agg.history[tf]) > HISTORY_LIMIT:
                    self._agg.history[tf] = self._agg.history[tf][-HISTORY_LIMIT:]

    def _rows_to_df(self, rows):
        """Convert feature rows to DataFrame matching build_dataset schema."""
        data = {
            'timestamp': [r['timestamp'] for r in rows],
        }
        features = np.array([r['features'] for r in rows])
        for i, name in enumerate(FEATURE_NAMES_79D):
            data[name] = features[:, i] if i < features.shape[1] else 0.0
        df = pd.DataFrame(data)
        df['timestamp'] = df['timestamp'].astype(np.int64)
        return df

    # ══════════════════════════════════════════════════════════════════
    # ACCESSORS
    # ══════════════════════════════════════════════════════════════════

    @property
    def aggregator(self):
        return self._agg

    @property
    def prev_velocities(self):
        return self._prev_vel

    @property
    def last_ts(self):
        return self._last_ts
