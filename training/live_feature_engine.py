"""
LiveFeatureEngine — computes 91D features using the SAME path as build_dataset.

Guarantees parity with training features by using identical SFE batch computation.
Loads pre-built TF bars from ATLAS_NT8 at startup, appends new bars from live feed,
runs batch SFE per TF only when that TF gets a new bar.

Usage (live):
    lfe = LiveFeatureEngine('DATA/ATLAS_NT8')
    lfe.load_history()               # load pre-built bars
    ...
    feat = lfe.on_bar(bar_5s)        # returns 91D or None

Usage (batch rebuild):
    lfe = LiveFeatureEngine('DATA/ATLAS_NT8')
    lfe.load_history()
    for bar in day_bars:
        feat = lfe.on_bar(bar)       # same function, same path
"""
import os
import glob
import numpy as np
import pandas as pd
from typing import Optional, Dict, List

from core.statistical_field_engine import StatisticalFieldEngine
from core.features import extract_features, FEATURE_NAMES, TF_ORDER

SFE_WINDOW = 300
SFE_MIN_BARS = 21

# TF periods in seconds (for bar aggregation)
TF_SECONDS = {
    '15s': 15, '30s': 30, '1m': 60, '5m': 300,
    '15m': 900, '1h': 3600, '1D': 86400,
}


class LiveFeatureEngine:
    """Compute 91D features — same SFE batch path as build_dataset."""

    def __init__(self, atlas_root: str = 'DATA/ATLAS_NT8'):
        self._atlas_root = atlas_root
        self._sfe = StatisticalFieldEngine()
        self._prev_vel: Dict[str, float] = {}

        # Per-TF bar stores: list of {timestamp, open, high, low, close, volume}
        self._bars: Dict[str, pd.DataFrame] = {}

        # Per-TF SFE state cache: (n_bars, states_list, sfe_start_offset)
        self._sfe_cache: Dict[str, tuple] = {}

        # Per-TF bar count at last SFE run (detect new bars)
        self._sfe_bar_count: Dict[str, int] = {}

        # 5s bar accumulator for higher TFs
        self._accumulators: Dict[str, dict] = {}
        for tf, secs in TF_SECONDS.items():
            self._accumulators[tf] = {
                'tf_seconds': secs, 'current_start': 0,
                'open': 0, 'high': -1e18, 'low': 1e18,
                'close': 0, 'volume': 0, 'count': 0,
            }

    # ══════════════════════════════════════════════════════════════════
    # LOAD — pre-built bars from ATLAS_NT8
    # ══════════════════════════════════════════════════════════════════

    def load_history(self):
        """Load all pre-built TF bars from ATLAS_NT8 into memory.

        Sets _last_loaded_ts per TF so on_bar only appends NEW bars.
        """
        self._last_loaded_ts: Dict[str, float] = {}

        # Load all TFs including 5s
        all_tfs = ['5s'] + list(TF_ORDER)
        for tf in all_tfs:
            tf_dir = os.path.join(self._atlas_root, tf)
            if not os.path.exists(tf_dir):
                continue
            files = sorted(glob.glob(os.path.join(tf_dir, '*.parquet')))
            if not files:
                continue
            dfs = [pd.read_parquet(f) for f in files]
            full = pd.concat(dfs, ignore_index=True).sort_values(
                'timestamp').reset_index(drop=True)
            self._bars[tf] = full
            self._sfe_bar_count[tf] = 0
            self._last_loaded_ts[tf] = float(full['timestamp'].iloc[-1])

        bar_counts = {tf: len(df) for tf, df in self._bars.items()}
        return bar_counts

    def load_velocities(self, velocities: dict):
        """Restore prev_velocities from checkpoint."""
        self._prev_vel = dict(velocities)

    # ══════════════════════════════════════════════════════════════════
    # ON_BAR — feed one 5s bar, get features back
    # ══════════════════════════════════════════════════════════════════

    def on_bar(self, bar: dict) -> Optional[np.ndarray]:
        """Feed one 5s bar. Dedupes replays (NT8 reconnect flood).

        For any bar we have NOT seen before (ts > highest in 5s store),
        append it and aggregate to higher TFs. Otherwise skip to avoid
        double-counting volumes or corrupting the SFE window.
        """
        ts = bar['timestamp']

        # Check if we already have this bar (pre-loaded or already appended)
        if '5s' in self._bars and len(self._bars['5s']) > 0:
            highest_seen = float(self._bars['5s']['timestamp'].iloc[-1])
        else:
            highest_seen = 0

        is_new = ts > highest_seen

        if is_new:
            self._append_bar('5s', bar)
            for tf in TF_ORDER:
                if tf == '5s':
                    continue
                completed = self._aggregate_bar(tf, bar)
                if completed:
                    self._append_bar(tf, completed)

        # Compute features either way — but if not new, return None to signal
        # "no new data" so caller skips the feature save
        if not is_new:
            return None
        return self._compute_features(ts)

    # ══════════════════════════════════════════════════════════════════
    # INTERNAL — bar management + SFE
    # ══════════════════════════════════════════════════════════════════

    def _append_bar(self, tf: str, bar: dict):
        """Append a completed bar to the TF store."""
        row = pd.DataFrame([{
            'timestamp': bar['timestamp'],
            'open': bar['open'], 'high': bar['high'],
            'low': bar['low'], 'close': bar['close'],
            'volume': bar.get('volume', 0),
        }])
        if tf not in self._bars:
            self._bars[tf] = row
        else:
            self._bars[tf] = pd.concat(
                [self._bars[tf], row], ignore_index=True)
            # Trim to keep memory bounded (keep last 5000 bars per TF)
            if len(self._bars[tf]) > 5000:
                self._bars[tf] = self._bars[tf].iloc[-5000:].reset_index(
                    drop=True)

    def _aggregate_bar(self, tf: str, bar_5s: dict) -> Optional[dict]:
        """Accumulate a 5s bar into a higher TF. Returns completed bar or None."""
        acc = self._accumulators[tf]
        ts = bar_5s['timestamp']
        boundary = (int(ts) // acc['tf_seconds']) * acc['tf_seconds']

        completed = None

        # Crossed into new TF period — close previous
        if acc['count'] > 0 and boundary != acc['current_start']:
            completed = {
                'timestamp': acc['current_start'],
                'open': acc['open'], 'high': acc['high'],
                'low': acc['low'], 'close': acc['close'],
                'volume': acc['volume'],
            }
            acc['current_start'] = boundary
            acc['open'] = bar_5s['open']
            acc['high'] = bar_5s['high']
            acc['low'] = bar_5s['low']
            acc['close'] = bar_5s['close']
            acc['volume'] = bar_5s['volume']
            acc['count'] = 1
        else:
            if acc['count'] == 0:
                acc['current_start'] = boundary
                acc['open'] = bar_5s['open']
            acc['high'] = max(acc['high'], bar_5s['high'])
            acc['low'] = min(acc['low'], bar_5s['low'])
            acc['close'] = bar_5s['close']
            acc['volume'] = acc.get('volume', 0) + bar_5s['volume']
            acc['count'] += 1

        return completed

    def _compute_features(self, ts: float) -> Optional[np.ndarray]:
        """Compute 91D using batch SFE — same windowing as process_one_day.

        Window: SFE_WINDOW bars before today's start + all bars up to now.
        This matches build_dataset exactly: warmup(300) + full day so far.
        """
        states_by_tf = {}
        ohlcv_by_tf = {}

        for tf in TF_ORDER:
            if tf not in self._bars:
                continue
            df = self._bars[tf]
            n_bars = len(df)
            if n_bars < SFE_MIN_BARS:
                continue

            ts_arr = df['timestamp'].values

            # Only bars <= current timestamp (no lookahead)
            valid_idx = int(np.searchsorted(ts_arr, ts, side='right'))
            if valid_idx < SFE_MIN_BARS:
                continue

            # Match process_one_day windowing:
            # sfe_start = max(0, today_start - SFE_WINDOW)
            # sfe_input = cumul[sfe_start : valid_idx]
            # This gives warmup(300) + all today's bars in one SFE pass
            today_start = self._find_today_start(tf, ts)
            warmup = min(SFE_WINDOW, today_start)
            sfe_start = max(0, today_start - warmup)
            sfe_input = df.iloc[sfe_start:valid_idx].reset_index(drop=True)

            # Check cache: rerun SFE only if valid_idx changed for this TF
            cached = self._sfe_cache.get(tf)
            if cached and cached[0] == valid_idx:
                states = cached[1]
                sfe_offset = cached[2]
            else:
                states = self._sfe.batch_compute_states(sfe_input)
                if not states:
                    continue
                sfe_offset = sfe_start
                self._sfe_cache[tf] = (valid_idx, states, sfe_offset)

            # Find state for latest bar <= ts
            state_idx = valid_idx - 1 - sfe_offset
            if state_idx < 0 or state_idx >= len(states):
                continue

            states_by_tf[tf] = states[state_idx]

            # OHLCV for extract_features (velocity, helpers)
            ohlcv = df.iloc[:valid_idx]
            if len(ohlcv) > SFE_WINDOW:
                ohlcv = ohlcv.tail(SFE_WINDOW).reset_index(drop=True)
            ohlcv_by_tf[tf] = ohlcv

        if '1m' not in states_by_tf and '5s' not in states_by_tf:
            return None

        feat, self._prev_vel = extract_features(
            states_by_tf, ohlcv_by_tf, self._prev_vel, ts)

        return feat

    def _find_today_start(self, tf: str, ts: float) -> int:
        """Find the index where today's bars start for this TF.

        'Today' = the UTC day of timestamp ts.
        Returns the index of the first bar on this day.
        """
        df = self._bars[tf]
        ts_arr = df['timestamp'].values

        # Day boundary in seconds (UTC midnight)
        day_start_ts = (int(ts) // 86400) * 86400
        idx = int(np.searchsorted(ts_arr, day_start_ts, side='left'))
        return idx

    # ══════════════════════════════════════════════════════════════════
    # ACCESSORS
    # ══════════════════════════════════════════════════════════════════

    @property
    def prev_velocities(self):
        return self._prev_vel

    @property
    def bar_counts(self):
        return {tf: len(df) for tf, df in self._bars.items()}
