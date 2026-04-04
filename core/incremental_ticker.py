"""
Incremental Ticker — feeds bars one at a time, zero lookahead by construction.

This is the DATA PIPELINE. It never sees the future. It maintains:
  - Running 1m bar accumulator
  - Partial higher TF bars (built from closed 1m bars only)
  - SFE states per TF (recomputed on rolling windows, not full history)
  - 79D feature vector per bar

The ticker emits a state_dict per bar. The wrapper (AI) consumes it.

Architecture:
  Ticker (this module) -> state_dict -> Wrapper (strategy_ticker.py)

  Ticker is DUMB and SECURE:
    - Appends one bar, recomputes, emits state
    - Cannot cheat — only has bars up to current
    - SFE runs on rolling window, not future data

  Wrapper is SMART:
    - Reads state_dict
    - Calls NN + Brain
    - Makes trade decisions

Performance:
  - SFE on rolling window (300 bars) not full history (25K bars)
  - Higher TFs only recompute when they get a new bar
  - GPU memory cleared between days
  - Pre-allocated numpy arrays, lazy DataFrame rebuild
"""
import gc
import numpy as np
import pandas as pd
import torch
from typing import Dict, Optional
from core.statistical_field_engine import StatisticalFieldEngine
from core.features_79d import (
    extract_tf_features, extract_79d, build_all_tf_ohlcv,
    aggregate_partial_bar,
    TF_ORDER, TF_SECONDS, N_FEATURES, FEATURE_NAMES_79D,
)

# Minimum bars for SFE regression
SFE_MIN_BARS = 21

# Rolling window for SFE: enough for regression context but not full history
# 300 1m bars = 5 hours — sufficient for regression and indicators
SFE_ROLLING_WINDOW = 300


class IncrementalTicker:
    """Zero-lookahead bar-by-bar ticker.

    Usage:
        ticker = IncrementalTicker(history_1m=prev_days_df)

        for bar in todays_1m_bars:  # fed one at a time
            state = ticker.feed_bar(bar)
            # state contains 79D features + metadata
            # pass state to wrapper for AI decisions

        ticker.cleanup()  # free GPU memory at end of day
    """

    def __init__(self, history_1m: pd.DataFrame = None, max_today_bars: int = 1500):
        self.sfe = StatisticalFieldEngine()

        # History: keep full history for SFE consistency with training
        if history_1m is not None and len(history_1m) > 0:
            self._history = history_1m.sort_values('timestamp').reset_index(drop=True)
            self._hist_len = len(self._history)
        else:
            self._history = pd.DataFrame(columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            self._hist_len = 0

        # Pre-allocated arrays for today's bars
        self._max_bars = max_today_bars
        self._today_ts = np.zeros(max_today_bars, dtype=np.float64)
        self._today_open = np.zeros(max_today_bars, dtype=np.float64)
        self._today_high = np.zeros(max_today_bars, dtype=np.float64)
        self._today_low = np.zeros(max_today_bars, dtype=np.float64)
        self._today_close = np.zeros(max_today_bars, dtype=np.float64)
        self._today_vol = np.zeros(max_today_bars, dtype=np.float64)
        self._today_count = 0

        # Lazy DataFrame
        self._rolling_df = None
        self._rolling_dirty = True

        # SFE state cache
        self._sfe_cache = {}
        self._tf_bar_counts = {}
        self._prev_velocities = {}

        # Higher TF aggregation cache (avoid recomputing every bar)
        self._tf_agg_cache = {}

        # Warmup: run SFE on history once
        self._warmup()

    def _warmup(self):
        """Run SFE on history for all TFs. One-time cost."""
        if self._hist_len < SFE_MIN_BARS:
            return

        # 1m
        states = self.sfe.batch_compute_states(self._history)
        if states:
            self._sfe_cache['1m'] = states[-1]
            self._tf_bar_counts['1m'] = self._hist_len

        # Higher TFs from history
        for tf in ['5m', '15m', '1h', '1D']:
            tf_sec = TF_SECONDS[tf]
            tf_bars = aggregate_partial_bar(self._history, tf_sec)
            if len(tf_bars) >= SFE_MIN_BARS:
                tf_states = self.sfe.batch_compute_states(tf_bars)
                if tf_states:
                    self._sfe_cache[tf] = tf_states[-1]
                    self._tf_bar_counts[tf] = len(tf_bars)

    def _get_closed_1m(self) -> pd.DataFrame:
        """Get all closed 1m bars (history + today). Rebuilt only when dirty."""
        if not self._rolling_dirty and self._rolling_df is not None:
            return self._rolling_df

        n = self._today_count
        if n == 0:
            self._rolling_df = self._history
        else:
            today_df = pd.DataFrame({
                'timestamp': self._today_ts[:n],
                'open': self._today_open[:n],
                'high': self._today_high[:n],
                'low': self._today_low[:n],
                'close': self._today_close[:n],
                'volume': self._today_vol[:n],
            })
            if self._hist_len > 0:
                self._rolling_df = pd.concat([self._history, today_df], ignore_index=True)
            else:
                self._rolling_df = today_df

        self._rolling_dirty = False
        return self._rolling_df

    def feed_bar(self, bar: dict) -> dict:
        """Feed one CLOSED 1m bar. Returns state_dict with 79D features."""
        i = self._today_count
        self._today_ts[i] = bar['timestamp']
        self._today_open[i] = bar['open']
        self._today_high[i] = bar['high']
        self._today_low[i] = bar['low']
        self._today_close[i] = bar['close']
        self._today_vol[i] = bar['volume']
        self._today_count += 1
        self._rolling_dirty = True
        ts = bar['timestamp']
        price = bar['close']

        # Rolling 1m window for SFE
        rolling_1m = self._get_closed_1m()
        n_1m = len(rolling_1m)

        # Aggregate higher TFs from rolling window
        tf_agg = {}
        for tf in ['5m', '15m', '1h', '1D']:
            tf_sec = TF_SECONDS[tf]
            tf_agg[tf] = aggregate_partial_bar(rolling_1m, tf_sec)

        # SFE: 1m runs every bar on rolling window (not full history)
        states_by_tf = {}
        active_tfs = []

        if n_1m >= SFE_MIN_BARS:
            tf_states = self.sfe.batch_compute_states(rolling_1m)
            if tf_states:
                self._sfe_cache['1m'] = tf_states[-1]
                states_by_tf['1m'] = tf_states[-1]
                active_tfs.append('1m')

        # Higher TFs: only recompute when bar count changed
        for tf in ['5m', '15m', '1h', '1D']:
            bars = tf_agg[tf]
            n_tf = len(bars)

            if n_tf >= SFE_MIN_BARS and n_tf != self._tf_bar_counts.get(tf, 0):
                tf_states = self.sfe.batch_compute_states(bars)
                if tf_states:
                    self._sfe_cache[tf] = tf_states[-1]
                    self._tf_bar_counts[tf] = n_tf
                    states_by_tf[tf] = tf_states[-1]
                    active_tfs.append(tf)
            elif tf in self._sfe_cache:
                states_by_tf[tf] = self._sfe_cache[tf]
                active_tfs.append(tf)

        # OHLCV for feature extraction
        ohlcv_by_tf = {'1m': rolling_1m}
        for tf in ['5m', '15m', '1h', '1D']:
            if len(tf_agg[tf]) > 0:
                ohlcv_by_tf[tf] = tf_agg[tf]

        # 79D features
        features, self._prev_velocities = extract_79d(
            states_by_tf, ohlcv_by_tf, self._prev_velocities, ts
        )

        return {
            'features_79d': features,
            'timestamp': ts,
            'price': price,
            'bar_idx': self._today_count - 1,
            'states_by_tf': states_by_tf,
            'active_tfs': active_tfs,
            'metadata': {
                'n_1m_bars': n_1m,
                'n_today_bars': self._today_count,
                'active_tfs': len(active_tfs),
            },
        }

    def cleanup(self):
        """Free GPU memory and caches. Call at end of day."""
        self._sfe_cache.clear()
        self._tf_bar_counts.clear()
        self._tf_agg_cache.clear()
        self._rolling_df = None
        del self.sfe
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self.sfe = StatisticalFieldEngine()

    def get_bar_count(self) -> int:
        return self._today_count
