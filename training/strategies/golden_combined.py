"""Combined entry-timing + direction classifier strategy.

At each 1m close:
  1. P_timing  = P(this bar is a golden entry moment) from entry-timing model
  2. If P_timing > T_timing:
  3.   P_long  = P(LONG | V2 features) from direction classifier
  4.   confidence = max(P_long, 1 - P_long)
  5.   If confidence > T_dir: fire in predicted direction

Two thresholds tune independently:
  T_timing  — how rare/precise the entry signal must be
  T_dir     — how confident the direction call must be

Both classifiers were trained on full IS through their respective pipelines.
"""
from __future__ import annotations

import os
import pickle
from typing import Optional

import numpy as np

from training.utils.state import BarState
from training.strategies.base import EntrySignal, Strategy


DEFAULT_TIMING_PKL = 'training_iso_v2/output/golden_entry_clf_lr.pkl'
DEFAULT_TIMING_CACHE = None   # optional: parquet with (timestamp, day, p_timing)
DEFAULT_DIR_PKL = 'training_iso_v2/output/direction_clf_pivot_gbm.pkl'
DEFAULT_T_TIMING = 0.85
DEFAULT_T_DIR = 0.65
DEFAULT_FIRE_CADENCE = '1m'


def _sigmoid(z: float) -> float:
    if z >= 0:
        return 1.0 / (1.0 + np.exp(-z))
    ez = np.exp(z)
    return float(ez / (1.0 + ez))


class GoldenCombinedStrategy(Strategy):
    """Fires when both entry-timing AND direction confidence exceed thresholds."""

    name = 'GOLDEN_COMBINED'

    def __init__(self, timing_pkl: str = DEFAULT_TIMING_PKL,
                 timing_cache: str = DEFAULT_TIMING_CACHE,
                 dir_pkl: str = DEFAULT_DIR_PKL,
                 t_timing: float = DEFAULT_T_TIMING,
                 t_dir: float = DEFAULT_T_DIR,
                 fire_cadence: str = DEFAULT_FIRE_CADENCE):
        # If timing_cache is provided, use cached P_timing (fast, supports GBM).
        # Otherwise load the model pickle for live inference.
        self._timing_cache = None
        if timing_cache:
            import pandas as pd
            cache_df = pd.read_parquet(timing_cache)
            # Build {(day, ts) -> p_timing} lookup
            self._timing_cache = {
                (d, int(ts)): float(p) for d, ts, p in zip(
                    cache_df['day'], cache_df['timestamp'], cache_df['p_timing']
                )
            }
            # Also keep ts-only fallback (most days disjoint)
            self._timing_cache_ts = {
                int(ts): float(p) for ts, p in zip(cache_df['timestamp'], cache_df['p_timing'])
            }
            self.timing_kind = 'cache'
            self.timing_cols = []
        else:
            if not os.path.exists(timing_pkl):
                raise FileNotFoundError(timing_pkl)
            with open(timing_pkl, 'rb') as f:
                tp = pickle.load(f)
            self.timing_scaler = tp.get('scaler')
            self.timing_clf = tp['clf']
            self.timing_cols = list(tp['v2_cols'])
            self.timing_kind = tp.get('model_kind', 'lr')
            if self.timing_kind == 'lr':
                self._tim_mean = self.timing_scaler.mean_.astype(np.float32)
                self._tim_scale = self.timing_scaler.scale_.astype(np.float32)
                self._tim_coef = self.timing_clf.coef_[0].astype(np.float32)
                self._tim_intercept = float(self.timing_clf.intercept_[0])

        if not os.path.exists(dir_pkl):
            raise FileNotFoundError(dir_pkl)
        with open(dir_pkl, 'rb') as f:
            dp = pickle.load(f)

        self.dir_scaler = dp['scaler']
        self.dir_clf = dp['clf']
        self.dir_cols = list(dp['v2_cols'])
        # Detect LR vs GBM. LR has scaler + coef; GBM has no scaler.
        self.dir_kind = dp.get('model_kind',
                                'lr' if self.dir_scaler is not None else 'gbm')
        if self.dir_kind == 'lr':
            self._dir_mean = self.dir_scaler.mean_.astype(np.float32)
            self._dir_scale = self.dir_scaler.scale_.astype(np.float32)
            self._dir_coef = self.dir_clf.coef_[0].astype(np.float32)
            self._dir_intercept = float(self.dir_clf.intercept_[0])

        self.t_timing = float(t_timing)
        self.t_dir = float(t_dir)
        self.fire_cadence = fire_cadence

    def _ready(self, state: BarState) -> bool:
        return {
            '5s': True,
            '15s': state.bar_idx % 3 == 0,
            '1m': state.is_1m_close,
            '5m': state.is_5m_close,
            '15m': state.is_15m_close,
        }.get(self.fire_cadence, False)

    def _build_vec(self, state: BarState, cols):
        v2 = state.v2
        x = np.empty(len(cols), dtype=np.float32)
        for i, c in enumerate(cols):
            v = v2.get(c, 0.0)
            x[i] = v if v == v else 0.0
        return x

    def _p_timing(self, state: BarState) -> float:
        if self._timing_cache is not None:
            # Lookup by (day, ts); fallback to ts-only if day mismatches
            key = (state.day, int(state.timestamp))
            if key in self._timing_cache:
                return self._timing_cache[key]
            # ts-only fallback (most timestamps disjoint across days)
            return self._timing_cache_ts.get(int(state.timestamp), 0.0)
        x = self._build_vec(state, self.timing_cols)
        if self.timing_kind == 'lr':
            xs = (x - self._tim_mean) / np.where(self._tim_scale > 1e-8,
                                                  self._tim_scale, 1.0)
            z = float(np.dot(xs, self._tim_coef) + self._tim_intercept)
            return _sigmoid(z)
        return float(self.timing_clf.predict_proba(x.reshape(1, -1))[0, 1])

    def _p_long(self, state: BarState) -> float:
        x = self._build_vec(state, self.dir_cols)
        if self.dir_kind == 'lr':
            xs = (x - self._dir_mean) / np.where(self._dir_scale > 1e-8,
                                                  self._dir_scale, 1.0)
            z = float(np.dot(xs, self._dir_coef) + self._dir_intercept)
            return _sigmoid(z)
        # GBM — slower per-row inference; acceptable given firing cadence
        return float(self.dir_clf.predict_proba(x.reshape(1, -1))[0, 1])

    def evaluate(self, state: BarState) -> Optional[EntrySignal]:
        if not self._ready(state):
            return None
        p_t = self._p_timing(state)
        if p_t < self.t_timing:
            return None
        p_d = self._p_long(state)
        conf = max(p_d, 1.0 - p_d)
        if conf < self.t_dir:
            return None
        direction = 'long' if p_d >= 0.5 else 'short'
        return EntrySignal(direction=direction, tier=self.name,
                           extras={'p_timing': p_t, 'p_long': p_d, 'conf': conf})
