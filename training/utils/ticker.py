"""V2-native ticker.

Yields BarState per 5s bar by joining:
  - V2 layered features  : core_v2.features.load_features([day])
  - 5s OHLCV             : DATA/ATLAS/5s/{day}.parquet
  - 1m OHLCV             : DATA/ATLAS/1m/{day}.parquet (for price + 1m boundary detection)
  - 2D regime label      : DATA/ATLAS/regime_labels_2d.csv

No V1 conversion, no compat cache, no v1_compat shim.
"""
from __future__ import annotations

import os
import glob
from typing import Iterator, List, Optional

import numpy as np
import pandas as pd

from core_v2.features import (FEATURE_NAMES, N_FEATURES, load_features,
                                    DEFAULT_FEATURES_ROOT)
from training.utils.state import (BarState, regime_to_idx,
                                                  classify_regime_from_velocity)
from training.utils.v2_cols import price_velocity_w, swing_noise_w


ATLAS_ROOT = 'DATA/ATLAS'
LABELS_CSV = 'DATA/ATLAS/regime_labels_2d.csv'
HOURLY_LABELS_CSV = 'DATA/ATLAS/regime_labels_hourly.csv'
VEL_REGIME_THR_JSON = 'training_iso_v2/output/velocity_regime_thresholds.json'

# 5s bar period — this is the anchor cadence of the V2 features
ANCHOR_PERIOD_S = 5

# Regime resolution mode:
#   'velocity' — per-bar from L2 velocity + L3 swing_noise (PREFERRED, no batch step)
#   'hourly'   — per-bar from last-completed-hour batch label (forward-pass)
#   'daily'    — full-day label (LOOKAHEAD; legacy)
# Default = velocity if its threshold JSON exists, else hourly, else daily.
def _default_regime_mode():
    if os.path.exists(VEL_REGIME_THR_JSON):
        return 'velocity'
    if os.path.exists(HOURLY_LABELS_CSV):
        return 'hourly'
    return 'daily'
DEFAULT_REGIME_MODE = _default_regime_mode()


_VEL_REGIME_THR: Optional[dict] = None


def _load_vel_regime_thresholds(path: str = VEL_REGIME_THR_JSON) -> dict:
    """Load velocity-regime thresholds JSON with sensible defaults."""
    global _VEL_REGIME_THR
    if _VEL_REGIME_THR is None:
        try:
            import json
            with open(path, 'r') as f:
                _VEL_REGIME_THR = json.load(f)
        except Exception:
            _VEL_REGIME_THR = {'tf': '1h', 'vel_thr': 1.0, 'sn_thr': 100.0}
    return _VEL_REGIME_THR


_REGIME_CACHE: Optional[dict] = None
_HOURLY_REGIME_CACHE: Optional[dict] = None   # day -> sorted [(hour_end_ts, regime), ...]


def _load_regime_lookup(labels_csv: str = LABELS_CSV) -> dict:
    """Daily regime lookup — LEGACY (full-day stat = lookahead at intraday)."""
    global _REGIME_CACHE
    if _REGIME_CACHE is None:
        try:
            df = pd.read_csv(labels_csv)
            df['date'] = df['date'].astype(str).str[:10]
            _REGIME_CACHE = dict(zip(df['date'], df['regime_2d']))
        except Exception:
            _REGIME_CACHE = {}
    return _REGIME_CACHE


def _load_hourly_regime_lookup(csv: str = HOURLY_LABELS_CSV) -> dict:
    """Hourly regime lookup — forward-pass-honest. Returns
    {day_str: list of (hour_end_ts, regime_2d)} sorted by hour_end_ts.

    At trade-time, look up the most recent (hour_end_ts <= current_ts) and
    use its regime. The label was computed from bars within
    [hour_end_ts - 3600, hour_end_ts] — strictly past data.
    """
    global _HOURLY_REGIME_CACHE
    if _HOURLY_REGIME_CACHE is None:
        try:
            df = pd.read_csv(csv)
            _HOURLY_REGIME_CACHE = {}
            for day, sub in df.groupby('day'):
                sub = sub.sort_values('hour_end_ts')
                _HOURLY_REGIME_CACHE[str(day)] = list(zip(
                    sub['hour_end_ts'].astype(int).tolist(),
                    sub['regime_2d'].astype(str).tolist(),
                ))
        except Exception:
            _HOURLY_REGIME_CACHE = {}
    return _HOURLY_REGIME_CACHE


def _normalize_ts(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce timestamp column to int64 Unix seconds in-place."""
    if 'timestamp' not in df.columns:
        return df
    if pd.api.types.is_datetime64_any_dtype(df['timestamp']):
        df = df.copy()
        df['timestamp'] = (df['timestamp'].astype('int64') // 10**9)
    return df


def _read_ohlcv(path: str) -> Optional[pd.DataFrame]:
    if not os.path.exists(path):
        return None
    df = pd.read_parquet(path)
    return _normalize_ts(df).sort_values('timestamp').reset_index(drop=True)


class V2Ticker:
    """Single-day V2-native ticker.

    Usage:
        for state in V2Ticker('2025_06_15'):
            ...  # state: BarState
    """

    def __init__(self, day: str, atlas_root: str = ATLAS_ROOT,
                 features_root: str = DEFAULT_FEATURES_ROOT,
                 labels_csv: str = LABELS_CSV,
                 regime_mode: str = DEFAULT_REGIME_MODE):
        self.day = day
        # Load V2 features (185 cols + timestamp). Anchor cadence = 5s.
        feats = load_features(days=[day], root=features_root)
        if feats.empty:
            raise FileNotFoundError(f'No V2 features for {day} under {features_root}')
        feats = _normalize_ts(feats).sort_values('timestamp').reset_index(drop=True)
        self._feats = feats
        self._n = len(feats)

        # 5s + 1m OHLCV (required) + 5m / 15m / 1h (optional, for wick tiers)
        self._ohlcv_5s = _read_ohlcv(os.path.join(atlas_root, '5s', f'{day}.parquet'))
        self._ohlcv_1m = _read_ohlcv(os.path.join(atlas_root, '1m', f'{day}.parquet'))
        self._ohlcv_5m = _read_ohlcv(os.path.join(atlas_root, '5m', f'{day}.parquet'))
        self._ohlcv_15m = _read_ohlcv(os.path.join(atlas_root, '15m', f'{day}.parquet'))
        self._ohlcv_1h = _read_ohlcv(os.path.join(atlas_root, '1h', f'{day}.parquet'))

        if self._ohlcv_5s is None or len(self._ohlcv_5s) == 0:
            raise FileNotFoundError(f'No 5s OHLCV for {day}')
        if self._ohlcv_1m is None or len(self._ohlcv_1m) == 0:
            raise FileNotFoundError(f'No 1m OHLCV for {day}')

        # Regime
        #   'velocity' — per-bar from L2 velocity + L3 swing_noise (PREFERRED)
        #   'hourly'   — per-bar from last-completed-hour batch label
        #   'daily'    — full-day label (LOOKAHEAD; legacy)
        self._regime_mode = regime_mode
        iso = day.replace('_', '-')

        if regime_mode == 'daily':
            regime = _load_regime_lookup(labels_csv).get(iso, 'UNKNOWN')
            self._regime_2d = str(regime)
            self._regime_idx = regime_to_idx(self._regime_2d)
            self._hourly_endpts = None
            self._hourly_regimes = None
        elif regime_mode == 'hourly':
            entries = _load_hourly_regime_lookup().get(day, [])
            self._hourly_endpts = (np.asarray([e[0] for e in entries],
                                                          dtype=np.int64)
                                            if entries else None)
            self._hourly_regimes = ([e[1] for e in entries]
                                                if entries else None)
            self._regime_2d = 'WARMUP'
            self._regime_idx = regime_to_idx('WARMUP')
        else:  # 'velocity'
            self._hourly_endpts = None
            self._hourly_regimes = None
            self._regime_2d = 'WARMUP'
            self._regime_idx = regime_to_idx('WARMUP')

        # Velocity-mode thresholds (used by _regime_at)
        thr = _load_vel_regime_thresholds()
        self._vel_thr = float(thr.get('vel_thr', 1.0))
        self._sn_thr = float(thr.get('sn_thr', 100.0))
        self._vel_tf = thr.get('tf', '1h')
        self._vel_col_name = price_velocity_w(self._vel_tf)
        self._sn_col_name = swing_noise_w(self._vel_tf)
        # Pre-resolve column indices in v2 vector for fast per-bar lookup
        self._vel_col_idx = (FEATURE_NAMES.index(self._vel_col_name)
                                      if self._vel_col_name in FEATURE_NAMES else -1)
        self._sn_col_idx = (FEATURE_NAMES.index(self._sn_col_name)
                                     if self._sn_col_name in FEATURE_NAMES else -1)

        # Pre-extract V2 vector matrix (n × 185) in canonical FEATURE_NAMES order
        # Missing cols → 0 (warmup). Defends against schema drift.
        v2_matrix = np.zeros((self._n, N_FEATURES), dtype=np.float32)
        feat_cols = set(feats.columns)
        for j, name in enumerate(FEATURE_NAMES):
            if name in feat_cols:
                v2_matrix[:, j] = feats[name].values.astype(np.float32)
        self._v2_matrix = v2_matrix

        # Pre-extract OHLCV arrays for fast searchsorted
        self._ts5s = self._ohlcv_5s['timestamp'].values.astype(np.int64)
        self._o5 = self._ohlcv_5s['open'].values
        self._h5 = self._ohlcv_5s['high'].values
        self._l5 = self._ohlcv_5s['low'].values
        self._c5 = self._ohlcv_5s['close'].values
        self._v5 = self._ohlcv_5s['volume'].values

        self._ts1m = self._ohlcv_1m['timestamp'].values.astype(np.int64)
        self._o1 = self._ohlcv_1m['open'].values
        self._h1 = self._ohlcv_1m['high'].values
        self._l1 = self._ohlcv_1m['low'].values
        self._c1 = self._ohlcv_1m['close'].values
        self._v1 = self._ohlcv_1m['volume'].values

        # Multi-TF OHLCV arrays (for wick tiers); empty arrays if missing
        def _arrs(df):
            if df is None or len(df) == 0:
                return (np.zeros(0, dtype=np.int64),) + tuple(
                    np.zeros(0) for _ in range(5))
            return (df['timestamp'].values.astype(np.int64),
                       df['open'].values, df['high'].values,
                       df['low'].values, df['close'].values,
                       df['volume'].values)
        (self._ts5m, self._o5m, self._h5m, self._l5m, self._c5m, self._v5m
            ) = _arrs(self._ohlcv_5m)
        (self._ts15m, self._o15m, self._h15m, self._l15m, self._c15m, self._v15m
            ) = _arrs(self._ohlcv_15m)
        (self._ts1h, self._o1h, self._h1h, self._l1h, self._c1h, self._v1h
            ) = _arrs(self._ohlcv_1h)

    def __len__(self) -> int:
        return self._n

    def _regime_at(self, ts: int, v2_vec: np.ndarray = None) -> tuple:
        """Look up regime active at timestamp `ts`.

        velocity: classify_regime_from_velocity(v2_vec) — uses per-bar L2
                  velocity + L3 swing_noise. Truly forward-pass-honest:
                  rolling-window features by construction use only past data.
        hourly  : last-completed-hour batch label. Forward-pass-honest.
        daily   : static day-level (LOOKAHEAD).
        """
        if self._regime_mode == 'velocity':
            if v2_vec is None or self._vel_col_idx < 0 or self._sn_col_idx < 0:
                return 'WARMUP', regime_to_idx('WARMUP')
            v = float(v2_vec[self._vel_col_idx])
            s = float(v2_vec[self._sn_col_idx])
            r = classify_regime_from_velocity(v, s, self._vel_thr, self._sn_thr)
            return r, regime_to_idx(r)
        if self._regime_mode == 'daily':
            return self._regime_2d, self._regime_idx
        if self._hourly_endpts is None or len(self._hourly_endpts) == 0:
            return 'WARMUP', regime_to_idx('WARMUP')
        idx = int(np.searchsorted(self._hourly_endpts, ts, side='right')) - 1
        if idx < 0:
            return 'WARMUP', regime_to_idx('WARMUP')
        regime_str = self._hourly_regimes[idx]
        return regime_str, regime_to_idx(regime_str)

    def __iter__(self) -> Iterator[BarState]:
        ts_arr = self._feats['timestamp'].values.astype(np.int64)

        # Pre-compute boundary flags. ATLAS uses ts%60==0 (bar start), NT8 uses
        # ts%60==59 (bar end). Detect from the actual 1m parquet's timestamps,
        # which are the SOURCE OF TRUTH for what counts as a 1m close.
        if self._ts1m is not None and len(self._ts1m) > 0:
            ts1m_set = set(int(t) for t in self._ts1m)
            is_1m = np.array([int(t) in ts1m_set for t in ts_arr], dtype=bool)
            # 5m/15m/1h boundaries: derive offset from the 1m mod
            mod = int(self._ts1m[0]) % 60   # 0 for ATLAS, 59 for NT8
            is_5m = ((ts_arr - mod) % 300) == 0
            is_15m = ((ts_arr - mod) % 900) == 0
            is_1h = ((ts_arr - mod) % 3600) == 0
        else:
            is_1m = (ts_arr % 60) == 0
            is_5m = (ts_arr % 300) == 0
            is_15m = (ts_arr % 900) == 0
            is_1h = (ts_arr % 3600) == 0

        for i in range(self._n):
            ts = int(ts_arr[i])
            v2_vec = self._v2_matrix[i]
            v2_dict = {name: float(v2_vec[j]) for j, name in enumerate(FEATURE_NAMES)}

            # 5s OHLCV — searchsorted nearest <= ts
            idx5 = np.searchsorted(self._ts5s, ts, side='right') - 1
            if 0 <= idx5 < len(self._ts5s):
                ohlcv_5s = {
                    'timestamp': float(self._ts5s[idx5]),
                    'open': float(self._o5[idx5]),
                    'high': float(self._h5[idx5]),
                    'low': float(self._l5[idx5]),
                    'close': float(self._c5[idx5]),
                    'volume': float(self._v5[idx5]),
                }
            else:
                ohlcv_5s = {'timestamp': float(ts), 'open': 0., 'high': 0.,
                                'low': 0., 'close': 0., 'volume': 0.}

            # 1m OHLCV — same pattern
            idx1 = np.searchsorted(self._ts1m, ts, side='right') - 1
            if 0 <= idx1 < len(self._ts1m):
                ohlcv_1m = {
                    'timestamp': float(self._ts1m[idx1]),
                    'open': float(self._o1[idx1]),
                    'high': float(self._h1[idx1]),
                    'low': float(self._l1[idx1]),
                    'close': float(self._c1[idx1]),
                    'volume': float(self._v1[idx1]),
                }
                price = float(self._c1[idx1])
            else:
                ohlcv_1m = None
                price = 0.0

            # Higher-TF most-recent CLOSED bar (LOOKAHEAD-FREE: bar at
            # ts_anchor - period, not current ts. A bar at time B closes
            # at B + period; the latest closed at `ts` is the largest B
            # with B + period <= ts, i.e., B <= ts - period.)
            def _latest_closed(ts_arr, period_s, o, h, l, c, v):
                if len(ts_arr) == 0: return None
                idx = int(np.searchsorted(ts_arr, ts - period_s, side='right')) - 1
                if idx < 0 or idx >= len(ts_arr):
                    return None
                return {
                    'timestamp': float(ts_arr[idx]),
                    'open': float(o[idx]), 'high': float(h[idx]),
                    'low': float(l[idx]), 'close': float(c[idx]),
                    'volume': float(v[idx]),
                }
            ohlcv_5m = _latest_closed(self._ts5m, 300, self._o5m, self._h5m,
                                                self._l5m, self._c5m, self._v5m)
            ohlcv_15m = _latest_closed(self._ts15m, 900, self._o15m, self._h15m,
                                                  self._l15m, self._c15m, self._v15m)
            ohlcv_1h = _latest_closed(self._ts1h, 3600, self._o1h, self._h1h,
                                                self._l1h, self._c1h, self._v1h)

            # Per-bar regime — forward-pass-honest under 'velocity' or 'hourly'
            bar_regime_2d, bar_regime_idx = self._regime_at(ts, v2_vec=v2_vec)

            yield BarState(
                timestamp=float(ts),
                bar_idx=i,
                day=self.day,
                price=price,
                ohlcv_5s=ohlcv_5s,
                ohlcv_1m=ohlcv_1m,
                ohlcv_5m=ohlcv_5m,
                ohlcv_15m=ohlcv_15m,
                ohlcv_1h=ohlcv_1h,
                v2=v2_dict,
                v2_vector=v2_vec,
                regime_2d=bar_regime_2d,
                regime_idx=bar_regime_idx,
                is_1m_close=bool(is_1m[i]),
                is_5m_close=bool(is_5m[i]),
                is_15m_close=bool(is_15m[i]),
                is_1h_close=bool(is_1h[i]),
            )


class MultiDayV2Ticker:
    """Replays multiple days through V2Ticker."""

    def __init__(self, days: Optional[List[str]] = None,
                 start_date: Optional[str] = None,
                 end_date: Optional[str] = None,
                 atlas_root: str = ATLAS_ROOT,
                 features_root: str = DEFAULT_FEATURES_ROOT,
                 labels_csv: str = LABELS_CSV):
        if days is None:
            l0_dir = os.path.join(features_root, 'L0')
            files = sorted(glob.glob(os.path.join(l0_dir, '*.parquet')))
            day_list = [os.path.basename(f).replace('.parquet', '') for f in files]
            if start_date:
                day_list = [d for d in day_list
                                if d.replace('_', '-') >= start_date]
            if end_date:
                day_list = [d for d in day_list
                                if d.replace('_', '-') <= end_date]
            days = day_list
        self._days = list(days)
        self._atlas_root = atlas_root
        self._features_root = features_root
        self._labels_csv = labels_csv
        self._current_day: Optional[str] = None

    @property
    def current_day(self) -> Optional[str]:
        return self._current_day

    def day_count(self) -> int:
        return len(self._days)

    def __iter__(self) -> Iterator[BarState]:
        for day in self._days:
            self._current_day = day
            try:
                ticker = V2Ticker(day=day, atlas_root=self._atlas_root,
                                       features_root=self._features_root,
                                       labels_csv=self._labels_csv)
            except FileNotFoundError:
                continue
            for state in ticker:
                yield state
