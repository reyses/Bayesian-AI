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

from core_v2.features import (FEATURE_NAMES, N_FEATURES, load_features)
from .state import BarState, regime_to_idx



# 5s bar period — this is the anchor cadence of the V2 features
ANCHOR_PERIOD_S = 5


_REGIME_CACHE: Optional[dict] = None


def _load_regime_lookup(labels_csv: str) -> dict:
    global _REGIME_CACHE
    if _REGIME_CACHE is None:
        try:
            df = pd.read_csv(labels_csv)
            df['date'] = df['date'].astype(str).str[:10]
            _REGIME_CACHE = dict(zip(df['date'], df['regime_2d']))
        except Exception:
            _REGIME_CACHE = {}
    return _REGIME_CACHE


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


class ForwardPassSystem:
    """Single-day V2-native ticker.

    Usage:
        for state in ForwardPassSystem('2025_06_15'):
            ...  # state: BarState
    """

    def __init__(self, day: str, atlas_root: str,
                 features_root: str,
                 labels_csv: str):
        self.day = day
        # Load V2 features (185 cols + timestamp). Anchor cadence = 5s.
        feats = load_features(days=[day], root=features_root)
        if feats.empty:
            raise FileNotFoundError(f'No V2 features for {day} under {features_root}')
        feats = _normalize_ts(feats).sort_values('timestamp').reset_index(drop=True)
        self._feats = feats
        self._n = len(feats)

        # 5s + 1m OHLCV
        self._ohlcv_5s = _read_ohlcv(os.path.join(atlas_root, '5s', f'{day}.parquet'))
        self._ohlcv_1m = _read_ohlcv(os.path.join(atlas_root, '1m', f'{day}.parquet'))

        if self._ohlcv_5s is None or len(self._ohlcv_5s) == 0:
            raise FileNotFoundError(f'No 5s OHLCV for {day}')
        if self._ohlcv_1m is None or len(self._ohlcv_1m) == 0:
            raise FileNotFoundError(f'No 1m OHLCV for {day}')

        # Regime
        iso = day.replace('_', '-')
        regime = _load_regime_lookup(labels_csv).get(iso, 'UNKNOWN')
        self._regime_2d = str(regime)
        self._regime_idx = regime_to_idx(self._regime_2d)

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

    def __len__(self) -> int:
        return self._n

    def __iter__(self) -> Iterator[BarState]:
        ts_arr = self._feats['timestamp'].values.astype(np.int64)

        # Detect timestamp convention (start-of-bar vs end-of-bar)
        offset = ts_arr[0] % 5 if len(ts_arr) > 0 else 0
        mod_1m = 0 if offset == 0 else 59
        mod_5m = 0 if offset == 0 else 299
        mod_15m = 0 if offset == 0 else 899
        mod_1h = 0 if offset == 0 else 3599

        # Pre-compute boundary flags vectorized
        is_1m = (ts_arr % 60) == mod_1m
        is_5m = (ts_arr % 300) == mod_5m
        is_15m = (ts_arr % 900) == mod_15m
        is_1h = (ts_arr % 3600) == mod_1h

        # Detect OHLCV conventions to prevent lookahead
        # DataBento (ATLAS) uses start-of-bar (ts % 5 == 0). A bar at 10:00:00 closes at 10:00:05.
        # NT8 (ATLAS_NT8) uses end-of-bar (ts % 5 == 4). A bar at 10:00:04 closes at 10:00:05.
        is_5s_start = (self._ts5s[0] % 5 == 0) if len(self._ts5s) > 0 else False
        is_1m_start = (self._ts1m[0] % 60 == 0) if len(self._ts1m) > 0 else False

        for i in range(self._n):
            ts = int(ts_arr[i])
            v2_vec = self._v2_matrix[i]
            v2_dict = {name: float(v2_vec[j]) for j, name in enumerate(FEATURE_NAMES)}

            # 5s OHLCV — searchsorted nearest <= search_ts
            search_ts_5s = ts - 5 if is_5s_start else ts
            idx5 = np.searchsorted(self._ts5s, search_ts_5s, side='right') - 1
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
            search_ts_1m = ts - 60 if is_1m_start else ts
            idx1 = np.searchsorted(self._ts1m, search_ts_1m, side='right') - 1
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

            yield BarState(
                timestamp=float(ts),
                bar_idx=i,
                day=self.day,
                price=price,
                ohlcv_5s=ohlcv_5s,
                ohlcv_1m=ohlcv_1m,
                v2=v2_dict,
                v2_vector=v2_vec,
                regime_2d=self._regime_2d,
                regime_idx=self._regime_idx,
                is_1m_close=bool(is_1m[i]),
                is_5m_close=bool(is_5m[i]),
                is_15m_close=bool(is_15m[i]),
                is_1h_close=bool(is_1h[i]),
            )


class MultiDayForwardPassSystem:
    """Replays multiple days through ForwardPassSystem."""

    def __init__(self, atlas_root: str,
                 features_root: str,
                 labels_csv: str,
                 days: Optional[List[str]] = None,
                 start_date: Optional[str] = None,
                 end_date: Optional[str] = None):
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
                ticker = ForwardPassSystem(day=day, atlas_root=self._atlas_root,
                                       features_root=self._features_root,
                                       labels_csv=self._labels_csv)
            except FileNotFoundError:
                continue
            for state in ticker:
                yield state
