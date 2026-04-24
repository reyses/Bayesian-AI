"""
1-second ticker for realistic backtesting — v2 (185D features).

Yields per-1s state dicts compatible with engine `on_state(state)`:

  state = {
    'timestamp': int,          # 1s bar timestamp (epoch seconds)
    'price':     float,        # 1s close
    'high':      float,        # 1s high (for TP/SL precision)
    'low':       float,        # 1s low  (for TP/SL precision)
    'features':  np.ndarray,   # 185D v2 features at nearest-past 5s row
                                # Column order = core_v2.features.FEATURE_NAMES
    'bar_data':  dict | None,  # MOST-RECENTLY-COMPLETED 1m bar (no lookahead)
    'bar_idx':   int,
  }

Timing convention (critical — no lookahead):
  - Price, high, low are for the CURRENT 1s bar.
  - Features use `searchsorted(feat_ts, ts, right) - 1` → feature row with
    timestamp <= current 1s ts. Nearest-past, no lookahead.
    (v2 features at timestamp T were computed using bars that closed by T,
     so looking up T via nearest-past preserves the zero-lookahead guarantee.)
  - bar_data refers to the most recently COMPLETED 1m bar.

Source files (v2):
  - 1s OHLCV:   DATA/ATLAS/1s/<day>.parquet
  - v2 features: DATA/ATLAS/FEATURES_5s_v2/{L0,L1_*,L2_*,L3_*}/<day>.parquet
  - 1m OHLCV:   DATA/ATLAS/1m/<day>.parquet
"""
import os
import sys
from typing import Iterator, Dict, Optional

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core_v2.features import FEATURE_NAMES, load_features, DEFAULT_FEATURES_ROOT


class OneSecondTicker:
    """Yields per-1s state with v2 5s features (nearest-past) and most-recently-
    completed 1m bar_data. No lookahead."""

    def __init__(self, day: str,
                 ps_dir: str = 'DATA/ATLAS/1s',
                 feat_dir: str = DEFAULT_FEATURES_ROOT,
                 p1m_dir: str = 'DATA/ATLAS/1m'):
        self.day = day
        p1s_path = os.path.join(ps_dir, f'{day}.parquet')
        p1m_path = os.path.join(p1m_dir, f'{day}.parquet')

        df1s = pd.read_parquet(p1s_path).sort_values('timestamp').reset_index(drop=True)
        self._ts_1s = df1s['timestamp'].values.astype(np.int64)
        self._open_1s = df1s['open'].values.astype(np.float64)
        self._high_1s = df1s['high'].values.astype(np.float64)
        self._low_1s = df1s['low'].values.astype(np.float64)
        self._close_1s = df1s['close'].values.astype(np.float64)
        self._vol_1s = df1s['volume'].values.astype(np.float64) if 'volume' in df1s.columns else np.zeros(len(df1s))
        self._n = len(df1s)

        # v2 features: join per-layer-family parquets for this single day.
        # require_all=False so days with partial family coverage don't crash.
        df_feat = load_features(days=[day], root=feat_dir, require_all=False)
        if df_feat.empty:
            raise FileNotFoundError(
                f"No v2 feature parquets found for {day} under {feat_dir}. "
                f"Run `python training/build_dataset_v2.py --atlas DATA/ATLAS` first."
            )
        df_feat = df_feat.sort_values('timestamp').reset_index(drop=True)
        self._feat_ts = df_feat['timestamp'].values.astype(np.int64)
        # Canonical column order from FEATURE_NAMES (drop timestamp).
        # Engine indexes positions via core_v2.features.get_feature_index(name).
        missing = [c for c in FEATURE_NAMES if c not in df_feat.columns]
        if missing:
            raise RuntimeError(
                f"v2 feature parquet for {day} is missing {len(missing)} cols: "
                f"{missing[:5]}... Run `build_dataset_v2.py` to regenerate."
            )
        self._feat_matrix = df_feat[list(FEATURE_NAMES)].values.astype(np.float32)

        df1m = pd.read_parquet(p1m_path).sort_values('timestamp').reset_index(drop=True)
        self._ts_1m = df1m['timestamp'].values.astype(np.int64)
        self._open_1m = df1m['open'].values.astype(np.float64)
        self._high_1m = df1m['high'].values.astype(np.float64)
        self._low_1m = df1m['low'].values.astype(np.float64)
        self._close_1m = df1m['close'].values.astype(np.float64)
        self._vol_1m = df1m['volume'].values.astype(np.float64) if 'volume' in df1m.columns else np.zeros(len(df1m))

    def __iter__(self) -> Iterator[Dict]:
        feat_ts = self._feat_ts
        feat_matrix = self._feat_matrix
        ts_1m = self._ts_1m

        for i in range(self._n):
            ts = int(self._ts_1s[i])

            # Features: nearest-past 5s row
            f_idx = int(np.searchsorted(feat_ts, ts, side='right')) - 1
            if f_idx < 0:
                continue  # warm-up before first feature row
            features = feat_matrix[f_idx]

            # bar_data: most-recently-COMPLETED 1m bar.
            # 1m bar at ts_1m[k] covers [ts_1m[k], ts_1m[k] + 60).
            # Completed at 1s tick ts means ts_1m[k] + 60 <= ts, i.e.
            # ts_1m[k] <= ts - 60.
            b_idx = int(np.searchsorted(ts_1m, ts - 60, side='right')) - 1
            if b_idx < 0:
                # No 1m bar has completed yet — emit state without bar_data
                bar_data = None
            else:
                bar_data = {
                    'timestamp': int(ts_1m[b_idx]),
                    'open': float(self._open_1m[b_idx]),
                    'high': float(self._high_1m[b_idx]),
                    'low': float(self._low_1m[b_idx]),
                    'close': float(self._close_1m[b_idx]),
                    'volume': float(self._vol_1m[b_idx]),
                }

            yield {
                'timestamp': ts,
                'price': float(self._close_1s[i]),
                'high': float(self._high_1s[i]),
                'low': float(self._low_1s[i]),
                'features': features,
                'bar_data': bar_data,
                'bar_idx': i,
            }

    def __len__(self) -> int:
        return self._n
