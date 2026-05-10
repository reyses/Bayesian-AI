"""
Feature Ticker — replays pre-computed 79D features from disk.

Same interface as the live pipeline (aggregator → 79D) but reads from
DATA/FEATURES_79D/ parquets. The downstream modules (NMP, tree, NN)
don't know the difference — they receive the same 79D data either way.

This is the TEST mode ticker. It replaces:
  ticker → aggregator → SFE → 79D (slow, for dataset build / live)
with:
  parquet → 79D (fast, for backtesting / training)

The contract: every consumer receives 79D features + timestamp + price.
The source is irrelevant.

Usage:
    from training.feature_ticker import FeatureTicker

    ft = FeatureTicker('DATA/FEATURES_79D/2026_01_06.parquet',
                       price_file='DATA/ATLAS/1m/2026_01_06.parquet')

    for state in ft:
        # state = {'timestamp', 'price', 'features', 'bar_idx'}
        nmp.on_state(state)
"""
import numpy as np
import pandas as pd
from typing import Iterator, Dict, Optional

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.features import FEATURE_NAMES, N_FEATURES


class FeatureTicker:
    """Replays pre-computed 79D features. Same output contract as live pipeline."""

    def __init__(self, feature_file: str, price_file: str = None):
        """
        Args:
            feature_file: path to 79D parquet (from build_dataset.py)
            price_file:   path to 1m OHLCV parquet (for price/OHLCV context)
                          If None, price comes from the 79D file's close
                          (if available) or is 0.
        """
        self._feat_df = pd.read_parquet(feature_file)
        self._n = len(self._feat_df)

        # Price source: 1m bars aligned by timestamp
        if price_file is not None and os.path.exists(price_file):
            price_df = pd.read_parquet(price_file).sort_values('timestamp')
            self._price_ts = price_df['timestamp'].values
            self._prices = price_df['close'].values
            self._highs = price_df['high'].values
            self._lows = price_df['low'].values
            self._opens = price_df['open'].values
            self._volumes = price_df['volume'].values
        else:
            self._price_ts = None
            self._prices = None

    def __iter__(self) -> Iterator[Dict]:
        feat_cols = [c for c in self._feat_df.columns if c in FEATURE_NAMES]
        feat_matrix = self._feat_df[feat_cols].values.astype(np.float32)
        timestamps = self._feat_df['timestamp'].values

        # V2 migration: extension columns produced by
        # tools/build_v2_to_v1_compat_cache.py. NOT in FEATURE_NAMES (91D),
        # so CNNs still see 91D unchanged. Tier engine reads via
        # state['extension_signals'] for V2-native classification.
        # Signals: upper/lower wick, body, swing_noise, vwap_dist, vol_velocity
        # per V1 TF + 4h-only signals + regime_2d.
        EXT_KEYS = ('_upper_wick', '_lower_wick', '_body', '_swing_noise',
                       '_vwap_dist', '_vol_velocity')
        EXT_NUMERIC_COLS = [c for c in self._feat_df.columns
                                 if any(c.endswith(k) for k in EXT_KEYS)
                                 or c.startswith('4h_')]
        ext_matrix = (self._feat_df[EXT_NUMERIC_COLS].values.astype(np.float32)
                          if EXT_NUMERIC_COLS else None)
        # Regime is a string column, broadcast per day. Stored separately.
        regime_per_row = (self._feat_df['regime_2d'].values
                              if 'regime_2d' in self._feat_df.columns else None)

        for i in range(self._n):
            ts = timestamps[i]
            features = feat_matrix[i]

            # Get price from 1m bars (nearest timestamp <= current)
            price = 0.0
            bar_data = None
            if self._price_ts is not None:
                idx = np.searchsorted(self._price_ts, ts, side='right') - 1
                if 0 <= idx < len(self._prices):
                    price = float(self._prices[idx])
                    bar_data = {
                        'timestamp': float(self._price_ts[idx]),
                        'open': float(self._opens[idx]),
                        'high': float(self._highs[idx]),
                        'low': float(self._lows[idx]),
                        'close': float(self._prices[idx]),
                        'volume': float(self._volumes[idx]),
                    }

            ext_signals = {}
            if ext_matrix is not None:
                for j, col in enumerate(EXT_NUMERIC_COLS):
                    ext_signals[col] = float(ext_matrix[i, j])
            if regime_per_row is not None:
                ext_signals['regime_2d'] = str(regime_per_row[i])

            yield {
                'timestamp': float(ts),
                'price': price,
                'features': features,
                'bar_idx': i,
                'bar_data': bar_data,  # 1m OHLCV for context (exits, etc.)
                'extension_signals': ext_signals,  # V2-native signals
            }

    def __len__(self) -> int:
        return self._n


class MultiDayFeatureTicker:
    """Replays multiple days of pre-computed 79D features sequentially."""

    def __init__(self, feature_dir: str, price_dir: str = None,
                 start_date: str = None, end_date: str = None):
        """
        Args:
            feature_dir: path to DATA/FEATURES_79D/
            price_dir:   path to DATA/ATLAS/1m/ (for price context)
            start_date:  YYYY-MM-DD filter
            end_date:    YYYY-MM-DD filter
        """
        import glob
        feat_files = sorted(glob.glob(os.path.join(feature_dir, '*.parquet')))

        if start_date:
            feat_files = [f for f in feat_files
                         if os.path.basename(f).replace('.parquet', '').replace('_', '-') >= start_date]
        if end_date:
            feat_files = [f for f in feat_files
                         if os.path.basename(f).replace('.parquet', '').replace('_', '-') <= end_date]

        self._files = []
        for ff in feat_files:
            day_name = os.path.basename(ff).replace('.parquet', '')
            pf = os.path.join(price_dir, f'{day_name}.parquet') if price_dir else None
            if pf and not os.path.exists(pf):
                pf = None
            self._files.append((ff, pf))

        self._current_day = None

    def __iter__(self) -> Iterator[Dict]:
        for feat_file, price_file in self._files:
            self._current_day = os.path.basename(feat_file).replace('.parquet', '')
            ft = FeatureTicker(feat_file, price_file)
            for state in ft:
                state['day'] = self._current_day
                yield state

    @property
    def current_day(self) -> Optional[str]:
        return self._current_day

    def day_count(self) -> int:
        return len(self._files)


# ---------------------------------------------------------------------------
# V2NativeTicker — eliminates the on-disk compat cache.
# Builds the V1-shape (91D + extension cols + regime) AND exposes the FULL V2
# layered feature vector (185D = L0 + 8 TFs × 23 concepts) per bar.
#
# Same state contract as FeatureTicker (features=91D V1-shape, extension_signals)
# plus two new entries:
#   - v2_features  : np.ndarray (185,) of V2 layered values (L0 + L1/L2/L3 × 8 TFs)
#   - v2_columns   : list[str] of column names in v2_features order (same per day)
#   - regime_idx   : int 0..len(REGIME_VOCAB)-1 derived from regime_2d string
# ---------------------------------------------------------------------------

# V2 grid layout — 8 TFs × 23 layered features per TF
V2_TFS = ('5s', '15s', '1m', '5m', '15m', '1h', '4h', '1D')
V2_L1_FEATS = ('price_velocity_1b', 'price_accel_1b', 'vol_velocity_1b',
                  'vol_accel_1b', 'bar_range', 'body')
V2_L2_FEATS = ('price_velocity_w', 'price_accel_w', 'vol_velocity_w', 'vol_accel_w',
                  'price_mean_w', 'price_sigma_w', 'vol_mean_w', 'vol_sigma_w', 'vwap_w')
V2_L3_FEATS = ('z_se_w', 'z_high_w', 'z_low_w', 'SE_high_w', 'SE_low_w',
                  'hurst_w', 'reversion_prob_w', 'swing_noise_w')
V2_PER_TF_FEATS = V2_L1_FEATS + V2_L2_FEATS + V2_L3_FEATS  # 23
N_V2_TFS = len(V2_TFS)                                      # 8
N_V2_PER_TF = len(V2_PER_TF_FEATS)                          # 23

REGIME_VOCAB = ('UNKNOWN', 'UP_SMOOTH', 'UP_CHOPPY',
                    'DOWN_SMOOTH', 'DOWN_CHOPPY',
                    'FLAT_SMOOTH', 'FLAT_CHOPPY')


def build_v2_grid_columns():
    """Return (185 col-name list) in canonical V2-grid order.

    Order: [L0_time_of_day] + for tf in V2_TFS for f in (L1+L2+L3): col_name
    The grid for the CNN is (N_V2_TFS, N_V2_PER_TF) — drops L0 scalar.
    """
    cols = ['L0_time_of_day']
    for tf in V2_TFS:
        for feat in V2_L1_FEATS:
            cols.append(f'L1_{tf}_{feat}')
        for feat in V2_L2_FEATS:
            cols.append(f'L2_{tf}_{feat}')
        for feat in V2_L3_FEATS:
            cols.append(f'L3_{tf}_{feat}')
    return cols


V2_COLUMNS = build_v2_grid_columns()
N_V2_TOTAL = len(V2_COLUMNS)  # 185


class V2NativeTicker:
    """V2-native ticker — no on-disk compat cache.

    Builds the V1-shape (91D + extension cols + regime) AND the full V2
    layered features (185D) in memory per day, then yields per-bar state.

    state contract:
      timestamp, price, features (91D V1-shape, np.float32), bar_idx,
      bar_data (dict — 1m OHLCV at the bar)
      extension_signals (dict — per-TF directional wicks, body, swing_noise,
                          vwap_dist, vol_velocity + 4h-only signals + regime_2d str)
      v2_features (np.float32 (185,)) — L0 + L1/L2/L3 × 8 TFs
      v2_columns  (list[str]) — column names in v2_features order
      regime_idx  (int) — index into REGIME_VOCAB
    """

    def __init__(self, day: str, atlas_root: str = 'DATA/ATLAS',
                 labels_csv: str = 'DATA/ATLAS/regime_labels_2d.csv',
                 price_dir: Optional[str] = None):
        from core_v2.v2_to_v1_inmemory import build_v1_shape_for_day

        self.day = day
        result = build_v1_shape_for_day(atlas_root, day, labels_csv,
                                              return_v2=True)
        if result is None:
            raise FileNotFoundError(f'No data for day {day}')
        self._compat_df, self._v2_df = result
        self._n = len(self._compat_df)

        # Pre-fill v2 grid matrix once. Missing cols → 0.
        v2_grid = np.zeros((self._n, N_V2_TOTAL), dtype=np.float32)
        for j, col in enumerate(V2_COLUMNS):
            if col in self._v2_df.columns:
                v2_grid[:, j] = self._v2_df[col].values.astype(np.float32)
        self._v2_grid = v2_grid

        # Regime int code (constant across the day)
        regime_2d = (self._compat_df['regime_2d'].iloc[0]
                          if 'regime_2d' in self._compat_df.columns
                          else 'UNKNOWN')
        try:
            self._regime_idx = REGIME_VOCAB.index(str(regime_2d))
        except ValueError:
            self._regime_idx = 0  # UNKNOWN
        self._regime_2d = str(regime_2d)

        # 1m OHLCV for price/bar_data context
        if price_dir is None:
            price_dir = os.path.join(atlas_root, '1m')
        price_file = os.path.join(price_dir, f'{day}.parquet')
        if os.path.exists(price_file):
            price_df = pd.read_parquet(price_file).sort_values('timestamp')
            if pd.api.types.is_datetime64_any_dtype(price_df['timestamp']):
                price_df = price_df.copy()
                price_df['timestamp'] = (price_df['timestamp']
                                              .astype('int64') // 10**9)
            self._price_ts = price_df['timestamp'].values.astype(np.int64)
            self._prices = price_df['close'].values
            self._highs = price_df['high'].values
            self._lows = price_df['low'].values
            self._opens = price_df['open'].values
            self._volumes = price_df['volume'].values
        else:
            self._price_ts = None
            self._prices = None

    def __iter__(self) -> Iterator[Dict]:
        # 91D V1-shape feature columns (drop timestamp + extension + regime)
        EXT_KEYS = ('_upper_wick', '_lower_wick', '_body', '_swing_noise',
                       '_vwap_dist', '_vol_velocity')
        compat_cols = list(self._compat_df.columns)
        feat_cols = [c for c in compat_cols
                          if c in FEATURE_NAMES]
        ext_cols = [c for c in compat_cols
                         if any(c.endswith(k) for k in EXT_KEYS)
                         or c.startswith('4h_')]

        feat_matrix = self._compat_df[feat_cols].values.astype(np.float32)
        ext_matrix = (self._compat_df[ext_cols].values.astype(np.float32)
                          if ext_cols else None)
        timestamps = self._compat_df['timestamp'].values

        for i in range(self._n):
            ts = timestamps[i]
            features = feat_matrix[i]

            price = 0.0
            bar_data = None
            if self._price_ts is not None:
                idx = np.searchsorted(self._price_ts, ts, side='right') - 1
                if 0 <= idx < len(self._prices):
                    price = float(self._prices[idx])
                    bar_data = {
                        'timestamp': float(self._price_ts[idx]),
                        'open': float(self._opens[idx]),
                        'high': float(self._highs[idx]),
                        'low': float(self._lows[idx]),
                        'close': float(self._prices[idx]),
                        'volume': float(self._volumes[idx]),
                    }

            ext_signals = {}
            if ext_matrix is not None:
                for j, col in enumerate(ext_cols):
                    ext_signals[col] = float(ext_matrix[i, j])
            ext_signals['regime_2d'] = self._regime_2d

            yield {
                'timestamp': float(ts),
                'price': price,
                'features': features,
                'bar_idx': i,
                'bar_data': bar_data,
                'extension_signals': ext_signals,
                'v2_features': self._v2_grid[i],
                'v2_columns': V2_COLUMNS,
                'regime_idx': self._regime_idx,
            }

    def __len__(self) -> int:
        return self._n


class MultiDayV2NativeTicker:
    """Replays multiple days through V2NativeTicker (no compat cache)."""

    def __init__(self, atlas_root: str = 'DATA/ATLAS',
                 labels_csv: str = 'DATA/ATLAS/regime_labels_2d.csv',
                 start_date: str = None, end_date: str = None,
                 day_strs: Optional[list] = None):
        """
        Args:
            atlas_root:  path to DATA/ATLAS (must contain 5s/, 1m/, FEATURES_5s_v2/)
            labels_csv:  path to regime_labels_2d.csv
            start_date:  YYYY-MM-DD filter (inclusive)
            end_date:    YYYY-MM-DD filter (inclusive)
            day_strs:    explicit list of YYYY_MM_DD day strings (overrides start/end)
        """
        import glob
        if day_strs is not None:
            self._days = list(day_strs)
        else:
            five_s_dir = os.path.join(atlas_root, '5s')
            files = sorted(glob.glob(os.path.join(five_s_dir, '*.parquet')))
            day_list = [os.path.basename(f).replace('.parquet', '') for f in files]
            if start_date:
                day_list = [d for d in day_list
                                if d.replace('_', '-') >= start_date]
            if end_date:
                day_list = [d for d in day_list
                                if d.replace('_', '-') <= end_date]
            self._days = day_list

        self._atlas_root = atlas_root
        self._labels_csv = labels_csv
        self._current_day = None

    def __iter__(self) -> Iterator[Dict]:
        for day in self._days:
            self._current_day = day
            try:
                ft = V2NativeTicker(day=day, atlas_root=self._atlas_root,
                                              labels_csv=self._labels_csv)
            except FileNotFoundError:
                continue
            for state in ft:
                state['day'] = day
                yield state

    @property
    def current_day(self) -> Optional[str]:
        return self._current_day

    def day_count(self) -> int:
        return len(self._days)
