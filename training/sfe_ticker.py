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
from core_v2.features import FEATURE_NAMES, N_FEATURES


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
        day = os.path.basename(feature_file).replace('.parquet', '')
        if 'FEATURES_79D' in feature_file or 'FEATURES_NT8' in feature_file or 'FEATURES_5s' in feature_file:
            from core_v2.features import load_features
            root = 'DATA/ATLAS_NT8/FEATURES_5s_v2' if 'NT8' in feature_file else 'DATA/ATLAS/FEATURES_5s_v2'
            try:
                self._feat_df = load_features([day], root=root).reset_index()
            except Exception as e:
                # fallback if it's actually a single file
                if os.path.exists(feature_file):
                    self._feat_df = pd.read_parquet(feature_file)
                else:
                    raise e
        else:
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

            yield {
                'timestamp': float(ts),
                'price': price,
                'features': features,
                'bar_idx': i,
                'bar_data': bar_data,  # 1m OHLCV for context (exits, etc.)
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
