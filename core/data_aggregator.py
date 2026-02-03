"""
Bayesian-AI - Data Aggregator
Manages real-time tick buffer and bar generation
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional

class DataAggregator:
    def __init__(self, max_ticks: int = 10000):
        self.max_ticks = max_ticks
        self._df_cache: Optional[pd.DataFrame] = None
        self._bars_cache: Dict[str, pd.DataFrame] = {}
        # Track total ticks added to manage incremental updates
        self._total_ticks_added = 0
        self._last_processed_ticks = 0

        # Ring buffer storage
        self._buffers: Dict[str, np.ndarray] = {}
        self._idx = 0  # Points to the next insertion slot
        self._size = 0 # Current number of valid items
        self._initialized = False

    @property
    def ticks(self) -> List[Dict]:
        """
        Reconstructs the list of ticks from buffers for backward compatibility.
        Warning: This is expensive. Avoid using in performance-critical paths.
        """
        if not self._initialized or self._size == 0:
            return []

        # Get data in chronological order
        data = self._get_ordered_data()

        # Convert to DataFrame then to records
        # This preserves types and column names
        return pd.DataFrame(data).to_dict('records')

    def add_tick(self, tick: Dict):
        """
        Add a tick to the buffer
        tick: {'timestamp': float, 'price': float, 'volume': float, ...}
        """
        # Ensure timestamp is present
        if 'timestamp' not in tick:
            tick['timestamp'] = pd.Timestamp.now().timestamp()

        self.ticks.append(tick)
        self._total_ticks_added += 1

        if len(self.ticks) > self.max_ticks:
            self.ticks.pop(0)

    def get_current_data(self) -> Dict:
        """
        Get snapshot of data for LayerEngine
        Returns dict with 'ticks', 'bars_5m', 'bars_15m', etc.
        """
        if not self._initialized or self._size == 0:
            return {
                'price': 0.0,
                'timestamp': 0.0,
                'ticks': np.array([]),
                'bars_5m': None,
                'bars_15m': None,
                'bars_1h': None,
                'bars_4hr': None
            }

        # Initialize or Update _df_cache
        if self._df_cache is None:
            # Initial build
            self._df_cache = pd.DataFrame(self.ticks)
            self._prepare_df_index(self._df_cache)
            self._last_processed_ticks = self._total_ticks_added
        else:
            # Incremental update
            new_count = self._total_ticks_added - self._last_processed_ticks

            if new_count > 0:
                if new_count >= len(self.ticks):
                    # Should be rare/impossible unless max_ticks is huge or logic error,
                    # but if we need to add more than we have, just rebuild.
                    self._df_cache = pd.DataFrame(self.ticks)
                    self._prepare_df_index(self._df_cache)
                else:
                    # Append new ticks
                    new_ticks_data = self.ticks[-new_count:]
                    new_df = pd.DataFrame(new_ticks_data)
                    self._prepare_df_index(new_df)
                    self._df_cache = pd.concat([self._df_cache, new_df])

                # Maintain cache size
                if len(self._df_cache) > self.max_ticks:
                    self._df_cache = self._df_cache.iloc[-self.max_ticks:]

                self._last_processed_ticks = self._total_ticks_added

        df = self._df_cache

        # Get current price/ts from buffers directly for speed
        last_idx = (self._idx - 1) % self.max_ticks

        # Helper to get value safely
        def get_last_val(key):
            if key in self._buffers:
                val = self._buffers[key][last_idx]
                if pd.isna(val): return 0.0 # Handle NaN
                return val
            return 0.0

        current_price = get_last_val('price')
        current_ts = get_last_val('timestamp')

        # Helper to safely resample with caching
        def get_bars(rule):
            try:
                if df.empty: return None

                # Use cache if available
                cached = self._bars_cache.get(rule)

                start_subset_idx = None

                if cached is not None and not cached.empty:
                    # Identify the last bar in the cache
                    last_bar_idx = cached.index[-1]

                    # Check if last_bar_idx is still within df range
                    if last_bar_idx <= df.index[-1]:
                         start_subset_idx = last_bar_idx

                if start_subset_idx is not None:
                    # Incremental update
                    # Get relevant data slice
                    subset = df[df.index >= start_subset_idx]

                    if subset.empty:
                        return cached

                    resampled_subset = subset.resample(rule).agg({
                        'open': 'first',
                        'high': 'max',
                        'low': 'min',
                        'close': 'last',
                        'volume': 'sum'
                    }).dropna()

                    if resampled_subset.empty:
                        return cached

                    # Merge: Cache (minus last bar) + Resampled Subset
                    base_cache = cached.loc[:start_subset_idx]
                    if not base_cache.empty and base_cache.index[-1] == start_subset_idx:
                         base_cache = base_cache.iloc[:-1]

                    updated_cache = pd.concat([base_cache, resampled_subset])
                else:
                    # Full resample (initial or cache invalid)
                    updated_cache = df.resample(rule).agg({
                        'open': 'first',
                        'high': 'max',
                        'low': 'min',
                        'close': 'last',
                        'volume': 'sum'
                    }).dropna()

                # Trim cache to match the window of ticks we hold
                if len(updated_cache) > self.max_ticks:
                    updated_cache = updated_cache.iloc[-self.max_ticks:]

                self._bars_cache[rule] = updated_cache
                return updated_cache

            except Exception:
                # In case of missing columns or other errors
                return None

        # Prepare ticks DataFrame for return
        if not df.empty:
            ticks_df = df.reset_index()
            # Ensure columns exist before selecting
            cols = [c for c in ['price', 'timestamp'] if c in ticks_df.columns]
            ticks_df = ticks_df[cols]
        else:
            ticks_df = pd.DataFrame(columns=['price', 'timestamp'])

        return {
            'price': current_price,
            'timestamp': current_ts,
            'ticks': ticks_df,
            'bars_5m': get_bars('5min'),
            'bars_15m': get_bars('15min'),
            'bars_1h': get_bars('1h'),
            'bars_4hr': get_bars('4h')
        }

    def _prepare_df_index(self, df: pd.DataFrame):
        """Helper to ensure datetime index exists"""
        if 'datetime' not in df.columns:
            if 'timestamp' in df.columns:
                if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
                     df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
                else:
                     df['datetime'] = df['timestamp']
            else:
                # Should not happen given add_tick logic
                pass

        if 'datetime' in df.columns:
            df.set_index('datetime', inplace=True)
