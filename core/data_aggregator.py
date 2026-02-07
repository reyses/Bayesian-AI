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
        self._last_tick_count = 0

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

        if not self._initialized:
            self._init_buffers(tick)

        # Update buffers
        # Iterate over ALL known buffers to ensure we overwrite stale data at current index
        for key, arr in self._buffers.items():
            val = tick.get(key)
            if val is not None:
                arr[self._idx] = val
            else:
                # Key missing in this tick: overwrite with NaN/None
                if np.issubdtype(arr.dtype, np.number):
                    arr[self._idx] = np.nan
                else:
                    arr[self._idx] = None

        self._idx = (self._idx + 1) % self.max_ticks
        if self._size < self.max_ticks:
            self._size += 1

        # Invalidate cache
        self._df_cache = None

    def _init_buffers(self, sample_tick: Dict):
        """Initialize numpy arrays based on first tick structure"""
        for key, value in sample_tick.items():
            if isinstance(value, (int, float, np.number)):
                dtype = np.float64
                fill_value = np.nan
            else:
                dtype = object
                fill_value = None

            self._buffers[key] = np.full(self.max_ticks, fill_value, dtype=dtype)

        self._initialized = True

    def _get_ordered_data(self) -> Dict[str, np.ndarray]:
        """Return dict of arrays ordered chronologically"""
        if not self._initialized or self._size == 0:
            return {}

        if self._size < self.max_ticks:
            # Not full, just slice 0 to size
            return {k: v[:self._size] for k, v in self._buffers.items()}
        else:
            # Full, wrap around.
            # Current _idx points to the OLDEST data (next to be overwritten)
            # So data is [idx:] + [:idx]
            idx = self._idx
            return {k: np.concatenate((v[idx:], v[:idx])) for k, v in self._buffers.items()}

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
                'bars_15s': None,
                'bars_5m': None,
                'bars_15m': None,
                'bars_1h': None,
                'bars_4hr': None
            }

        # Convert to DataFrame
        if self._df_cache is None:
            data = self._get_ordered_data()
            self._df_cache = pd.DataFrame(data)

            # Ensure timestamp column is datetime for resampling
            if 'timestamp' in self._df_cache.columns:
                if not pd.api.types.is_datetime64_any_dtype(self._df_cache['timestamp']):
                    # Assuming timestamp is float (epoch)
                    self._df_cache['datetime'] = pd.to_datetime(self._df_cache['timestamp'], unit='s')
                else:
                    self._df_cache['datetime'] = self._df_cache['timestamp']

                self._df_cache.set_index('datetime', inplace=True)
            else:
                pass

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

        # Generate bars
        # Note: In a real high-freq system, we wouldn't resample full history every tick.
        # We would update the last bar. This is a simplified implementation.

        # Helper to safely resample
        def get_bars(rule):
            try:
                if df.empty: return None

                # Check if we have OHLC columns, if not, try to use 'price'
                has_ohlc = all(c in df.columns for c in ['open', 'high', 'low', 'close'])

                if has_ohlc:
                    resampled = df.resample(rule).agg({
                        'open': 'first',
                        'high': 'max',
                        'low': 'min',
                        'close': 'last',
                        'volume': 'sum'
                    }).dropna()
                elif 'price' in df.columns:
                    # Aggregate price to OHLC
                    resampled = df['price'].resample(rule).ohlc()
                    if 'volume' in df.columns:
                         resampled['volume'] = df['volume'].resample(rule).sum()
                    resampled = resampled.dropna()
                else:
                    return None

                return resampled
            except Exception:
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
            'bars_15s': get_bars('15s'),
            'bars_5m': get_bars('5min'),
            'bars_15m': get_bars('15min'),
            'bars_1h': get_bars('1h'),
            'bars_4hr': get_bars('4h')
        }
