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
        self.ticks: List[Dict] = []
        self._df_cache: Optional[pd.DataFrame] = None
        self._last_tick_count = 0

    def add_tick(self, tick: Dict):
        """
        Add a tick to the buffer
        tick: {'timestamp': float, 'price': float, 'volume': float, ...}
        """
        # Ensure timestamp is present
        if 'timestamp' not in tick:
            tick['timestamp'] = pd.Timestamp.now().timestamp()

        self.ticks.append(tick)
        if len(self.ticks) > self.max_ticks:
            self.ticks.pop(0)

        # Invalidate cache
        self._df_cache = None

    def get_current_data(self) -> Dict:
        """
        Get snapshot of data for LayerEngine
        Returns dict with 'ticks', 'bars_5m', 'bars_15m', etc.
        """
        if not self.ticks:
            return {
                'price': 0.0,
                'timestamp': 0.0,
                'ticks': np.array([]),
                'bars_5m': None,
                'bars_15m': None,
                'bars_1h': None,
                'bars_4hr': None
            }

        # Convert to DataFrame
        if self._df_cache is None:
            self._df_cache = pd.DataFrame(self.ticks)
            # Ensure timestamp column is datetime for resampling
            if not pd.api.types.is_datetime64_any_dtype(self._df_cache['timestamp']):
                # Assuming timestamp is float (epoch)
                self._df_cache['datetime'] = pd.to_datetime(self._df_cache['timestamp'], unit='s')
            else:
                self._df_cache['datetime'] = self._df_cache['timestamp']

            self._df_cache.set_index('datetime', inplace=True)

        df = self._df_cache
        current_price = self.ticks[-1]['price']
        current_ts = self.ticks[-1]['timestamp']

        # Generate bars
        # Note: In a real high-freq system, we wouldn't resample full history every tick.
        # We would update the last bar. This is a simplified implementation.

        # Helper to safely resample
        def get_bars(rule):
            try:
                if df.empty: return None
                resampled = df.resample(rule).agg({
                    'open': 'first',
                    'high': 'max',
                    'low': 'min',
                    'close': 'last',
                    'volume': 'sum'
                }).dropna()
                return resampled
            except Exception:
                return None

        return {
            'price': current_price,
            'timestamp': current_ts,
            'ticks': df.reset_index()[['price', 'timestamp']], # Pass DataFrame with price/timestamp columns
            'bars_5m': get_bars('5min'),
            'bars_15m': get_bars('15min'),
            'bars_1h': get_bars('1h'),
            'bars_4hr': get_bars('4h')
        }
