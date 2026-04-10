"""
Aggregator — builds all timeframes from 1s bars, fires events on bar close.

Takes 1s bars one at a time from the ticker. Accumulates and aggregates into:
  5s, 15s, 1m, 5m, 15m, 1h, 1D

Fires callbacks when a TF bar closes:
  on_bar_close(tf, bar_dict)

The AI subscribes to these events.

Usage:
    agg = Aggregator()
    agg.on_bar_close = my_callback  # called with (tf, bar) when any TF closes

    for bar_1s in ticker:
        agg.feed(bar_1s)
        # callbacks fire automatically when TF bars complete
"""
import numpy as np
from typing import Callable, Dict, Optional, List

# TFs to aggregate (in seconds)
TF_SECONDS = {
    '5s': 5, '15s': 15, '1m': 60, '5m': 300, '15m': 900, '1h': 3600, '1D': 86400,
}

TF_ORDER = ['5s', '15s', '1m', '5m', '15m', '1h', '1D']


class BarAccumulator:
    """Accumulates 1s bars into a single higher TF bar."""

    __slots__ = ['tf_seconds', 'current_start', 'open', 'high', 'low', 'close',
                 'volume', 'count']

    def __init__(self, tf_seconds: int):
        self.tf_seconds = tf_seconds
        self.reset(0)

    def reset(self, bar_start_ts: float):
        self.current_start = bar_start_ts
        self.open = 0.0
        self.high = -1e18
        self.low = 1e18
        self.close = 0.0
        self.volume = 0.0
        self.count = 0

    def add(self, bar: Dict) -> Optional[Dict]:
        """Add a 1s bar. Returns the completed TF bar if the boundary is crossed, else None."""
        ts = bar['timestamp']
        bar_boundary = (ts // self.tf_seconds) * self.tf_seconds

        completed = None

        # If we crossed into a new TF bar, close the previous one
        if self.count > 0 and bar_boundary != self.current_start:
            completed = {
                'timestamp': self.current_start,
                'open': self.open,
                'high': self.high,
                'low': self.low,
                'close': self.close,
                'volume': self.volume,
            }
            self.reset(bar_boundary)

        # First bar in this TF period
        if self.count == 0:
            self.current_start = bar_boundary
            self.open = bar['open']

        # Accumulate
        self.high = max(self.high, bar['high'])
        self.low = min(self.low, bar['low'])
        self.close = bar['close']
        self.volume += bar['volume']
        self.count += 1

        return completed

    def get_partial(self) -> Optional[Dict]:
        """Get the current incomplete bar (for partial TF reads)."""
        if self.count == 0:
            return None
        return {
            'timestamp': self.current_start,
            'open': self.open,
            'high': self.high,
            'low': self.low,
            'close': self.close,
            'volume': self.volume,
        }


class Aggregator:
    """Aggregates 1s bars into all TFs. Fires callbacks on bar close.

    Maintains history of closed bars per TF for lookback.
    """

    def __init__(self, history_limit: int = 2000):
        """
        Args:
            history_limit: max closed bars to keep per TF (rolling window).
                           2000 1m bars = ~33 hours. Enough for SFE + features.
        """
        self._accumulators = {tf: BarAccumulator(secs)
                              for tf, secs in TF_SECONDS.items()}
        self._history_limit = history_limit

        # Closed bar history per TF (list of bar dicts)
        self.history = {tf: [] for tf in TF_ORDER}

        # Also keep raw 1s history
        self.history['1s'] = []

        # Callback: called with (tf_label, bar_dict) when a bar closes
        self.on_bar_close: Optional[Callable] = None

        # Counters
        self.total_1s_bars = 0

    def feed(self, bar_1s: Dict):
        """Feed one 1s bar. Aggregates into all TFs. Fires callbacks."""
        self.total_1s_bars += 1

        # Store raw 1s
        self.history['1s'].append(bar_1s)
        if len(self.history['1s']) > self._history_limit:
            self.history['1s'] = self.history['1s'][-self._history_limit:]

        # Aggregate into each TF
        for tf in TF_ORDER:
            acc = self._accumulators[tf]
            completed = acc.add(bar_1s)

            if completed is not None:
                # Bar closed — store and fire callback
                self.history[tf].append(completed)
                if len(self.history[tf]) > self._history_limit:
                    self.history[tf] = self.history[tf][-self._history_limit:]

                if self.on_bar_close is not None:
                    self.on_bar_close(tf, completed)

    def get_closed_bars(self, tf: str) -> List[Dict]:
        """Get all closed bars for a TF."""
        return self.history.get(tf, [])

    def get_partial_bar(self, tf: str) -> Optional[Dict]:
        """Get the current incomplete bar for a TF."""
        if tf in self._accumulators:
            return self._accumulators[tf].get_partial()
        return None

    def get_closed_bars_df(self, tf: str):
        """Get closed bars as a DataFrame (for SFE input)."""
        import pandas as pd
        bars = self.history.get(tf, [])
        if not bars:
            return pd.DataFrame(columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        return pd.DataFrame(bars)

    def get_bar_count(self, tf: str) -> int:
        """How many closed bars for this TF."""
        return len(self.history.get(tf, []))

    def save_state_parquet(self, state_dir: str = 'live/state/bars'):
        """Save all TF bar history as parquet. Version-proof, inspectable."""
        import os, json
        import pandas as pd
        from datetime import datetime
        os.makedirs(state_dir, exist_ok=True)

        manifest = {
            'version': 2,
            'created_at': datetime.utcnow().isoformat(),
            'bar_counts': {},
            'checksums': {},
        }

        for tf in list(self.history.keys()):
            bars = self.history[tf]
            if not bars:
                continue
            df = pd.DataFrame(bars)
            path = os.path.join(state_dir, f'{tf}.parquet')
            df.to_parquet(path, index=False)

            manifest['bar_counts'][tf] = len(bars)
            manifest['checksums'][tf] = {
                'rows': len(bars),
                'last_ts': float(bars[-1]['timestamp']),
                'last_close': float(bars[-1]['close']),
            }

        manifest['last_bar_ts'] = max(
            (cs['last_ts'] for cs in manifest['checksums'].values()), default=0)

        manifest_path = os.path.join(os.path.dirname(state_dir), 'manifest.json')
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)

        return manifest

    def load_state_parquet(self, state_dir: str = 'live/state/bars',
                           max_stale_hours: float = 18.0) -> dict:
        """Load bar history from parquet. Validates staleness + integrity."""
        import os, json, time, logging
        import pandas as pd
        _logger = logging.getLogger(__name__)

        manifest_path = os.path.join(os.path.dirname(state_dir), 'manifest.json')
        if not os.path.exists(manifest_path):
            raise FileNotFoundError(f'No manifest at {manifest_path}')

        with open(manifest_path) as f:
            manifest = json.load(f)

        if manifest.get('version', 1) < 2:
            raise ValueError('Warmup state is v1 (pickle). Run maintenance to rebuild.')

        last_ts = manifest.get('last_bar_ts', 0)
        stale_hours = (time.time() - last_ts) / 3600
        if stale_hours > max_stale_hours:
            raise ValueError(
                f'Warmup state is {stale_hours:.1f}h old (max {max_stale_hours}h). '
                f'Run maintenance to refresh.')

        loaded = {}
        for tf, expected in manifest.get('checksums', {}).items():
            path = os.path.join(state_dir, f'{tf}.parquet')
            if not os.path.exists(path):
                _logger.warning(f'Missing warmup parquet: {tf}')
                continue

            df = pd.read_parquet(path)

            if len(df) != expected['rows']:
                raise ValueError(f'{tf}: expected {expected["rows"]} rows, got {len(df)}')
            if abs(float(df['timestamp'].iloc[-1]) - expected['last_ts']) > 1.0:
                raise ValueError(f'{tf}: last_ts mismatch')

            bars = df.to_dict('records')
            self.history[tf] = bars
            loaded[tf] = len(bars)

        # Rebuild accumulators from last bar per TF
        for tf, bars in self.history.items():
            if not bars or tf not in self._accumulators:
                continue
            acc = self._accumulators[tf]
            last = bars[-1]
            if hasattr(acc, 'current_start'):
                acc.current_start = (last['timestamp'] // acc.tf_seconds) * acc.tf_seconds

        _logger.info(f'Warmup loaded: {loaded} (stale={stale_hours:.1f}h)')
        return manifest
