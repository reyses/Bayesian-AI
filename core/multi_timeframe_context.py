"""
Multi-Timeframe Context Engine
Computes cascading context from 8 timeframe layers (1D → 1S)

Architecture:
  Layer 1: Daily (21-day lookback)   → macro trend + volatility regime
  Layer 2: 4-Hour (21-bar lookback)  → session context
  Layer 3: 1-Hour (21-bar lookback)  → intraday wave structure
  Layer 4: 15-Min (21-bar lookback)  → PRIMARY DECISION LAYER (always in engine)
  Layer 5: 5-Min (20-bar lookback)   → pattern setup
  Layer 6: 1-Min (20-bar lookback)   → confirmation
  Layer 7: 15-Sec (20-bar lookback)  → tactical entry (existing engine)
  Layer 8: 1-Sec                     → execution (existing sim)

Context availability depends on day index:
  Day 1:    15m + lower only
  Day 2-3:  + 1h context
  Day 4-21: + 4h context
  Day 22+:  + daily context (FULL)
"""
import numpy as np
import pandas as pd
from scipy.stats import linregress
from typing import Dict, Optional, Tuple
from dataclasses import dataclass


@dataclass
class TimeframeContext:
    """Context computed from a single higher timeframe"""
    trend: Optional[str] = None          # 'UP', 'DOWN', 'RANGE' (or 'BULL','BEAR' for daily)
    volatility: Optional[str] = None     # 'HIGH', 'NORMAL', 'LOW'
    slope: float = 0.0                   # Raw regression slope
    z_score: float = 0.0                 # Current z-score on this TF
    session: Optional[str] = None        # Trading session (4h only)


class MultiTimeframeContext:
    """
    Computes and caches multi-timeframe context for the training pipeline.

    Usage:
        mtf = MultiTimeframeContext()
        all_tf = mtf.resample_all(full_data_1s)              # Once per dataset
        ctx = mtf.get_context_for_day(day_idx, all_tf, date)  # Per day
    """

    TIMEFRAMES = {
        '1d':  {'rule': '1D',   'lookback': 21, 'min_days': 22},
        '4h':  {'rule': '4h',   'lookback': 21, 'min_days': 4},
        '1h':  {'rule': '1h',   'lookback': 21, 'min_days': 2},
        '15m': {'rule': '15min','lookback': 21, 'min_days': 1},
        '5m':  {'rule': '5min', 'lookback': 20, 'min_days': 1},
        '1m':  {'rule': '1min', 'lookback': 20, 'min_days': 1},
    }

    def resample_all(self, data_1s: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """
        Resample full 1s dataset to all higher timeframes. Run once per dataset.

        Args:
            data_1s: Full 1s OHLCV data with 'timestamp' column

        Returns:
            Dict of {timeframe_key: DataFrame} for each TF
        """
        df = data_1s.copy()

        # Ensure datetime index
        if 'timestamp' in df.columns:
            if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
                df['timestamp'] = pd.to_datetime(pd.to_numeric(df['timestamp']), unit='s')
            df = df.set_index('timestamp')

        # Build aggregation dict from available columns
        agg_dict = {}
        if 'open' in df.columns:
            agg_dict['open'] = 'first'
        if 'high' in df.columns:
            agg_dict['high'] = 'max'
        if 'low' in df.columns:
            agg_dict['low'] = 'min'
        if 'close' in df.columns:
            agg_dict['close'] = 'last'
        if 'price' in df.columns:
            agg_dict['price'] = 'last'
        if 'volume' in df.columns:
            agg_dict['volume'] = 'sum'

        # Derive close from price if needed
        if 'close' not in df.columns and 'price' in df.columns:
            df['close'] = df['price']
            agg_dict['close'] = 'last'

        dropna_col = 'close' if 'close' in agg_dict else 'price'

        result = {}
        for tf_key, tf_config in self.TIMEFRAMES.items():
            resampled = df.resample(tf_config['rule']).agg(agg_dict).dropna(subset=[dropna_col])
            # Ensure price column
            if 'price' not in resampled.columns and 'close' in resampled.columns:
                resampled['price'] = resampled['close']
            result[tf_key] = resampled

        return result

    def get_context_for_day(self, day_idx: int, all_tf: Dict[str, pd.DataFrame],
                            current_date: str) -> Dict[str, Optional[TimeframeContext]]:
        """
        Compute available context for a specific training day.

        Args:
            day_idx: 0-based day index in training sequence
            all_tf: Dict from resample_all()
            current_date: Current day date string (YYYY-MM-DD)

        Returns:
            Dict with keys 'daily', 'h4', 'h1' — each TimeframeContext or None
        """
        current_date_pd = pd.Timestamp(current_date)

        context = {
            'daily': None,
            'h4': None,
            'h1': None,
            'context_level': 'MINIMAL',
        }

        # Daily context: available Day 22+
        if day_idx >= 21:
            daily_bars = self._get_bars_before_date(all_tf.get('1d'), current_date_pd, 21)
            if daily_bars is not None and len(daily_bars) >= 21:
                context['daily'] = self._compute_trend_context(daily_bars, label='daily')
                context['context_level'] = 'FULL'

        # 4-hour context: available Day 4+
        if day_idx >= 3:
            h4_bars = self._get_bars_before_date(all_tf.get('4h'), current_date_pd, 21)
            if h4_bars is not None and len(h4_bars) >= 21:
                ctx = self._compute_trend_context(h4_bars, label='h4')
                ctx.session = self._detect_session(current_date_pd)
                context['h4'] = ctx
                if context['context_level'] == 'MINIMAL':
                    context['context_level'] = 'PARTIAL'

        # 1-hour context: available Day 2+
        if day_idx >= 1:
            h1_bars = self._get_bars_before_date(all_tf.get('1h'), current_date_pd, 21)
            if h1_bars is not None and len(h1_bars) >= 21:
                context['h1'] = self._compute_trend_context(h1_bars, label='h1')
                if context['context_level'] == 'MINIMAL':
                    context['context_level'] = 'PARTIAL'

        return context

    def _get_bars_before_date(self, df: Optional[pd.DataFrame],
                               current_date: pd.Timestamp, lookback: int) -> Optional[pd.DataFrame]:
        """Get the last N bars strictly before the current date."""
        if df is None or df.empty:
            return None

        mask = df.index < current_date
        before = df[mask]

        if len(before) < lookback:
            return None

        return before.iloc[-lookback:]

    def _compute_trend_context(self, bars: pd.DataFrame, label: str = '') -> TimeframeContext:
        """
        Compute trend direction and volatility from a lookback window.

        Uses linear regression slope for trend, residual std for volatility.
        """
        close = bars['close'].values if 'close' in bars.columns else bars['price'].values
        x = np.arange(len(close), dtype=np.float64)

        slope, intercept, _, _, _ = linregress(x, close)
        center = slope * x[-1] + intercept
        residuals = close - (slope * x + intercept)
        sigma = np.sqrt(np.sum(residuals ** 2) / max(len(close) - 2, 1))
        sigma = max(sigma, 1e-10)

        z_score = (close[-1] - center) / sigma

        # Trend classification based on slope significance
        # Normalize slope by sigma to get a dimensionless trend strength
        slope_strength = abs(slope * len(close)) / sigma if sigma > 0 else 0

        if slope_strength > 1.0:
            if slope > 0:
                trend = 'BULL' if label == 'daily' else 'UP'
            else:
                trend = 'BEAR' if label == 'daily' else 'DOWN'
        else:
            trend = 'RANGE'

        # Volatility regime: compare recent vs historical
        recent_vol = np.std(close[-7:]) if len(close) >= 7 else sigma
        hist_vol = np.std(close)
        vol_ratio = recent_vol / (hist_vol + 1e-10)

        if vol_ratio > 1.3:
            volatility = 'HIGH'
        elif vol_ratio < 0.7:
            volatility = 'LOW'
        else:
            volatility = 'NORMAL'

        return TimeframeContext(
            trend=trend,
            volatility=volatility,
            slope=slope,
            z_score=z_score,
        )

    def _detect_session(self, timestamp: pd.Timestamp) -> str:
        """Detect trading session from timestamp hour (US Eastern approximation)."""
        hour = timestamp.hour

        if 0 <= hour < 8:
            return 'ASIA'
        elif 8 <= hour < 13:
            return 'EUROPE'
        elif 13 <= hour < 16:
            return 'OVERLAP'
        else:
            return 'US'

    def get_confidence_modifier(self, context_level: str) -> float:
        """
        Return confidence modifier based on available context depth.
        Used for cold-start handling.
        """
        modifiers = {
            'MINIMAL': 0.6,   # Day 1: core only
            'PARTIAL': 0.8,   # Day 2-21: some higher TF
            'FULL': 1.0,      # Day 22+: all TF available
        }
        return modifiers.get(context_level, 0.6)
