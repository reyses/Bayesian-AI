"""
LevelTracker — rolling z-extreme support/resistance from physics.

Tracks the max(z_high) and min(z_low) over rolling windows for two layers:
- 1D layer: last N days (persistent structure, multi-month memory)
- 1h layer: last M hours (recent tactical, ~few days of intraday memory)

These represent the actual z-space ceiling/floor the market has tested
and respected. Used as entry filter (block if no headroom in your direction)
and as exit target (TP at the opposite extreme).

Pure function on bar history — no internal state. Call get_levels() at
each entry decision with the current bar store.
"""
import numpy as np
from typing import Dict, Optional

# Z_HIGH / Z_LOW indices in the core feature vector (per TF)
# Within a TF's 12 core features: z_high=10, z_low=11
# We track from the bar history's OHLC, computing z relative to current center/sigma

DEFAULT_1D_WINDOW = 60   # last 60 daily bars (~3 months of trading)
DEFAULT_1H_WINDOW = 120  # last 120 hourly bars (~5 days)


def compute_z_extremes(bars_df, window_size: int):
    """For a bars DataFrame, compute the rolling z-high and z-low extremes.

    Args:
        bars_df: DataFrame with timestamp, open, high, low, close columns
        window_size: number of bars to look back

    Returns:
        dict with:
          ceiling_z: max z_high over window (highest tested z)
          floor_z: min z_low over window (lowest tested z)
          ceiling_price: actual price at ceiling
          floor_price: actual price at floor
          center: current regression center
          sigma: current regression std
    """
    if bars_df is None or len(bars_df) < 21:
        return None

    # Use only the last `window_size` bars
    if len(bars_df) > window_size:
        recent = bars_df.iloc[-window_size:].reset_index(drop=True)
    else:
        recent = bars_df.reset_index(drop=True)

    # Compute regression on the window: linear fit to closes
    closes = recent['close'].values.astype(np.float64)
    highs = recent['high'].values.astype(np.float64)
    lows = recent['low'].values.astype(np.float64)
    n = len(closes)

    # Linear regression slope/intercept (ordinary least squares)
    x = np.arange(n)
    x_mean = x.mean()
    y_mean = closes.mean()
    cov = ((x - x_mean) * (closes - y_mean)).sum()
    var = ((x - x_mean) ** 2).sum()
    if var < 1e-12:
        return None
    slope = cov / var
    intercept = y_mean - slope * x_mean

    # Residual std from regression line
    line = intercept + slope * x
    residuals = closes - line
    sigma = residuals.std()
    if sigma < 1e-8:
        return None

    # Center is the line's value at the latest bar
    center = line[-1]

    # Z values for each bar's high/low
    z_highs = (highs - line) / sigma
    z_lows = (lows - line) / sigma

    ceiling_z = float(z_highs.max())
    floor_z = float(z_lows.min())
    ceiling_price = float(center + ceiling_z * sigma)
    floor_price = float(center + floor_z * sigma)

    return {
        'ceiling_z': ceiling_z,
        'floor_z': floor_z,
        'ceiling_price': ceiling_price,
        'floor_price': floor_price,
        'center': float(center),
        'sigma': float(sigma),
    }


def get_levels(bar_stores: Dict[str, 'pd.DataFrame'],
               window_1d: int = DEFAULT_1D_WINDOW,
               window_1h: int = DEFAULT_1H_WINDOW):
    """Compute both layers of S/R levels from bar history.

    Args:
        bar_stores: LiveFeatureEngine._bars dict {tf: DataFrame}
        window_1d: days of 1D history to track
        window_1h: hours of 1h history to track

    Returns:
        dict with '1D' and '1h' sub-dicts (each from compute_z_extremes)
    """
    out = {}
    if '1D' in bar_stores:
        out['1D'] = compute_z_extremes(bar_stores['1D'], window_1d)
    if '1h' in bar_stores:
        out['1h'] = compute_z_extremes(bar_stores['1h'], window_1h)
    return out


def headroom(levels: dict, current_price: float, direction: str):
    """Compute z headroom in the trade direction for both layers.

    Args:
        levels: dict from get_levels()
        current_price: current 1m price
        direction: 'long' or 'short'

    Returns:
        dict with min headroom across layers, plus per-layer detail.
        Returns None if levels not available.
    """
    if not levels:
        return None

    layer_headrooms = {}
    for layer_name, lvl in levels.items():
        if lvl is None:
            continue
        sigma = lvl['sigma']
        if sigma < 1e-8:
            continue
        center = lvl['center']
        current_z = (current_price - center) / sigma

        if direction == 'long':
            # Long: how much z space until ceiling
            head = lvl['ceiling_z'] - current_z
        else:
            # Short: how much z space until floor
            head = current_z - lvl['floor_z']

        layer_headrooms[layer_name] = {
            'headroom': head,
            'current_z': current_z,
            'level_z': lvl['ceiling_z'] if direction == 'long' else lvl['floor_z'],
            'level_price': lvl['ceiling_price'] if direction == 'long' else lvl['floor_price'],
        }

    if not layer_headrooms:
        return None

    # Take the min headroom across layers (most restrictive)
    min_layer = min(layer_headrooms.items(), key=lambda x: x[1]['headroom'])
    return {
        'min_headroom': min_layer[1]['headroom'],
        'min_layer': min_layer[0],
        'layers': layer_headrooms,
    }
