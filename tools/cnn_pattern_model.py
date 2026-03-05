"""
CNN Pattern Detection Model for Waveform Analysis R.
Conv1D on 5-channel OHLCV windows + liquidation level context scalars.

Dual-path architecture:
  - Conv path: learns local shape patterns from normalized OHLCV (5ch × 64 bars)
  - Context path: 2 scalar features — distance to nearest liquidation level above/below
    (from peak-touch sliding window detection on daily TF)
"""
import numpy as np

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# 7 pattern classes (mapped from 20 seed primitives)
PATTERN_CLASSES = [
    'RAMP_UP', 'RAMP_DOWN',
    'V_REVERSAL_UP', 'V_REVERSAL_DOWN',
    'STEP_UP', 'STEP_DOWN',
    'OSCILLATION',
]

SEED_TO_CLASS = {
    'LINEAR_UP': 0, 'EXPONENTIAL_UP': 0, 'LOGARITHMIC_UP': 0,
    'LINEAR_DOWN': 1, 'EXPONENTIAL_DOWN': 1, 'LOGARITHMIC_DOWN': 1,
    'SYMMETRIC_V_UP': 2, 'ROUNDED_U_UP': 2, 'FRONT_SKEWED_UP': 2, 'BACK_SKEWED_UP': 2,
    'SYMMETRIC_V_DOWN': 3, 'ROUNDED_U_DOWN': 3, 'FRONT_SKEWED_DOWN': 3, 'BACK_SKEWED_DOWN': 3,
    'STEP_UP': 4,
    'STEP_DOWN': 5,
    'SINE_WAVE': 6, 'DAMPED_OSCILLATOR': 6, 'EXPAND_OSCILLATOR': 6, 'FLATLINE': 6,
}

N_CLASSES = len(PATTERN_CLASSES)
WINDOW_LEN = 64


if TORCH_AVAILABLE:
    class PatternCNN(nn.Module):
        """Dual-path Conv1D pattern classifier.

        Conv path:  (batch, 5, 64) OHLCV → shape features (64-dim)
        Context:    (batch, n_context) scalars — liquidation level distances
        Combined:   (64 + n_context) → 32 → 7 classes
        """
        def __init__(self, in_channels=5, n_context=3, n_classes=N_CLASSES):
            super().__init__()
            # Conv path — learns local shape patterns
            self.conv1 = nn.Conv1d(in_channels, 32, kernel_size=5, padding=2)
            self.bn1 = nn.BatchNorm1d(32)
            self.conv2 = nn.Conv1d(32, 64, kernel_size=5, padding=2)
            self.bn2 = nn.BatchNorm1d(64)
            self.conv3 = nn.Conv1d(64, 64, kernel_size=3, padding=1)
            self.bn3 = nn.BatchNorm1d(64)
            self.pool = nn.AdaptiveAvgPool1d(1)
            # FC path — combines shape features + regime context
            self.fc1 = nn.Linear(64 + n_context, 32)
            self.fc2 = nn.Linear(32, n_classes)
            self.relu = nn.ReLU()
            self.dropout = nn.Dropout(0.3)

        def forward(self, x, ctx=None):
            # Conv path: shape features from OHLCV
            x = self.relu(self.bn1(self.conv1(x)))
            x = self.relu(self.bn2(self.conv2(x)))
            x = self.relu(self.bn3(self.conv3(x)))
            x = self.pool(x).squeeze(-1)           # (batch, 64)
            # Concat regime context if provided
            if ctx is not None:
                x = torch.cat([x, ctx], dim=1)      # (batch, 67)
            x = self.dropout(self.relu(self.fc1(x)))
            return self.fc2(x)


def extract_ohlcv_windows(df, indices, window_len=WINDOW_LEN):
    """Extract normalized OHLCV windows (conv input only).

    Returns:
        windows: np.array (n_valid, 5, window_len)
        valid_mask: boolean array (len(indices),)
    """
    o = df['open'].values.astype(float)
    h = df['high'].values.astype(float)
    lo = df['low'].values.astype(float)
    c = df['close'].values.astype(float)
    v = df['volume'].values.astype(float)

    windows = []
    valid_mask = np.zeros(len(indices), dtype=bool)

    for i, idx in enumerate(indices):
        start = idx - window_len + 1
        if start < 0 or idx + 1 > len(c):
            continue

        w_o, w_h, w_l, w_c = o[start:idx+1], h[start:idx+1], lo[start:idx+1], c[start:idx+1]
        w_v = v[start:idx+1]
        if len(w_c) != window_len:
            continue

        # Price channels: per-window z-score
        price_stack = np.stack([w_o, w_h, w_l, w_c])
        p_mean, p_std = price_stack.mean(), price_stack.std()
        if p_std < 1e-10:
            continue
        price_norm = (price_stack - p_mean) / p_std

        # Volume: log1p then z-score
        v_log = np.log1p(w_v)
        v_mean, v_std = v_log.mean(), v_log.std()
        v_norm = (v_log - v_mean) / max(v_std, 1e-10)

        window = np.vstack([price_norm, v_norm[np.newaxis, :]])  # (5, W)
        windows.append(window)
        valid_mask[i] = True

    if not windows:
        return np.empty((0, 5, window_len)), valid_mask
    return np.array(windows, dtype=np.float32), valid_mask


def compute_regime_context(df, indices, regime_ids, regime_meta):
    """Compute per-bar regime-sectioned context scalars.

    Each regime section has its own local high/low/duration — the context tells
    the CNN where this bar sits within its local section (soft boundaries).

    Returns:
        context: np.array (len(indices), 3)
            [0] range_pct:   (close - regime_low) / (regime_high - regime_low)
                             0=at section floor, 1=at section ceiling
            [1] time_pct:    bars into regime / regime duration
                             0=start of section, 1=end
            [2] vol_ratio:   regime volatility / median volatility across all regimes
                             >1 = high energy section, <1 = calm
    """
    c = df['close'].values.astype(float)
    h = df['high'].values.astype(float)
    lo = df['low'].values.astype(float)

    # Precompute per-regime high, low, duration
    regime_bounds = {}
    median_vol = np.median([rm['volatility'] for rm in regime_meta]) if regime_meta else 1.0
    if median_vol < 1e-10:
        median_vol = 1.0

    for rm in regime_meta:
        rid = rm['regime_id']
        s, e = rm['start_idx'], rm['end_idx']
        r_high = float(h[s:e+1].max())
        r_low = float(lo[s:e+1].min())
        r_range = r_high - r_low
        if r_range < 1e-10:
            r_range = 1.0
        regime_bounds[rid] = {
            'high': r_high, 'low': r_low, 'range': r_range,
            'start': s, 'duration': rm['n_bars'],
            'vol_ratio': rm['volatility'] / median_vol,
        }

    context = np.zeros((len(indices), 3), dtype=np.float32)
    for i, idx in enumerate(indices):
        rid = regime_ids[idx] if idx < len(regime_ids) else -1
        if rid < 0 or rid not in regime_bounds:
            # Warmup bar — use neutral context
            context[i] = [0.5, 0.5, 1.0]
            continue

        rb = regime_bounds[rid]
        # Where in the section's price range
        context[i, 0] = (c[idx] - rb['low']) / rb['range']
        # How far through the section's duration
        context[i, 1] = min(1.0, (idx - rb['start']) / max(rb['duration'], 1))
        # Section energy level
        context[i, 2] = rb['vol_ratio']

    return context


def detect_peak_touch_levels(highs, lows, lookback_bars=60, swing_order=3,
                             tolerance=75.0, merge_dist=150.0, top_k=7):
    """Detect liquidation levels via peak-touch scanning on daily OHLCV.

    Slides horizontal lines across the price axis and counts how many swing
    peaks/troughs each line touches (within tolerance). More touches = stronger
    level. Uses a trailing window of `lookback_bars` daily bars.

    Args:
        highs: daily high prices (1D array)
        lows: daily low prices (1D array)
        lookback_bars: trailing window size in daily bars (default 60 ≈ 3 months)
        swing_order: argrelextrema order for swing detection
        tolerance: price distance (pts) for a peak to "touch" a horizontal line
        merge_dist: merge levels closer than this (pts)
        top_k: return top K levels by touch count

    Returns:
        levels: list of (price, touch_count) tuples, sorted by price
    """
    from scipy.signal import argrelextrema

    n = len(highs)
    if n < lookback_bars:
        lookback_bars = n

    # Use last `lookback_bars` of data
    h = np.asarray(highs[-lookback_bars:], dtype=float)
    lo = np.asarray(lows[-lookback_bars:], dtype=float)

    # Find swing highs and lows
    peak_idx = argrelextrema(h, np.greater, order=swing_order)[0]
    trough_idx = argrelextrema(lo, np.less, order=swing_order)[0]

    # Collect all swing prices
    swing_prices = np.concatenate([h[peak_idx], lo[trough_idx]]) if \
        (len(peak_idx) > 0 or len(trough_idx) > 0) else np.array([])

    if len(swing_prices) < 2:
        return []

    # Scan price axis — count touches per horizontal line
    p_min, p_max = swing_prices.min() - tolerance, swing_prices.max() + tolerance
    scan_step = tolerance / 3.0  # fine resolution
    scan_levels = np.arange(p_min, p_max, scan_step)

    best = []  # (price, touch_count)
    for level in scan_levels:
        touches = np.sum(np.abs(swing_prices - level) <= tolerance)
        if touches >= 2:
            best.append((float(level), int(touches)))

    if not best:
        return []

    # Sort by touch count descending, then merge nearby levels
    best.sort(key=lambda x: -x[1])
    merged = []
    used = set()
    for price, count in best:
        if any(abs(price - mp) < merge_dist for mp, _ in merged):
            continue
        merged.append((price, count))
        if len(merged) >= top_k:
            break

    # Sort by price
    merged.sort(key=lambda x: x[0])
    return merged


def compute_level_context(close_prices, levels, indices, atr=None):
    """Compute per-bar distance to nearest liquidation level above/below.

    Args:
        close_prices: full close price array (all bars, intraday TF)
        levels: list of (price, touch_count) or just list of prices
        indices: bar indices to compute context for
        atr: optional ATR array for normalization (same length as close_prices).
             If None, uses median inter-level spacing for normalization.

    Returns:
        context: np.array (len(indices), 2)
            [0] dist_above: normalized distance to nearest level above (0-1 scale)
            [1] dist_below: normalized distance to nearest level below (0-1 scale)
    """
    # Extract price values from (price, count) tuples if needed
    if levels and isinstance(levels[0], (list, tuple)):
        level_prices = np.array([l[0] for l in levels])
    else:
        level_prices = np.array(levels, dtype=float)

    if len(level_prices) == 0:
        return np.full((len(indices), 2), 0.5, dtype=np.float32)

    level_prices = np.sort(level_prices)

    # Normalization scale: median inter-level spacing or ATR
    if len(level_prices) > 1:
        spacings = np.diff(level_prices)
        norm_scale = float(np.median(spacings))
    else:
        norm_scale = 1.0

    c = np.asarray(close_prices, dtype=float)
    context = np.zeros((len(indices), 2), dtype=np.float32)

    for i, idx in enumerate(indices):
        price = c[idx]
        local_norm = norm_scale
        if atr is not None and idx < len(atr) and atr[idx] > 0:
            local_norm = atr[idx]

        above = level_prices[level_prices >= price]
        below = level_prices[level_prices <= price]

        # Distance to nearest level above (capped at 1.0)
        if len(above) > 0:
            context[i, 0] = min(1.0, (above[0] - price) / max(local_norm, 1.0))
        else:
            context[i, 0] = 1.0  # above all levels

        # Distance to nearest level below (capped at 1.0)
        if len(below) > 0:
            context[i, 1] = min(1.0, (price - below[-1]) / max(local_norm, 1.0))
        else:
            context[i, 1] = 1.0  # below all levels

    return context
