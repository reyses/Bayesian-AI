import numpy as np
import pandas as pd

# Pattern Constants
PATTERN_NONE = 'NONE'
PATTERN_COMPRESSION = 'COMPRESSION'
PATTERN_WEDGE = 'WEDGE'
PATTERN_BREAKDOWN = 'BREAKDOWN'

CANDLESTICK_NONE = 'NONE'
CANDLESTICK_DOJI = 'DOJI'
CANDLESTICK_HAMMER = 'HAMMER'
CANDLESTICK_ENGULFING_BULL = 'ENGULFING_BULL'
CANDLESTICK_ENGULFING_BEAR = 'ENGULFING_BEAR'

# Thresholds
COMPRESSION_RATIO = 0.7
DOJI_BODY_RATIO = 0.1
HAMMER_BODY_RATIO = 0.3
HAMMER_LOWER_SHADOW_RATIO = 2.0
HAMMER_UPPER_SHADOW_RATIO = 0.1

def detect_geometric_pattern(highs: np.ndarray, lows: np.ndarray) -> str:
    """
    Detects the latest geometric pattern (compression, wedge, breakdown) from price data.
    Returns 'NONE' if no pattern is detected or insufficient data.
    Checks conditions for the LAST bar.
    """
    n = len(highs)
    if n < 10:  # Need at least 10 bars for 5-bar lookback for prev_range
        return PATTERN_NONE

    # Recent range (last 5 bars)
    rec_range = np.max(highs[-5:]) - np.min(lows[-5:])
    # Prev range (5 bars before the last 5)
    prev_range = np.max(highs[-10:-5]) - np.min(lows[-10:-5])

    # Compression (Priority 1)
    if prev_range > 0 and rec_range < prev_range * COMPRESSION_RATIO:
        return PATTERN_COMPRESSION

    # Wedge: Higher Lows AND Lower Highs over 5 bars (last bar vs 4 bars ago)
    # Compare current vs 4 bars ago
    if lows[-1] > lows[-5] and highs[-1] < highs[-5]:
        return PATTERN_WEDGE

    # Breakdown: Current low < min of previous 4 lows
    if lows[-1] < np.min(lows[-5:-1]):
        return PATTERN_BREAKDOWN

    return PATTERN_NONE

def detect_candlestick_pattern(opens: np.ndarray, highs: np.ndarray,
                               lows: np.ndarray, closes: np.ndarray) -> str:
    """
    Detects the latest candlestick pattern (Engulfing, Hammer, Doji) from price data.
    Returns 'NONE' if no pattern is detected or insufficient data.
    Checks conditions for the LAST bar.
    """
    n = len(closes)
    if n < 2:  # Need at least 2 bars for candlestick patterns
        return CANDLESTICK_NONE

    c = closes[-1]
    o = opens[-1]
    h = highs[-1]
    l = lows[-1]

    pc = closes[-2]
    po = opens[-2]

    body = abs(c - o)
    rng = h - l
    if rng == 0: rng = 1e-10  # Avoid division by zero

    # Doji
    if body / rng < DOJI_BODY_RATIO:
        return CANDLESTICK_DOJI

    # Hammer
    upper_shadow = h - max(c, o)
    lower_shadow = min(c, o) - l

    if (lower_shadow > HAMMER_LOWER_SHADOW_RATIO * body and
        upper_shadow < HAMMER_UPPER_SHADOW_RATIO * rng and
        body < HAMMER_BODY_RATIO * rng):
        return CANDLESTICK_HAMMER

    # Engulfing Bull
    if pc < po and c > o and o <= pc and c >= po:
        return CANDLESTICK_ENGULFING_BULL

    # Engulfing Bear
    if pc > po and c < o and o >= pc and c <= po:
        return CANDLESTICK_ENGULFING_BEAR

    return CANDLESTICK_NONE

def detect_geometric_patterns_vectorized(highs: np.ndarray, lows: np.ndarray) -> np.ndarray:
    """
    Vectorized geometric pattern detection (Compression, Wedge, Breakdown)
    Returns array of pattern strings.
    Uses pandas rolling windows for efficiency and correct boundary handling.
    """
    n = len(highs)
    patterns = np.full(n, PATTERN_NONE, dtype=object)

    if n < 10:
        return patterns

    # Convert to pandas Series for efficient rolling operations
    highs_s = pd.Series(highs)
    lows_s = pd.Series(lows)

    # Recent 5 bars range (window=5)
    rec_range = highs_s.rolling(5).max() - lows_s.rolling(5).min()

    # Previous 5 bars range (shifted by 5)
    prev_range = rec_range.shift(5)

    # Compression (Priority 1)
    # Compare ranges. Pandas handles NaNs (result is False), so no mask needed.
    compression_mask = (prev_range > 0) & (rec_range < prev_range * COMPRESSION_RATIO)
    # Fill patterns where mask is True. Use .to_numpy() to align with array.
    # fillna(False) ensures we don't have boolean ambiguity with NaNs
    patterns[compression_mask.fillna(False).to_numpy()] = PATTERN_COMPRESSION

    # Wedge (Higher Lows AND Lower Highs) over 5 bars (Priority 2)
    # Compare current vs 4 bars ago
    wedge_mask = (lows > lows_s.shift(4)) & (highs < highs_s.shift(4))
    patterns[wedge_mask.fillna(False).to_numpy()] = PATTERN_WEDGE

    # Breakdown (Low < min of previous 4 lows) (Priority 3)
    # Previous 4 lows: shift 1, then rolling min 4
    prev_4_min = lows_s.shift(1).rolling(4).min()
    breakdown_mask = lows < prev_4_min
    patterns[breakdown_mask.fillna(False).to_numpy()] = PATTERN_BREAKDOWN

    # Clear first 9 bars explicitly (warmup period for rolling windows)
    patterns[:9] = PATTERN_NONE

    return patterns

def detect_candlestick_patterns_vectorized(opens: np.ndarray, highs: np.ndarray,
                                           lows: np.ndarray, closes: np.ndarray) -> np.ndarray:
    """
    Vectorized candlestick pattern detection.
    Detects: Doji (High Priority), Hammer, Engulfing (Low Priority)
    Priority: Doji > Hammer > Engulfing
    """
    n = len(closes)
    patterns = np.full(n, CANDLESTICK_NONE, dtype=object)

    if n < 2:
        return patterns

    # Helper arrays
    body = np.abs(closes - opens)
    upper_shadow = highs - np.maximum(closes, opens)
    lower_shadow = np.minimum(closes, opens) - lows
    total_range = highs - lows

    # Avoid division by zero
    total_range = np.where(total_range == 0, 1e-10, total_range)

    # 1. DOJI (Priority 1: Overwrites any lower priority)
    doji_mask = (body / total_range) < DOJI_BODY_RATIO
    patterns[doji_mask] = CANDLESTICK_DOJI

    # 2. HAMMER (Priority 2: Apply only where patterns is NONE)
    hammer_mask = (
        (lower_shadow > (HAMMER_LOWER_SHADOW_RATIO * body)) &
        (upper_shadow < (HAMMER_UPPER_SHADOW_RATIO * total_range)) &
        (body < (HAMMER_BODY_RATIO * total_range))
    )
    # Only update where no pattern detected (Doji takes precedence)
    patterns[hammer_mask & (patterns == CANDLESTICK_NONE)] = CANDLESTICK_HAMMER

    # 3. ENGULFING (Priority 3: Apply only where patterns is NONE)
    # Use pd.Series.shift(1) to avoid wrap-around (np.roll wraps end to start)
    # Convert back to numpy for mask operations
    prev_opens = pd.Series(opens).shift(1).to_numpy()
    prev_closes = pd.Series(closes).shift(1).to_numpy()

    # Fill NaNs from shift with a safe value that won't trigger pattern
    # (e.g. same as current open/close to avoid crossover)
    prev_opens = np.nan_to_num(prev_opens, nan=opens)
    prev_closes = np.nan_to_num(prev_closes, nan=closes)

    # Bullish Engulfing:
    # Prev Red, Curr Green, Curr Open < Prev Close, Curr Close > Prev Open
    bull_eng_mask = (
        (prev_closes < prev_opens) &  # Prev Red
        (closes > opens) &            # Curr Green
        (opens <= prev_closes) &      # Open below prev close
        (closes >= prev_opens)        # Close above prev open
    )

    # Bearish Engulfing:
    # Prev Green, Curr Red, Curr Open > Prev Close, Curr Close < Prev Open
    bear_eng_mask = (
        (prev_closes > prev_opens) &  # Prev Green
        (closes < opens) &            # Curr Red
        (opens >= prev_closes) &      # Open above prev close
        (closes <= prev_opens)        # Close below prev open
    )

    # Only update where no pattern detected (Doji/Hammer take precedence)
    patterns[bull_eng_mask & (patterns == CANDLESTICK_NONE)] = CANDLESTICK_ENGULFING_BULL
    patterns[bear_eng_mask & (patterns == CANDLESTICK_NONE)] = CANDLESTICK_ENGULFING_BEAR

    # Clear first bar (due to shift/rolling) explicitly just in case
    patterns[0] = CANDLESTICK_NONE

    return patterns
