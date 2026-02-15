"""
CUDA-Accelerated Pattern Detector
Modernized implementation of legacy pattern recognition logic using Numba CUDA kernels.
Designed for high-throughput batch processing in QuantumFieldEngine.
"""
import numpy as np
from core.pattern_utils import (
    PATTERN_NONE, PATTERN_COMPRESSION, PATTERN_WEDGE, PATTERN_BREAKDOWN,
    CANDLESTICK_NONE, CANDLESTICK_DOJI, CANDLESTICK_HAMMER,
    CANDLESTICK_ENGULFING_BULL, CANDLESTICK_ENGULFING_BEAR
)

# Geometric Constants (must be int)
K_GEO_NONE = 0
K_GEO_COMPRESSION = 1
K_GEO_WEDGE = 2
K_GEO_BREAKDOWN = 3

# Candlestick Constants (must be int)
K_CDL_NONE = 0
K_CDL_DOJI = 1
K_CDL_HAMMER = 2
K_CDL_ENGULFING_BULL = 3
K_CDL_ENGULFING_BEAR = 4

# Thresholds (must match core/pattern_utils.py)
COMPRESSION_RATIO = 0.7
DOJI_BODY_RATIO = 0.1
HAMMER_BODY_RATIO = 0.3
HAMMER_LOWER_SHADOW_RATIO = 2.0
HAMMER_UPPER_SHADOW_RATIO = 0.1

try:
    from numba import cuda
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False

if NUMBA_AVAILABLE:
    @cuda.jit
    def detect_patterns_kernel(opens, highs, lows, closes, out_geo, out_cdl):
        """
        Unified CUDA Kernel to detect both Geometric and Candlestick patterns in parallel.
        """
        start = cuda.grid(1)
        stride = cuda.gridsize(1)
        n = highs.shape[0]

        for idx in range(start, n, stride):
            # === GEOMETRIC PATTERNS ===
            out_geo[idx] = K_GEO_NONE

            if idx >= 9: # Need 10 bars (idx-9 to idx)
                # 1. COMPRESSION (Priority 1)
                rec_max = -1e30
                rec_min = 1e30
                for k in range(idx - 4, idx + 1):
                    h = highs[k]
                    l = lows[k]
                    if h > rec_max: rec_max = h
                    if l < rec_min: rec_min = l
                rec_range = rec_max - rec_min

                prev_max = -1e30
                prev_min = 1e30
                for k in range(idx - 9, idx - 4):
                    h = highs[k]
                    l = lows[k]
                    if h > prev_max: prev_max = h
                    if l < prev_min: prev_min = l
                prev_range = prev_max - prev_min

                if prev_range > 0 and rec_range < prev_range * COMPRESSION_RATIO:
                    out_geo[idx] = K_GEO_COMPRESSION

                # 2. WEDGE (Priority 2)
                if lows[idx] > lows[idx-4] and highs[idx] < highs[idx-4]:
                    out_geo[idx] = K_GEO_WEDGE

                # 3. BREAKDOWN (Priority 3)
                min_prev_4 = 1e30
                for k in range(idx - 4, idx):
                    if lows[k] < min_prev_4:
                        min_prev_4 = lows[k]

                if lows[idx] < min_prev_4:
                    out_geo[idx] = K_GEO_BREAKDOWN

            # === CANDLESTICK PATTERNS ===
            out_cdl[idx] = K_CDL_NONE

            if idx >= 1: # Need at least 2 bars (idx-1, idx)
                c = closes[idx]
                o = opens[idx]
                h = highs[idx]
                l = lows[idx]

                pc = closes[idx-1]
                po = opens[idx-1]

                body = abs(c - o)
                rng = h - l
                if rng == 0: rng = 1e-10

                upper_shadow = h - max(c, o)
                lower_shadow = min(c, o) - l

                # 1. DOJI (Priority 1)
                if body / rng < DOJI_BODY_RATIO:
                    out_cdl[idx] = K_CDL_DOJI
                else:
                    # 2. HAMMER (Priority 2, only if not Doji)
                    is_hammer = (lower_shadow > HAMMER_LOWER_SHADOW_RATIO * body and
                                 upper_shadow < HAMMER_UPPER_SHADOW_RATIO * rng and
                                 body < HAMMER_BODY_RATIO * rng)

                    if is_hammer:
                        out_cdl[idx] = K_CDL_HAMMER
                    else:
                        # 3. ENGULFING (Priority 3, only if not Doji/Hammer)
                        # Bullish
                        if pc < po and c > o and o <= pc and c >= po:
                            out_cdl[idx] = K_CDL_ENGULFING_BULL
                        # Bearish
                        elif pc > po and c < o and o >= pc and c <= po:
                            out_cdl[idx] = K_CDL_ENGULFING_BEAR

else:
    detect_patterns_kernel = None


def detect_patterns_cuda(opens: np.ndarray, highs: np.ndarray, lows: np.ndarray, closes: np.ndarray):
    """
    Execute unified CUDA kernel for pattern detection.
    Returns tuple: (geometric_patterns, candlestick_patterns)
    """
    if not NUMBA_AVAILABLE:
        raise RuntimeError("Numba not installed/available for CUDA patterns")

    if not cuda.is_available():
        raise RuntimeError("CUDA GPU not available")

    n = len(highs)
    # Ensure contiguous float32 arrays
    opens = np.ascontiguousarray(opens.astype(np.float32))
    highs = np.ascontiguousarray(highs.astype(np.float32))
    lows = np.ascontiguousarray(lows.astype(np.float32))
    closes = np.ascontiguousarray(closes.astype(np.float32))

    # Alloc GPU memory
    d_opens = cuda.to_device(opens)
    d_highs = cuda.to_device(highs)
    d_lows = cuda.to_device(lows)
    d_closes = cuda.to_device(closes)

    d_out_geo = cuda.device_array(n, dtype=np.int32)
    d_out_cdl = cuda.device_array(n, dtype=np.int32)

    # Config
    threads_per_block = 256
    blocks_per_grid = (n + (threads_per_block - 1)) // threads_per_block

    # Launch
    detect_patterns_kernel[blocks_per_grid, threads_per_block](
        d_opens, d_highs, d_lows, d_closes, d_out_geo, d_out_cdl
    )

    # Copy back
    out_geo = d_out_geo.copy_to_host()
    out_cdl = d_out_cdl.copy_to_host()

    # Create lookups
    geo_lookup = np.array([
        PATTERN_NONE, PATTERN_COMPRESSION, PATTERN_WEDGE, PATTERN_BREAKDOWN
    ], dtype=object)

    cdl_lookup = np.array([
        CANDLESTICK_NONE, CANDLESTICK_DOJI, CANDLESTICK_HAMMER,
        CANDLESTICK_ENGULFING_BULL, CANDLESTICK_ENGULFING_BEAR
    ], dtype=object)

    # Safety clamp
    out_geo = np.clip(out_geo, 0, 3)
    out_cdl = np.clip(out_cdl, 0, 4)

    return geo_lookup[out_geo], cdl_lookup[out_cdl]

# Legacy wrapper for backward compatibility if needed (though we will update caller)
def detect_geometric_patterns_cuda(highs: np.ndarray, lows: np.ndarray) -> np.ndarray:
    """Legacy wrapper: Runs unified kernel but passes dummy open/close and returns only geo."""
    n = len(highs)
    dummy = np.zeros(n, dtype=np.float32)
    geo, _ = detect_patterns_cuda(dummy, highs, lows, dummy)
    return geo
