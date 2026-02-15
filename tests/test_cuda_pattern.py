import pytest
import numpy as np
import pandas as pd
from core.pattern_utils import (
    detect_geometric_patterns_vectorized, detect_candlestick_patterns_vectorized,
    PATTERN_COMPRESSION, PATTERN_WEDGE, PATTERN_BREAKDOWN,
    CANDLESTICK_DOJI, CANDLESTICK_HAMMER, CANDLESTICK_ENGULFING_BULL, CANDLESTICK_ENGULFING_BEAR
)

# Try importing CUDA module
try:
    from core.cuda_pattern_detector import detect_patterns_cuda, NUMBA_AVAILABLE
    from numba import cuda
    CUDA_READY = NUMBA_AVAILABLE and cuda.is_available()
except (ImportError, Exception):
    CUDA_READY = False

@pytest.mark.skipif(not CUDA_READY, reason="CUDA not available")
def test_cuda_unified_pattern_equivalence():
    """
    Verify that unified CUDA pattern detection produces identical results to CPU vectorization
    for both Geometric and Candlestick patterns.
    """
    np.random.seed(42)
    n = 1000

    # Random Walk
    close = np.cumsum(np.random.randn(n)) + 1000
    highs = close + np.abs(np.random.randn(n)) * 2
    lows = close - np.abs(np.random.randn(n)) * 2
    opens = close + np.random.randn(n) * 0.5 # Close to close

    # Inject Geometric Patterns
    # Compression (idx 100)
    highs[90:95] = 1100; lows[90:95] = 900 # Range 200
    highs[95:101] = 1010; lows[95:101] = 990 # Range 20 < 200 * 0.7

    # Wedge (idx 200)
    base = 200
    lows[base-4] = 950; highs[base-4] = 960
    lows[base] = 952;   highs[base] = 958 # Higher Low, Lower High

    # Breakdown (idx 300)
    base = 300
    lows[base-4:base] = 950
    lows[base] = 940 # Lower than min prev 4

    # Inject Candlestick Patterns
    # Doji (idx 400)
    base = 400
    opens[base] = 1000; close[base] = 1000.1
    highs[base] = 1010; lows[base] = 990 # Range 20, Body 0.1 -> Ratio 0.005 < 0.1

    # Hammer (idx 500)
    base = 500
    opens[base] = 1000; close[base] = 1002 # Body 2
    highs[base] = 1002.5 # Upper shadow 0.5 (small)
    lows[base] = 990 # Lower shadow 10 (long) > 2 * Body

    # Engulfing Bull (idx 600)
    base = 600
    # Prev: Red
    opens[base-1] = 1000; close[base-1] = 995
    # Curr: Green, engulfs
    opens[base] = 994; close[base] = 1001

    # CPU
    geo_cpu = detect_geometric_patterns_vectorized(highs, lows)
    cdl_cpu = detect_candlestick_patterns_vectorized(opens, highs, lows, close)

    # GPU
    geo_gpu, cdl_gpu = detect_patterns_cuda(opens, highs, lows, close)

    # Checks
    assert geo_cpu[100] == PATTERN_COMPRESSION, f"CPU missed Compression: {geo_cpu[100]}"
    assert geo_gpu[100] == PATTERN_COMPRESSION, f"GPU missed Compression: {geo_gpu[100]}"

    assert geo_cpu[200] == PATTERN_WEDGE, f"CPU missed Wedge: {geo_cpu[200]}"
    assert geo_gpu[200] == PATTERN_WEDGE, f"GPU missed Wedge: {geo_gpu[200]}"

    assert geo_cpu[300] == PATTERN_BREAKDOWN, f"CPU missed Breakdown: {geo_cpu[300]}"
    assert geo_gpu[300] == PATTERN_BREAKDOWN, f"GPU missed Breakdown: {geo_gpu[300]}"

    assert cdl_cpu[400] == CANDLESTICK_DOJI, f"CPU missed Doji: {cdl_cpu[400]}"
    assert cdl_gpu[400] == CANDLESTICK_DOJI, f"GPU missed Doji: {cdl_gpu[400]}"

    assert cdl_cpu[500] == CANDLESTICK_HAMMER, f"CPU missed Hammer: {cdl_cpu[500]}"
    assert cdl_gpu[500] == CANDLESTICK_HAMMER, f"GPU missed Hammer: {cdl_gpu[500]}"

    assert cdl_cpu[600] == CANDLESTICK_ENGULFING_BULL, f"CPU missed Engulfing: {cdl_cpu[600]}"
    assert cdl_gpu[600] == CANDLESTICK_ENGULFING_BULL, f"GPU missed Engulfing: {cdl_gpu[600]}"

    np.testing.assert_array_equal(geo_cpu, geo_gpu)
    np.testing.assert_array_equal(cdl_cpu, cdl_gpu)

if __name__ == "__main__":
    test_cuda_unified_pattern_equivalence()
