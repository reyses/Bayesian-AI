import pytest
import numpy as np
import pandas as pd
from core.pattern_utils import detect_geometric_patterns_vectorized, PATTERN_COMPRESSION, PATTERN_WEDGE, PATTERN_BREAKDOWN

# Try importing CUDA module
try:
    from core.cuda_pattern_detector import detect_geometric_patterns_cuda, NUMBA_AVAILABLE
    from numba import cuda
    CUDA_READY = NUMBA_AVAILABLE and cuda.is_available()
except (ImportError, Exception):
    CUDA_READY = False

@pytest.mark.skipif(not CUDA_READY, reason="CUDA not available")
def test_cuda_pattern_equivalence():
    """
    Verify that CUDA pattern detection produces identical results to CPU vectorization.
    """
    np.random.seed(42)
    n = 1000

    # Generate synthetic OHLC data
    # Create random walk for close
    close = np.cumsum(np.random.randn(n)) + 100
    highs = close + np.random.rand(n) * 2
    lows = close - np.random.rand(n) * 2

    # Inject patterns manually to ensure coverage

    # Inject Compression at index 100
    # prev_range (90-95): large
    highs[90:95] = 110
    lows[90:95] = 90
    # rec_range (95-100): small
    highs[95:101] = 101
    lows[95:101] = 99

    # Inject Wedge at index 200 (Higher Lows, Lower Highs)
    # i-4 to i
    base = 200
    lows[base-4] = 50; highs[base-4] = 60
    lows[base] = 52;   highs[base] = 58

    # Inject Breakdown at index 300
    # i-4 to i-1 min low = 50
    # i low = 48
    base = 300
    lows[base-4:base] = 50
    lows[base] = 48

    # Run CPU
    cpu_result = detect_geometric_patterns_vectorized(highs, lows)

    # Run GPU
    gpu_result = detect_geometric_patterns_cuda(highs, lows)

    # Compare
    # Check if injected patterns were detected by both (sanity check)
    assert cpu_result[100] == PATTERN_COMPRESSION
    assert gpu_result[100] == PATTERN_COMPRESSION

    assert cpu_result[200] == PATTERN_WEDGE
    assert gpu_result[200] == PATTERN_WEDGE

    assert cpu_result[300] == PATTERN_BREAKDOWN
    assert gpu_result[300] == PATTERN_BREAKDOWN

    # Check full equality
    # Note: There might be edge cases with NaNs or types
    # Ensure string equality
    np.testing.assert_array_equal(cpu_result, gpu_result)

if __name__ == "__main__":
    test_cuda_pattern_equivalence()
