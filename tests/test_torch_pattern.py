import pytest
import numpy as np
import pandas as pd
from core.pattern_utils import (
    detect_geometric_patterns_vectorized, detect_candlestick_patterns_vectorized,
    PATTERN_COMPRESSION, PATTERN_WEDGE, PATTERN_BREAKDOWN, PATTERN_NONE,
    CANDLESTICK_DOJI, CANDLESTICK_HAMMER, CANDLESTICK_ENGULFING_BULL, CANDLESTICK_ENGULFING_BEAR, CANDLESTICK_NONE
)
from core.quantum_field_engine import QuantumFieldEngine

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

@pytest.mark.skipif(not TORCH_AVAILABLE, reason="Torch not available")
def test_torch_pattern_equivalence():
    """
    Verify that Torch pattern detection produces identical results to CPU vectorization
    for both Geometric and Candlestick patterns.
    """
    np.random.seed(42)
    n = 1000

    # Random Walk
    rng = np.random.default_rng(seed=42) # Use a fixed seed for determinism
    closes = np.cumsum(rng.standard_normal(n)) + 1000
    highs = closes + np.abs(rng.standard_normal(n)) * 2
    lows = closes - np.abs(rng.standard_normal(n)) * 2
    opens = closes + rng.standard_normal(n) * 0.5 # Close to close

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

    # CPU Reference
    geo_cpu = detect_geometric_patterns_vectorized(highs, lows)
    cdl_cpu = detect_candlestick_patterns_vectorized(opens, highs, lows, close)

    # Torch Implementation
    engine = QuantumFieldEngine()
    # Force GPU usage flag to ensure we might test logic branches if needed,
    # but we are calling _detect_patterns_torch directly.
    # Prepare tensors on CPU (to run everywhere)
    device = torch.device('cpu')

    t_opens = torch.tensor(opens, device=device, dtype=torch.float64)
    t_highs = torch.tensor(highs, device=device, dtype=torch.float64)
    t_lows = torch.tensor(lows, device=device, dtype=torch.float64)
    t_closes = torch.tensor(close, device=device, dtype=torch.float64)

    geo_codes_t, cdl_codes_t = engine._detect_patterns_torch(t_opens, t_highs, t_lows, t_closes)

    geo_codes = geo_codes_t.numpy()
    cdl_codes = cdl_codes_t.numpy()

    # Map codes to strings
    geo_lookup = np.array([
        PATTERN_NONE, PATTERN_COMPRESSION, PATTERN_WEDGE, PATTERN_BREAKDOWN
    ], dtype=object)

    cdl_lookup = np.array([
        CANDLESTICK_NONE, CANDLESTICK_DOJI, CANDLESTICK_HAMMER,
        CANDLESTICK_ENGULFING_BULL, CANDLESTICK_ENGULFING_BEAR
    ], dtype=object)

    geo_torch = geo_lookup[geo_codes]
    cdl_torch = cdl_lookup[cdl_codes]

    # Checks
    assert geo_cpu[100] == PATTERN_COMPRESSION
    assert geo_torch[100] == PATTERN_COMPRESSION, f"Torch missed Compression: {geo_torch[100]}"

    assert geo_cpu[200] == PATTERN_WEDGE
    assert geo_torch[200] == PATTERN_WEDGE, f"Torch missed Wedge: {geo_torch[200]}"

    assert geo_cpu[300] == PATTERN_BREAKDOWN
    assert geo_torch[300] == PATTERN_BREAKDOWN, f"Torch missed Breakdown: {geo_torch[300]}"

    assert cdl_cpu[400] == CANDLESTICK_DOJI
    assert cdl_torch[400] == CANDLESTICK_DOJI, f"Torch missed Doji: {cdl_torch[400]}"

    assert cdl_cpu[500] == CANDLESTICK_HAMMER
    assert cdl_torch[500] == CANDLESTICK_HAMMER, f"Torch missed Hammer: {cdl_torch[500]}"

    assert cdl_cpu[600] == CANDLESTICK_ENGULFING_BULL
    assert cdl_torch[600] == CANDLESTICK_ENGULFING_BULL, f"Torch missed Engulfing: {cdl_torch[600]}"

    # Full array comparison
    # Note: There might be slight differences if CPU implementation has bugs that Torch fixes or vice versa,
    # but we aligned logic carefully.

    diff_geo = np.where(geo_cpu != geo_torch)[0]
    if len(diff_geo) > 0:
        print(f"Geometric diffs at: {diff_geo[:10]}")
        print(f"CPU: {geo_cpu[diff_geo[:5]]}")
        print(f"Torch: {geo_torch[diff_geo[:5]]}")

    diff_cdl = np.where(cdl_cpu != cdl_torch)[0]
    if len(diff_cdl) > 0:
        print(f"Candlestick diffs at: {diff_cdl[:10]}")
        print(f"CPU: {cdl_cpu[diff_cdl[:5]]}")
        print(f"Torch: {cdl_torch[diff_cdl[:5]]}")

    assert len(diff_geo) == 0
    assert len(diff_cdl) == 0

if __name__ == "__main__":
    test_torch_pattern_equivalence()
