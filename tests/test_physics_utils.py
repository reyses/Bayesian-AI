import numpy as np
import time
import pytest
from core.physics_utils import compute_adx_dmi_cpu

def reference_adx_dmi_cpu(tr_raw, plus_dm_raw, minus_dm_raw, period=14):
    """
    Pure Python reference implementation of Wilder's smoothed ADX/DMI.
    """
    n = len(tr_raw)
    adx = np.zeros(n)
    dmi_plus = np.zeros(n)
    dmi_minus = np.zeros(n)

    if n < period + 1:
        return adx, dmi_plus, dmi_minus

    # Initial sums (first `period` bars)
    smooth_tr = np.sum(tr_raw[1:period+1])
    smooth_plus = np.sum(plus_dm_raw[1:period+1])
    smooth_minus = np.sum(minus_dm_raw[1:period+1])

    # First DI values
    if smooth_tr > 0:
        dmi_plus[period] = 100.0 * smooth_plus / smooth_tr
        dmi_minus[period] = 100.0 * smooth_minus / smooth_tr

    # First DX
    di_sum = dmi_plus[period] + dmi_minus[period]
    if di_sum > 0:
        dx_first = 100.0 * abs(dmi_plus[period] - dmi_minus[period]) / di_sum
    else:
        dx_first = 0.0

    # Wilder smoothing for remaining bars
    dx_sum = dx_first
    dx_count = 1

    for i in range(period + 1, n):
        # Wilder smoothing: smooth = prev_smooth - (prev_smooth / period) + current
        smooth_tr = smooth_tr - (smooth_tr / period) + tr_raw[i]
        smooth_plus = smooth_plus - (smooth_plus / period) + plus_dm_raw[i]
        smooth_minus = smooth_minus - (smooth_minus / period) + minus_dm_raw[i]

        if smooth_tr > 0:
            dmi_plus[i] = 100.0 * smooth_plus / smooth_tr
            dmi_minus[i] = 100.0 * smooth_minus / smooth_tr

        di_sum = dmi_plus[i] + dmi_minus[i]
        if di_sum > 0:
            dx = 100.0 * abs(dmi_plus[i] - dmi_minus[i]) / di_sum
        else:
            dx = 0.0

        # ADX = Wilder smoothed DX
        if dx_count < period:
            dx_sum += dx
            dx_count += 1
            if dx_count == period:
                adx[i] = dx_sum / period
        else:
            adx[i] = (adx[i-1] * (period - 1) + dx) / period

    return adx, dmi_plus, dmi_minus

def test_adx_correctness():
    # Setup random data
    n = 1000
    np.random.seed(42)
    tr = np.random.random(n) + 1.0
    plus = np.random.random(n)
    minus = np.random.random(n)

    # Run reference (Pure Python)
    adx_ref, plus_ref, minus_ref = reference_adx_dmi_cpu(tr, plus, minus)

    # Run optimized (Numba) - assuming core.physics_utils is updated
    # Currently it's the same, so this test passes trivially.
    # After update, it ensures regression matching.
    adx_opt, plus_opt, minus_opt = compute_adx_dmi_cpu(tr, plus, minus)

    # Verify
    np.testing.assert_allclose(adx_opt, adx_ref, rtol=1e-8, atol=1e-8)
    np.testing.assert_allclose(plus_opt, plus_ref, rtol=1e-8, atol=1e-8)
    np.testing.assert_allclose(minus_opt, minus_ref, rtol=1e-8, atol=1e-8)

def test_adx_speed():
    # Large data for benchmark
    n = 100000
    np.random.seed(42)
    tr = np.random.random(n) + 1.0
    plus = np.random.random(n)
    minus = np.random.random(n)

    # Warmup Numba (if present)
    compute_adx_dmi_cpu(tr[:100], plus[:100], minus[:100])

    # Time Reference
    t0 = time.perf_counter()
    reference_adx_dmi_cpu(tr, plus, minus)
    t_ref = time.perf_counter() - t0

    # Time Optimized
    t0 = time.perf_counter()
    compute_adx_dmi_cpu(tr, plus, minus)
    t_opt = time.perf_counter() - t0

    print(f"Reference: {t_ref:.4f}s | Optimized: {t_opt:.4f}s | Speedup: {t_ref/t_opt:.2f}x")

    # Assert at least 5x speedup (will fail until optimization is applied)
    # We can use try-except or just let it fail/pass
    if t_ref > 0.05: # Only check if ref takes measurable time
        speedup = t_ref / t_opt if t_opt > 0 else float('inf')
        assert speedup > 5, f"Expected at least 5x speedup, but got {speedup:.2f}x"

if __name__ == "__main__":
    test_adx_correctness()
    test_adx_speed()
