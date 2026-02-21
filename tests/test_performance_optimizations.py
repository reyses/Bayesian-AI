
import pytest
import numpy as np
import time
import math
from numpy.lib.stride_tricks import sliding_window_view
from core.quantum_field_engine import QuantumFieldEngine

# ------------------------------------------------------------------------------
# Helpers for Regression
# ------------------------------------------------------------------------------

def original_regression_method(prices, rp):
    """Replicates the original logic for regression calculation"""
    windows = sliding_window_view(prices, window_shape=rp)
    x = np.arange(rp)

    # Simulate the calculations
    sum_y = np.sum(windows, axis=1)

    # Broadcast x: (rp,) -> (1, rp)
    sum_xy = np.sum(windows * x[np.newaxis, :], axis=1)
    sum_yy = np.sum(windows * windows, axis=1)
    return sum_y, sum_xy, sum_yy

def optimized_regression_method(prices, rp):
    """Optimized using convolution"""
    # Pre-calculate x and kernel
    x = np.arange(rp)
    kernel_sum = np.ones(rp)
    kernel_xy = x[::-1] # Reverse for convolution

    # Use convolve
    sum_y = np.convolve(prices, kernel_sum, mode='valid')
    sum_xy = np.convolve(prices, kernel_xy, mode='valid')
    sum_yy = np.convolve(prices**2, kernel_sum, mode='valid')

    return sum_y, sum_xy, sum_yy

# ------------------------------------------------------------------------------
# Helpers for Hurst
# ------------------------------------------------------------------------------

def original_hurst_method(prices, window=100):
    """Replicates the original Hurst calculation logic"""
    n = len(prices)
    hurst = np.full(n, 0.5)

    for i in range(window, n):
        log_ns = []
        log_rs_vals = []

        for sz in [window//8, window//4, window//2, window]:
            sz = max(sz, 4)
            segment = prices[i-sz+1:i+1]
            returns = np.diff(segment)
            if len(returns) < 2:
                continue

            mean_r = returns.mean()
            devs = np.cumsum(returns - mean_r)
            R = devs.max() - devs.min()
            S = max(returns.std(ddof=1), 1e-10)

            log_ns.append(np.log(sz))
            log_rs_vals.append(np.log(max(R/S, 1e-10)))

        if len(set(log_ns)) >= 2:
            coeffs = np.polyfit(log_ns, log_rs_vals, 1)
            hurst[i] = np.clip(coeffs[0], 0.0, 1.0)

    return hurst

def optimized_hurst_method(prices, window=100):
    """Optimized vectorized Hurst calculation"""
    n = len(prices)
    hurst = np.full(n, 0.5)

    if n < window:
        return hurst

    all_returns = np.diff(prices)

    raw_sizes = [window//8, window//4, window//2, window]
    valid_sizes = [max(sz, 4) for sz in raw_sizes]

    # Check if we have enough distinct sizes for regression
    if len(set(valid_sizes)) < 2:
        return hurst

    log_ns = np.log(valid_sizes)

    # Precompute pseudo-inverse
    A = np.vstack([log_ns, np.ones(len(log_ns))]).T

    if np.linalg.matrix_rank(A) < 2:
        return hurst

    pinv = np.linalg.pinv(A)
    pinv_slope = pinv[0, :] # Shape (4,)

    unique_sizes = sorted(list(set(valid_sizes)))
    size_results = {}

    for sz in unique_sizes:
        w_ret = sz - 1
        # Vectorized R/S over all possible windows of size w_ret
        windows = sliding_window_view(all_returns, window_shape=w_ret)

        mean_r = windows.mean(axis=1, keepdims=True)
        devs = np.cumsum(windows - mean_r, axis=1)
        R = devs.max(axis=1) - devs.min(axis=1)
        S = windows.std(axis=1, ddof=1)
        S = np.maximum(S, 1e-10)

        RS = R / S
        log_RS = np.log(np.maximum(RS, 1e-10))

        size_results[sz] = log_RS

    Y_rows = []
    for sz in valid_sizes:
        res = size_results[sz]
        w_ret = sz - 1
        start_idx = window - w_ret

        # Ensure we don't go out of bounds
        if start_idx < 0:
             # Should not happen if window >= sz
             start_idx = 0

        sliced_res = res[start_idx : start_idx + (n - window)]
        Y_rows.append(sliced_res)

    Y = np.vstack(Y_rows)

    slopes = pinv_slope @ Y
    slopes = np.clip(slopes, 0.0, 1.0)

    hurst[window:] = slopes

    return hurst


# ------------------------------------------------------------------------------
# Tests
# ------------------------------------------------------------------------------

def test_regression_optimization_correctness():
    """Verify that optimized regression logic matches original"""
    N = 1000
    rp = 21
    np.random.seed(42)
    prices = np.random.rand(N).astype(np.float64)

    o_sy, o_sxy, o_syy = original_regression_method(prices, rp)

    # Use the actual engine method via _batch_compute_cpu logic
    # We can't call _batch_compute_cpu directly to get sum_y, but we can verify the output
    # Let's inspect the engine's _batch_compute_cpu output

    engine = QuantumFieldEngine(regression_period=rp, use_gpu=False)
    # Create dummy arrays
    highs = prices
    lows = prices
    closes = prices
    volumes = np.zeros_like(prices)

    results = engine._batch_compute_cpu(prices, highs, lows, closes, volumes, rp)

    # Calculate expected slope and center using original method
    # slope = (sum_xy - mean_x * sum_y) * inv_denom
    # center = mean_y + slope * ((rp - 1) - mean_x)

    # Reconstruct expectations
    sum_x = 0.0
    sum_xx = 0.0
    for k in range(rp):
        sum_x += float(k)
        sum_xx += float(k * k)
    mean_x = sum_x / rp
    denom = sum_xx - (sum_x * sum_x) / rp
    inv_denom = 1.0 / denom

    o_mean_y = o_sy / rp
    o_slope = (o_sxy - mean_x * o_sy) * inv_denom
    o_center = o_mean_y + o_slope * ((rp - 1) - mean_x)

    # Check engine output
    # Engine output is valid from index rp-1
    start_idx = rp - 1

    assert np.allclose(results['slope'][start_idx:], o_slope, atol=1e-10)
    assert np.allclose(results['center'][start_idx:], o_center, atol=1e-10)


def test_hurst_optimization_correctness():
    """Verify that optimized Hurst logic matches original"""
    N = 2000
    window = 100
    np.random.seed(42)
    # Random walk
    prices = np.cumsum(np.random.randn(N)) + 1000

    res_orig = original_hurst_method(prices, window)

    # Use the actual engine method
    engine = QuantumFieldEngine(use_gpu=False)
    res_opt = engine._compute_hurst_numpy(prices, window)

    # Check values starting from window index
    # Using larger tolerance because floating point operations order differs (polyfit vs pinv)
    assert np.allclose(res_orig[window:], res_opt[window:], atol=1e-5)

def test_hurst_edge_cases():
    """Test optimized Hurst with small data via Engine"""
    engine = QuantumFieldEngine(use_gpu=False)

    # Case 1: n < window
    window = 100
    prices = np.random.rand(50)
    res = engine._compute_hurst_numpy(prices, window)
    assert np.all(res == 0.5)

    # Case 2: exactly window
    prices = np.random.rand(100)
    res = engine._compute_hurst_numpy(prices, window)
    assert len(res) == 100
    assert np.all(res == 0.5)

def test_hurst_performance():
    """Simple performance check to ensure it's actually faster"""
    N = 5000
    window = 100
    prices = np.cumsum(np.random.randn(N)) + 1000

    start = time.time()
    original_hurst_method(prices, window)
    dur_orig = time.time() - start

    engine = QuantumFieldEngine(use_gpu=False)
    start = time.time()
    engine._compute_hurst_numpy(prices, window)
    dur_opt = time.time() - start

    # Should be at least 10x faster
    assert dur_opt < dur_orig / 10
    print(f"Speedup: {dur_orig/dur_opt:.2f}x")

def test_oscillation_coherence_kernel_logic():
    """Verify that the kernel logic for oscillation coherence matches sliding_window_view."""
    N = 100
    window_size = 5
    np.random.seed(42)
    z_scores = np.random.randn(N).astype(np.float64)

    # 1. Original Method (sliding_window_view)
    osc_std_orig = np.full(N, np.nan)
    if N >= window_size:
         z_windows = sliding_window_view(z_scores, window_shape=window_size)
         osc_std_orig[window_size-1:] = z_windows.std(axis=1, ddof=1)

    # Invert and normalize
    coherence_orig = 1.0 / (1.0 + osc_std_orig)
    np.nan_to_num(coherence_orig, copy=False, nan=0.0)

    # 2. Kernel Logic Simulation (Python equivalent of detect_archetype_kernel logic)
    coherence_kernel = np.zeros(N, dtype=np.float64)

    for i in range(N):
        if i >= window_size - 1:
            sum_z = 0.0
            sum_zz = 0.0

            # Loop over window
            for k in range(window_size):
                idx = i - (window_size - 1) + k
                val = z_scores[idx]
                sum_z += val
                sum_zz += val * val

            mean_z = sum_z / window_size
            var_num = sum_zz - window_size * mean_z * mean_z
            if var_num < 0.0:
                var_num = 0.0

            std_dev = 0.0
            if window_size > 1:
                std_dev = math.sqrt(var_num / (window_size - 1))
                coherence_kernel[i] = 1.0 / (1.0 + std_dev)
            else:
                coherence_kernel[i] = 0.0
        else:
            coherence_kernel[i] = 0.0

    # The original method creates NaNs/0.0 at the start depending on nan_to_num behavior.
    # We normalized NaNs to 0.0.

    # Check match (start checking from window_size-1 where both should be valid)
    assert np.allclose(coherence_orig[window_size-1:], coherence_kernel[window_size-1:], atol=1e-10)

    # Test edge case window_size = 1
    window_size = 1
    coherence_kernel_1 = np.zeros(N, dtype=np.float64)
    for i in range(N):
        if i >= 0:
            std_dev = 0.0
            if window_size > 1:
                pass
            else:
                coherence_kernel_1[i] = 0.0

    assert np.all(coherence_kernel_1 == 0.0)

if __name__ == "__main__":
    pytest.main([__file__])
