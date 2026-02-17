
import time
import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

def naive_rolling_regression(close, rp):
    n = len(close)
    centers = np.zeros(n)
    slopes = np.zeros(n)

    x = np.arange(rp)
    x_mean = x.mean()
    # sum((x - x_mean)**2)
    x_var = np.sum((x - x_mean)**2)

    for i in range(n):
        # Window ending at current bar (inclusive)
        # Note: The snippet had start_idx = i + 1, end_idx = i + 1 + rp
        # This seems to imply a lookahead or the loop index i is the start of the window?
        # Standard rolling window ending at i: close[i-rp+1 : i+1]

        # Let's match the snippet's logic:
        # start_idx = i + 1
        # end_idx = i + 1 + rp
        # This looks like it's iterating from 0 to n-rp-1?
        # Or maybe i is the index in the *output* array which corresponds to the *end* of the window?

        # If I want to calculate for *every* bar, and use a window of size rp:

        if i < rp - 1:
            continue

        y = close[i - rp + 1 : i + 1]

        y_mean = y.mean()
        slope = np.sum((x - x_mean) * (y - y_mean)) / x_var
        intercept = y_mean - slope * x_mean
        center = slope * x[-1] + intercept

        centers[i] = center
        slopes[i] = slope

    return centers, slopes

def optimized_rolling_regression(close, rp):
    n = len(close)

    # 1. Create rolling window view
    # shape: (n - rp + 1, rp)
    windows = sliding_window_view(close, window_shape=rp)

    # 2. Precompute X constants
    x = np.arange(rp)
    x_mean = x.mean()
    x_var = np.sum((x - x_mean)**2)

    # 3. Vectorized regression
    # y_mean per window: shape (n - rp + 1,)
    y_mean = np.mean(windows, axis=1)

    # (x - x_mean) is shape (rp,)
    # (y - y_mean) requires broadcasting
    # windows is (N, rp), y_mean is (N,) -> (N, 1)
    y_centered = windows - y_mean[:, np.newaxis]

    # sum((x - x_mean) * (y - y_mean))
    # broadcast x_centered: (rp,) -> (1, rp)
    x_centered = x - x_mean
    numerator = np.sum(x_centered * y_centered, axis=1)

    slope = numerator / x_var
    intercept = y_mean - slope * x_mean
    center = slope * x[-1] + intercept

    # Pad results to match input length (first rp-1 are 0)
    # We need to prepend rp-1 zeros
    pad = np.zeros(rp - 1)

    full_centers = np.concatenate([pad, center])
    full_slopes = np.concatenate([pad, slope])

    return full_centers, full_slopes

def benchmark():
    np.random.seed(42)
    data_len = 100000
    rp = 21
    close = np.random.randn(data_len).cumsum() + 1000

    print(f"Data length: {data_len}, Regression Period: {rp}")

    # Benchmark Naive
    start = time.time()
    res_naive = naive_rolling_regression(close, rp)
    end = time.time()
    naive_time = end - start
    print(f"Naive Loop Time: {naive_time:.4f}s")

    # Benchmark Optimized
    start = time.time()
    res_opt = optimized_rolling_regression(close, rp)
    end = time.time()
    opt_time = end - start
    print(f"Optimized Time: {opt_time:.4f}s")

    print(f"Speedup: {naive_time / opt_time:.2f}x")

    # Verify correctness
    # Ignore first rp-1 elements
    np.testing.assert_allclose(res_naive[0][rp-1:], res_opt[0][rp-1:], atol=1e-10)
    np.testing.assert_allclose(res_naive[1][rp-1:], res_opt[1][rp-1:], atol=1e-10)
    print("Verification Passed!")

if __name__ == "__main__":
    benchmark()
