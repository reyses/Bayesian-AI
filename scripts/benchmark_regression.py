import time
import numpy as np
import logging
from numpy.lib.stride_tricks import sliding_window_view

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def naive_rolling_regression(close, rp):
    n = len(close)
    centers = np.zeros(n)
    slopes = np.zeros(n)

    x = np.arange(rp)
    x_mean = x.mean()
    x_var = np.sum((x - x_mean)**2)

    for i in range(n):
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
    windows = sliding_window_view(close, window_shape=rp)

    x = np.arange(rp)
    x_mean = x.mean()
    x_var = np.sum((x - x_mean)**2)

    y_mean = np.mean(windows, axis=1)
    y_centered = windows - y_mean[:, np.newaxis]
    x_centered = x - x_mean
    numerator = np.sum(x_centered * y_centered, axis=1)

    slope = numerator / x_var
    intercept = y_mean - slope * x_mean
    center = slope * x[-1] + intercept

    pad = np.zeros(rp - 1)
    full_centers = np.concatenate([pad, center])
    full_slopes = np.concatenate([pad, slope])

    return full_centers, full_slopes

def benchmark():
    np.random.seed(42)
    data_len = 100000
    rp = 21
    close = np.random.randn(data_len).cumsum() + 1000

    logger.info(f"Data length: {data_len}, Regression Period: {rp}")

    # Benchmark Naive
    start = time.time()
    res_naive = naive_rolling_regression(close, rp)
    end = time.time()
    naive_time = end - start
    logger.info(f"Naive Loop Time: {naive_time:.4f}s")

    # Benchmark Optimized
    start = time.time()
    res_opt = optimized_rolling_regression(close, rp)
    end = time.time()
    opt_time = end - start
    logger.info(f"Optimized Time: {opt_time:.4f}s")

    logger.info(f"Speedup: {naive_time / opt_time:.2f}x")

    # Verify correctness
    np.testing.assert_allclose(res_naive[0][rp-1:], res_opt[0][rp-1:], atol=1e-10)
    np.testing.assert_allclose(res_naive[1][rp-1:], res_opt[1][rp-1:], atol=1e-10)
    logger.info("Verification Passed!")

if __name__ == "__main__":
    benchmark()
