import numpy as np
import time
from core.quantum_field_engine import _compute_rs_numba
from numpy.lib.stride_tricks import sliding_window_view
import numba

@numba.njit(parallel=True, cache=True)
def optimized_rs_numba(returns, window):
    n = len(returns)
    output_rs = np.empty(n - window + 1, dtype=np.float64)

    for i in numba.prange(n - window + 1):
        # Calculate mean
        mean = 0.0
        for j in range(window):
            mean += returns[i+j]
        mean /= window

        # Calculate deviations and std
        current_cum = 0.0
        max_cum = -np.inf
        min_cum = np.inf
        sum_sq = 0.0

        for j in range(window):
            val = returns[i+j] - mean
            current_cum += val

            if current_cum > max_cum:
                max_cum = current_cum
            if current_cum < min_cum:
                min_cum = current_cum

            sum_sq += val * val

        std_dev = np.sqrt(sum_sq / (window - 1)) if window > 1 else 1e-10
        if std_dev < 1e-10:
            std_dev = 1e-10

        r = max_cum - min_cum
        output_rs[i] = r / std_dev

    return output_rs

np.random.seed(42)
returns = np.random.randn(200000)
window = 100

# Warmup
_compute_rs_numba(returns, window)
optimized_rs_numba(returns, window)

start = time.perf_counter()
rs1 = _compute_rs_numba(returns, window)
t1 = time.perf_counter() - start

start = time.perf_counter()
rs2 = optimized_rs_numba(returns, window)
t2 = time.perf_counter() - start

print(f"Original Time: {t1:.6f}s")
print(f"Parallel Time: {t2:.6f}s")
print(f"Speedup: {t1/t2:.2f}x")
print(f"Max diff: {np.abs(rs1 - rs2).max()}")
