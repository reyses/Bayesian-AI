import numpy as np
import time
from numba import njit, prange

def original_swing_noise(highs, lows, n, _noise_window, _tick_size):
    swing_noise = np.full(n, 35.0)
    for _ni in range(_noise_window, n):
        _seg_hi = highs[_ni - _noise_window:_ni + 1]
        _seg_lo = lows[_ni - _noise_window:_ni + 1]
        _run_hi = np.maximum.accumulate(_seg_hi)
        _dd = (_run_hi - _seg_lo).max() / _tick_size
        _run_lo = np.minimum.accumulate(_seg_lo)
        _du = (_seg_hi - _run_lo).max() / _tick_size
        swing_noise[_ni] = max(_dd, _du)
    return swing_noise

@njit(parallel=True, cache=True)
def numba_swing_noise(highs, lows, n, _noise_window, _tick_size):
    swing_noise = np.full(n, 35.0)
    for i in prange(_noise_window, n):
        start_idx = i - _noise_window

        run_hi = highs[start_idx]
        run_lo = lows[start_idx]

        # Initialize correctly per memory instructions
        max_dd = run_hi - run_lo
        max_du = run_hi - run_lo

        for k in range(start_idx + 1, i + 1):
            h = highs[k]
            l = lows[k]

            if h > run_hi:
                run_hi = h

            dd = run_hi - l
            if dd > max_dd:
                max_dd = dd

            if l < run_lo:
                run_lo = l

            du = h - run_lo
            if du > max_du:
                max_du = du

        swing_noise[i] = max(max_dd, max_du) / _tick_size

    return swing_noise

# Test correctness
np.random.seed(42)
n = 10000
highs = np.random.randn(n).cumsum()
lows = highs - np.random.rand(n) * 2

_noise_window = 32
_tick_size = 0.25

out1 = original_swing_noise(highs, lows, n, _noise_window, _tick_size)

# Compile first
numba_swing_noise(highs, lows, n, _noise_window, _tick_size)

out2 = numba_swing_noise(highs, lows, n, _noise_window, _tick_size)

print("All close:", np.allclose(out1, out2))

# Benchmark
import timeit

t1 = timeit.timeit(lambda: original_swing_noise(highs, lows, n, _noise_window, _tick_size), number=10)
print(f"Original: {t1:.4f}s")

t2 = timeit.timeit(lambda: numba_swing_noise(highs, lows, n, _noise_window, _tick_size), number=10)
print(f"Numba: {t2:.4f}s")
