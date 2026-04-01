import numpy as np
import time

def compute_swing_noise_orig(highs, lows, n, noise_window, tick_size):
    swing_noise = np.full(n, 35.0)
    for _ni in range(noise_window, n):
        _seg_hi = highs[_ni - noise_window:_ni + 1]
        _seg_lo = lows[_ni - noise_window:_ni + 1]
        _run_hi = np.maximum.accumulate(_seg_hi)
        _dd = (_run_hi - _seg_lo).max() / tick_size
        _run_lo = np.minimum.accumulate(_seg_lo)
        _du = (_seg_hi - _run_lo).max() / tick_size
        swing_noise[_ni] = max(_dd, _du)
    return swing_noise

from numba import njit, prange

@njit(parallel=True, cache=True)
def _compute_swing_noise_numba(highs, lows, n, noise_window, tick_size):
    swing_noise = np.full(n, 35.0)
    for i in prange(noise_window, n):
        start_idx = i - noise_window
        run_hi = highs[start_idx]
        run_lo = lows[start_idx]

        max_dd = run_hi - run_lo
        max_du = run_hi - run_lo

        for j in range(start_idx + 1, i + 1):
            curr_hi = highs[j]
            curr_lo = lows[j]

            if curr_hi > run_hi:
                run_hi = curr_hi
            if curr_lo < run_lo:
                run_lo = curr_lo

            dd = run_hi - curr_lo
            if dd > max_dd:
                max_dd = dd

            du = curr_hi - run_lo
            if du > max_du:
                max_du = du

        swing_noise[i] = max(max_dd, max_du) / tick_size
    return swing_noise

n = 100000
highs = np.random.rand(n) * 100 + 100
lows = highs - np.random.rand(n) * 10
noise_window = 30
tick_size = 0.25

# Compile
_compute_swing_noise_numba(highs[:100], lows[:100], 100, noise_window, tick_size)

t0 = time.perf_counter()
orig = compute_swing_noise_orig(highs, lows, n, noise_window, tick_size)
t1 = time.perf_counter()
print(f"Original: {t1 - t0:.4f}s")

t0 = time.perf_counter()
numba_res = _compute_swing_noise_numba(highs, lows, n, noise_window, tick_size)
t1 = time.perf_counter()
print(f"Numba: {t1 - t0:.4f}s")

print("All close:", np.allclose(orig, numba_res, rtol=1e-4))
