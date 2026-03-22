import numpy as np
import time
from numba import njit, prange

def compute_swing_noise_python(highs, lows, n, noise_window, tick_size):
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

@njit(parallel=True, cache=True)
def _compute_swing_noise_numba(highs, lows, n, noise_window, tick_size):
    swing_noise = np.full(n, 35.0)
    for i in prange(noise_window, n):
        run_hi = highs[i - noise_window]
        run_lo = lows[i - noise_window]

        # Proper initialization according to memory
        max_dd = run_hi - lows[i - noise_window]
        max_du = highs[i - noise_window] - run_lo

        for j in range(i - noise_window + 1, i + 1):
            if highs[j] > run_hi:
                run_hi = highs[j]
            if lows[j] < run_lo:
                run_lo = lows[j]

            dd = run_hi - lows[j]
            if dd > max_dd:
                max_dd = dd

            du = highs[j] - run_lo
            if du > max_du:
                max_du = du

        swing_noise[i] = max(max_dd / tick_size, max_du / tick_size)

    return swing_noise

n = 100000
np.random.seed(42)
highs = np.random.rand(n) * 100 + 100
lows = highs - np.random.rand(n) * 2
noise_window = 32
tick_size = 0.25

# warmup
_compute_swing_noise_numba(highs[:100], lows[:100], 100, noise_window, tick_size)

t0 = time.perf_counter()
res_py = compute_swing_noise_python(highs, lows, n, noise_window, tick_size)
t1 = time.perf_counter()

t2 = time.perf_counter()
res_nb = _compute_swing_noise_numba(highs, lows, n, noise_window, tick_size)
t3 = time.perf_counter()

print(f"Python: {t1 - t0:.4f}s")
print(f"Numba:  {t3 - t2:.4f}s")
print(f"Speedup: {(t1-t0)/(t3-t2):.2f}x")
print(f"Matches: {np.allclose(res_py, res_nb, rtol=1e-4)}")
