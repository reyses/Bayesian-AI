import numpy as np
import time

def original_swing_noise(highs, lows, n, noise_window, tick_size):
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
def optimized_swing_noise(highs, lows, n, noise_window, tick_size):
    swing_noise = np.full(n, 35.0, dtype=np.float64)
    for i in prange(noise_window, n):
        start_idx = i - noise_window
        run_hi = highs[start_idx]
        run_lo = lows[start_idx]
        max_dd = run_hi - lows[start_idx]
        max_du = highs[start_idx] - run_lo

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

def test_swing_noise():
    np.random.seed(42)
    n = 100000
    noise_window = 30
    tick_size = 0.25

    # Generate random walk
    prices = 100.0 + np.random.randn(n).cumsum()
    highs = prices + np.random.rand(n) * 2.0
    lows = prices - np.random.rand(n) * 2.0

    # Warmup
    optimized_swing_noise(highs[:100], lows[:100], 100, noise_window, tick_size)

    t0 = time.perf_counter()
    res_orig = original_swing_noise(highs, lows, n, noise_window, tick_size)
    t1 = time.perf_counter()
    orig_time = t1 - t0

    t0 = time.perf_counter()
    res_opt = optimized_swing_noise(highs, lows, n, noise_window, tick_size)
    t1 = time.perf_counter()
    opt_time = t1 - t0

    print(f"Original: {orig_time:.4f}s")
    print(f"Optimized: {opt_time:.4f}s")
    print(f"Speedup: {orig_time/opt_time:.2f}x")

    assert np.allclose(res_orig, res_opt, rtol=1e-4), "Mismatch!"
    print("Match!")

if __name__ == "__main__":
    test_swing_noise()
