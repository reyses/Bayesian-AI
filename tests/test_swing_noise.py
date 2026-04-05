import numpy as np
import time

def compute_swing_noise_numpy(highs, lows, n, _noise_window, _tick_size):
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

from numba import njit, prange

@njit(parallel=True, cache=True)
def compute_swing_noise_numba(highs, lows, n, noise_window, tick_size):
    swing_noise = np.full(n, 35.0)
    for _ni in prange(noise_window, n):
        start_idx = _ni - noise_window
        run_hi = highs[start_idx]
        run_lo = lows[start_idx]
        max_dd = run_hi - lows[start_idx]
        max_du = highs[start_idx] - run_lo

        for j in range(start_idx + 1, _ni + 1):
            if highs[j] > run_hi:
                run_hi = highs[j]
            dd = run_hi - lows[j]
            if dd > max_dd:
                max_dd = dd

            if lows[j] < run_lo:
                run_lo = lows[j]
            du = highs[j] - run_lo
            if du > max_du:
                max_du = du

        swing_noise[_ni] = max(max_dd, max_du) / tick_size
    return swing_noise

def test_swing_noise():
    n = 100000
    np.random.seed(42)
    base = np.cumsum(np.random.randn(n)) + 1000
    highs = base + np.random.rand(n) * 5
    lows = base - np.random.rand(n) * 5

    # warmup
    compute_swing_noise_numba(highs[:100], lows[:100], 100, 30, 0.25)

    t0 = time.perf_counter()
    res_np = compute_swing_noise_numpy(highs, lows, n, 30, 0.25)
    t1 = time.perf_counter()
    res_nb = compute_swing_noise_numba(highs, lows, n, 30, 0.25)
    t2 = time.perf_counter()

    print(f"NumPy: {t1-t0:.4f}s")
    print(f"Numba: {t2-t1:.4f}s")
    print(f"Speedup: {(t1-t0)/(t2-t1):.2f}x")
    print("All close:", np.allclose(res_np, res_nb))

    assert np.allclose(res_np, res_nb)

if __name__ == '__main__':
    test_swing_noise()
