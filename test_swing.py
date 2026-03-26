import numpy as np
import time
import numba

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

@numba.njit(parallel=True, cache=True)
def compute_swing_noise_numba(highs, lows, n, noise_window, tick_size):
    swing_noise = np.full(n, 35.0)
    for i in numba.prange(noise_window, n):
        start_idx = i - noise_window

        # We need to compute:
        # _dd = (np.maximum.accumulate(_seg_hi) - _seg_lo).max()
        # _du = (_seg_hi - np.minimum.accumulate(_seg_lo)).max()

        # Max drawdown from running high
        run_hi = highs[start_idx]
        max_dd = run_hi - lows[start_idx]

        # Max drawup from running low
        run_lo = lows[start_idx]
        max_du = highs[start_idx] - run_lo

        for j in range(start_idx + 1, i + 1):
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

        dd_ticks = max_dd / tick_size
        du_ticks = max_du / tick_size
        swing_noise[i] = dd_ticks if dd_ticks > du_ticks else du_ticks
    return swing_noise

def test():
    n = 100000
    np.random.seed(42)
    highs = np.cumsum(np.random.randn(n)) + 1000
    lows = highs - np.random.rand(n) * 2

    noise_window = 30
    tick_size = 0.25

    # Warmup
    compute_swing_noise_numba(highs, lows, n, noise_window, tick_size)

    t0 = time.perf_counter()
    res_py = compute_swing_noise_python(highs, lows, n, noise_window, tick_size)
    t1 = time.perf_counter()
    res_nb = compute_swing_noise_numba(highs, lows, n, noise_window, tick_size)
    t2 = time.perf_counter()

    print(f"Python: {t1-t0:.4f}s")
    print(f"Numba:  {t2-t1:.4f}s")
    print(f"Speedup: {(t1-t0)/(t2-t1):.2f}x")
    print(f"Allclose: {np.allclose(res_py, res_nb, rtol=1e-4)}")

if __name__ == '__main__':
    test()
