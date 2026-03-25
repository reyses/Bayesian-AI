2025-03-02 - [Numba JIT for Drawdown/Drawup tracking]
Learning: Running tracking functions (like max drawdown/drawup tracking via `np.maximum.accumulate` and array slicing) within an iteration loop incurs extremely high PyObject and array allocation overhead.
Action: Write custom Numba JIT functions using `prange` and `@njit(parallel=True, cache=True)` instead of NumPy slicing for rolling window calculations of running min/max. This eliminates slicing overhead and avoids creating intermediate arrays, easily yielding over 180x speedup.
