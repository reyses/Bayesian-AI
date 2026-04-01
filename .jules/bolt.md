2025-02-28 - [Replace python loops over sliced arrays in `compute_swing_noise`]
Learning: Using `np.maximum.accumulate` and slicing inside a standard python loop creates massive PyObject and array allocation overhead.
Action: Replace sliding windows of complex accumulations with a unified Numba JIT `@njit(parallel=True, cache=True)` and `prange` loops. Calculate metrics directly using scalar variables (`run_hi`, `max_dd`).
