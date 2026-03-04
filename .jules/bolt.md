## 2024-05-18 - [Numba JIT vs sliding_window_view]
Learning: For small rolling window calculations (like a 5-period standard deviation for Oscillation Coherence), `numpy.lib.stride_tricks.sliding_window_view` followed by `.std(axis=1)` incurs severe execution and memory overhead compared to a custom `@numba.njit(cache=True)` rolling loop. This overhead is critical in hot paths running every bar (e.g. `batch_compute_states`).
Action: Use explicit Numba loops (`@njit(cache=True)`) instead of `sliding_window_view` for rolling aggregations on large arrays in hot paths.
