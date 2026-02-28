## 2025-05-23 - [Numpy Vectorization vs Object Creation]
**Learning:** Optimizing logic inside a Python loop (using vectorized numpy arrays for `if/else` logic) provided only marginal gains (~20ms on 1s task) because the dominant cost was the `dataclass` instantiation itself, which happens in Python.
**Action:** When optimizing object creation loops, focus on reducing the number of objects or using bulk constructors if possible. If not, look for other bottlenecks (like heavy computations called within the process). In this case, optimizing the Hurst calculation (heavy math) using Numba provided 6x speedup for that component, yielding a larger overall gain.

## 2025-05-24 - [Avoid `sliding_window_view` for Rolling Aggregations]
**Learning:** Using `numpy.lib.stride_tricks.sliding_window_view(..., window_shape).std(axis=1)` provides a clean API for rolling standard deviations, but it executes poorly on large arrays due to temporary array generation and unoptimized multi-axis operations.
**Action:** Instead, implement a custom `@numba.njit(cache=True)` function that manually iterates and computes the rolling aggregation with `O(1)` memory overhead. This approach provides a significant speedup (e.g., ~8x for computing Oscillation Coherence rolling std).
