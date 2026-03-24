
2024-05-24 - [Replace sliding_window_view with Numba Rolling Std]
Learning: Using `sliding_window_view` combined with `.std(axis=1)` in NumPy creates a lot of memory overhead for small windows and large arrays, preventing efficient scalar computation. The overhead is largely from Python loop allocations inside the axis aggregation.
Action: Whenever calculating simple rolling statistics (like mean or variance) over a tight window inside a hot path (`core/statistical_field_engine.py`), prefer a `@njit(cache=True, parallel=True)` implementation with `prange` over `sliding_window_view`.
