2025-02-20 - [Avoid np.maximum.accumulate in loops for sliding max]
Learning: Using `np.maximum.accumulate` and `.max()` inside a Python loop for sliding window operations is extremely slow because it creates a new array object on every iteration and does not map efficiently to CPU cache.
Action: Write a custom numba `@njit(parallel=True, cache=True)` function with explicit loops to maintain the running min/max. It's much faster as it processes scalars and easily scales via `prange`.
