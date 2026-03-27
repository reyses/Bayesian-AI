
2024-03-27 - [Optimize Swing Noise with Numba prange]
Learning: A large bottleneck in the `StatisticalFieldEngine` was found in tracking the swing noise (maximum drawdown and drawup of running highs and lows). Looping over `np.maximum.accumulate` with rolling slice creations causes massive Python object overhead (~1s for 100k loops).
Action: Replacing the slice allocation loop with a custom scalar logic `@numba.njit(parallel=True, cache=True)` using `numba.prange` resulted in ~250x speedup and no memory allocations, preserving exact numerical parity with NumPy slice views.
