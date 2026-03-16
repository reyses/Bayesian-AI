YYYY-MM-DD - [Parallelizing Rolling R/S calculation]
Learning: Numba's `@njit(parallel=True)` combined with `prange` allows seamless parallelization of sliding window calculations without breaking numerical accuracy or introducing any dependencies. The original `_compute_rs_numba` iterates independently over the output array items, calculating rolling stats. Changing `range` to `numba.prange` takes full advantage of multiple cores without changing the algorithm logic.
Action: Add `@njit(parallel=True, cache=True)` and replace `range` with `prange` for completely independent tight rolling calculations on 1D arrays, as long as there is no data mutation inside the loop across iterations.

2024-05-24 - [Optimising swing noise calculation]
Learning: Calculating rolling drawdowns and drawups within a python loop using array slicing and `np.maximum.accumulate` incurs significant overhead from array allocations and python object overhead. Moving the calculation to a `@njit(parallel=True, cache=True)` function with scalar calculations inside `prange` makes it over 150x faster.
Action: Replace numpy slice accumulators inside loops with parallel Numba JIT functions doing scalar math.