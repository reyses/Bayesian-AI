## 2025-05-23 - [Numpy Vectorization vs Object Creation]
**Learning:** Optimizing logic inside a Python loop (using vectorized numpy arrays for `if/else` logic) provided only marginal gains (~20ms on 1s task) because the dominant cost was the `dataclass` instantiation itself, which happens in Python.
**Action:** When optimizing object creation loops, focus on reducing the number of objects or using bulk constructors if possible. If not, look for other bottlenecks (like heavy computations called within the process). In this case, optimizing the Hurst calculation (heavy math) using Numba provided 6x speedup for that component, yielding a larger overall gain.

## 2025-05-23 - [Manual Loop Unrolling inside Numba]
**Learning:** Relying on standard numerical library functions like `scipy.fft.fft` or `np.polyfit` inside a per-bar loop can introduce massive overhead (around 16s for 100k iterations). By manually unrolling the mathematical logic (such as a discrete fourier transform for a few frequencies or the linear regression sum equations) and decorating with `@numba.njit(cache=True, fastmath=True)`, we bypass Python function call overhead and intermediate array allocations.
**Action:** For simple but frequent math operations within hot paths, manual unrolling combined with Numba's `fastmath` optimization yields extreme performance boosts (>10x). Combine this with `prange` parallelism where appropriate to maximize throughput.

## 2026-02-26 - [Numba JIT for Wilder Smoothing]
**Learning:** Python loops with data dependencies (like Wilder smoothing for ADX) are extremely slow and cannot be easily vectorized with NumPy. JIT compiling these sequential loops with Numba provided a massive ~23x speedup (3.1s -> 0.13s for 1M points), far exceeding gains from micro-optimizing attribute access.
**Action:** Always prioritize JIT compilation for sequential numeric loops (accumulators, recursive filters) over trying to force them into NumPy vectorization or optimizing object access.
