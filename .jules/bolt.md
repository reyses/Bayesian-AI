
2024-05-24 - [Optimize swing noise calculation]
Learning: `np.maximum.accumulate` and `np.minimum.accumulate` within slice allocations inside Python loops incur significant PyObject overhead.
Action: Use `numba.prange` for direct scalar calculation and parallel computation to bypass PyObject and memory overhead.
