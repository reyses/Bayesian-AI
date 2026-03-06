import numpy as np
from core.quantum_field_engine import _compute_rs_numba
from numpy.lib.stride_tricks import sliding_window_view

np.random.seed(42)
returns = np.random.randn(1000)
window = 100

def original_rs(returns, w_ret):
    windows = sliding_window_view(returns, window_shape=w_ret)
    mean_r = windows.mean(axis=1, keepdims=True)
    devs = np.cumsum(windows - mean_r, axis=1)
    R = devs.max(axis=1) - devs.min(axis=1)
    S = windows.std(axis=1, ddof=1)
    S = np.maximum(S, 1e-10)
    RS = R / S
    return RS

rs1 = _compute_rs_numba(returns, window)
rs2 = original_rs(returns, window)

print(f"Max abs diff: {np.abs(rs1 - rs2).max()}")
print(f"All close: {np.allclose(rs1, rs2)}")
