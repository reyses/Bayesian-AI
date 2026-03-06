import numpy as np
import time
from core.quantum_field_engine import _compute_rs_numba

returns = np.random.randn(100000)
window = 100

# Warm up
_compute_rs_numba(returns, window)

start = time.perf_counter()
_compute_rs_numba(returns, window)
end = time.perf_counter()

print(f"Time: {end - start:.6f}s")
