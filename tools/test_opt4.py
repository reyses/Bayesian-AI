import numpy as np
import time
from core.physics_utils import extract_dominant_cycle
from numba import njit, prange

z = np.random.randn(60)

t0 = time.perf_counter()
for i in range(100000):
    p = extract_dominant_cycle(z, 1.0)
t1 = time.perf_counter()

print(f"Original time: {t1-t0:.4f}")

@njit(cache=True, fastmath=True)
def _extract_dominant_cycle_numba(z_scores, dt):
    # Try using np.fft.fft inside njit if numba supports it? No, numba does not support np.fft.fft.
    # What about np.correlate or similar?
    pass
