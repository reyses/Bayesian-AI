import numpy as np
import time
from core.physics_utils import extract_dominant_cycle, calculate_kinetic_damping
from numba import njit, prange

z = np.random.randn(60)
v = np.random.randn(20)

t0 = time.perf_counter()
for i in range(10000):
    p = extract_dominant_cycle(z, 1.0)
    d = calculate_kinetic_damping(v)
t1 = time.perf_counter()

print(f"Original time: {t1-t0:.4f}")
