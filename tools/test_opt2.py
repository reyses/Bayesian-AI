import numpy as np
import time
from core.physics_utils import extract_dominant_cycle, calculate_kinetic_damping
from numba import njit, prange

z = np.random.randn(60)
v = np.random.randn(20)

@njit(cache=True, fastmath=True)
def calculate_kinetic_damping_numba(peaks):
    n = len(peaks)
    if n < 5: return 1.0

    # We want slope of np.polyfit(x, y, 1) where y = np.log(peaks + 1e-5)
    sum_x = 0.0
    sum_y = 0.0
    sum_xy = 0.0
    sum_xx = 0.0

    for i in range(n):
        x = float(i)
        y = np.log(peaks[i] + 1e-5)
        sum_x += x
        sum_y += y
        sum_xy += x * y
        sum_xx += x * x

    denom = n * sum_xx - sum_x * sum_x
    if denom == 0:
        return 1.0

    slope = (n * sum_xy - sum_x * sum_y) / denom
    return abs(slope)


t0 = time.perf_counter()
for i in range(10000):
    d = calculate_kinetic_damping(v)
t1 = time.perf_counter()

# Warmup
d_n = calculate_kinetic_damping_numba(np.abs(v))

t2 = time.perf_counter()
for i in range(10000):
    d_n = calculate_kinetic_damping_numba(np.abs(v))
t3 = time.perf_counter()

print(f"Original time: {t1-t0:.4f}")
print(f"Numba time: {t3-t2:.4f}")

# Assert
d_orig = calculate_kinetic_damping(v)
d_new = calculate_kinetic_damping_numba(np.abs(v))
print(f"Diff: {abs(d_orig - d_new)}")
