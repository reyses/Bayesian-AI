import numpy as np
import time
from core.physics_utils import extract_dominant_cycle
from scipy.fft import fft, fftfreq

# we want to numba-ize extract_dominant_cycle too.
# it uses scipy.fft.fft. Scipy is not numba compatible.
# Can we use np.fft.fft?
from numba import njit, prange

z = np.random.randn(60)

t0 = time.perf_counter()
for i in range(100000):
    p = extract_dominant_cycle(z, 1.0)
t1 = time.perf_counter()

print(f"Original time: {t1-t0:.4f}")

@njit(cache=True, fastmath=True)
def _extract_dominant_cycle_numba(z_scores, dt):
    n = len(z_scores)
    if n < 10: return 0.0

    n_half = n // 2
    max_amp2 = 0.0
    best_freq_idx = 0

    # Pre-calculate factor
    base_factor = -2.0 * np.pi / n

    for k in range(1, n_half):
        re = 0.0
        im = 0.0
        factor = base_factor * k
        for t in range(n):
            angle = factor * t
            # Using fastmath will automatically inline cos/sin efficiently
            re += z_scores[t] * np.cos(angle)
            im += z_scores[t] * np.sin(angle)

        amp2 = re*re + im*im
        if amp2 > max_amp2:
            max_amp2 = amp2
            best_freq_idx = k

    if max_amp2 == 0.0: return 0.0

    peak_freq = best_freq_idx / (n * dt)
    return 1.0 / peak_freq if peak_freq != 0 else 0.0


# Warmup
p_n = _extract_dominant_cycle_numba(z, 1.0)

t2 = time.perf_counter()
for i in range(100000):
    p_n = _extract_dominant_cycle_numba(z, 1.0)
t3 = time.perf_counter()

print(f"Numba time: {t3-t2:.4f}")

# Assert
p_orig = extract_dominant_cycle(z, 1.0)
p_new = _extract_dominant_cycle_numba(z, 1.0)
print(f"Original: {p_orig}, New: {p_new}")
print(f"Diff: {abs(p_orig - p_new)}")
