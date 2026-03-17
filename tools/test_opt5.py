import numpy as np
import time
from core.physics_utils import extract_dominant_cycle, calculate_kinetic_damping
from numba import njit, prange
import math

@njit(cache=True, fastmath=True)
def calculate_kinetic_damping_numba(peaks):
    n = len(peaks)
    if n < 5: return 1.0

    sum_x = 0.0
    sum_y = 0.0
    sum_xy = 0.0
    sum_xx = 0.0

    for i in range(n):
        x = float(i)
        y = math.log(peaks[i] + 1e-5)
        sum_x += x
        sum_y += y
        sum_xy += x * y
        sum_xx += x * x

    denom = n * sum_xx - sum_x * sum_x
    if denom == 0:
        return 1.0

    slope = (n * sum_xy - sum_x * sum_y) / denom
    return abs(slope)


@njit(cache=True, fastmath=True)
def _extract_dominant_cycle_numba(z_scores, dt):
    n = len(z_scores)
    if n < 10: return 0.0

    n_half = n // 2
    max_amp2 = 0.0
    best_freq_idx = 0

    base_factor = -2.0 * math.pi / n

    for k in range(1, n_half):
        re = 0.0
        im = 0.0
        factor = base_factor * k
        for t in range(n):
            angle = factor * t
            re += z_scores[t] * math.cos(angle)
            im += z_scores[t] * math.sin(angle)

        amp2 = re*re + im*im
        if amp2 > max_amp2:
            max_amp2 = amp2
            best_freq_idx = k

    if max_amp2 == 0.0: return 0.0

    peak_freq = best_freq_idx / (n * dt)
    return 1.0 / peak_freq if peak_freq != 0 else 0.0


@njit(parallel=True, cache=True, fastmath=True)
def process_periods_dampings(z_scores, velocities, Z_SCORE_CYCLE_WINDOW, VELOCITY_DAMPING_WINDOW, dt):
    n = len(z_scores)
    periods = np.zeros(n)
    dampings = np.zeros(n)
    for i in prange(10, n):
        z_start = max(0, i - Z_SCORE_CYCLE_WINDOW)
        w_z = z_scores[z_start:i]
        periods[i] = _extract_dominant_cycle_numba(w_z, dt)

        v_start = max(0, i - VELOCITY_DAMPING_WINDOW)
        # we need np.abs(w_v) -> we can just inline it or do it in the damping function
        w_v = velocities[v_start:i]

        # calculate damping inline to avoid np.abs allocation
        n_v = len(w_v)
        if n_v < 5:
            dampings[i] = 1.0
            continue

        sum_x = 0.0
        sum_y = 0.0
        sum_xy = 0.0
        sum_xx = 0.0

        for j in range(n_v):
            x = float(j)
            y = math.log(abs(w_v[j]) + 1e-5)
            sum_x += x
            sum_y += y
            sum_xy += x * y
            sum_xx += x * x

        denom = n_v * sum_xx - sum_x * sum_x
        if denom == 0:
            dampings[i] = 1.0
        else:
            slope = (n_v * sum_xy - sum_x * sum_y) / denom
            dampings[i] = abs(slope)

    return periods, dampings

n = 10000
z = np.random.randn(n)
v = np.random.randn(n)

t0 = time.perf_counter()
periods_orig = np.zeros(n)
dampings_orig = np.zeros(n)
for i in range(10, n):
    w_z = z[max(0, i - 60):i]
    w_v = v[max(0, i - 20):i]
    periods_orig[i] = extract_dominant_cycle(w_z, dt=1.0)
    dampings_orig[i] = calculate_kinetic_damping(w_v)
t1 = time.perf_counter()

print(f"Original time: {t1-t0:.4f}")

# Warmup
periods_new, dampings_new = process_periods_dampings(z, v, 60, 20, 1.0)

t2 = time.perf_counter()
periods_new, dampings_new = process_periods_dampings(z, v, 60, 20, 1.0)
t3 = time.perf_counter()

print(f"Numba parallel time: {t3-t2:.4f}")

print(f"Diff periods: {np.max(np.abs(periods_orig - periods_new))}")
print(f"Diff dampings: {np.max(np.abs(dampings_orig - dampings_new))}")
