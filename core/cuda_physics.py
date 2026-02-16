"""
CUDA-Accelerated Physics Engine
Implements fused kernels for QuantumFieldEngine.
Replacing CPU-bound physics calculations with parallel GPU compute.
"""
import math
import numpy as np
from numba import cuda

# Physics Constants
reg_period = 21
SIGMA_ROCHE = 2.0
SIGMA_EVENT = 3.0
GRAVITY_THETA = 0.5
PID_KP = 0.5
PID_KI = 0.1
PID_KD = 0.2

# Precomputed Regression Constants
_SUM_X = 0.0
_SUM_XX = 0.0
for _k in range(reg_period):
    _SUM_X += float(_k)
    _SUM_XX += float(_k * _k)

_MEAN_X = _SUM_X / reg_period
_DENOM = _SUM_XX - (_SUM_X * _SUM_X) / reg_period
_INV_REG_PERIOD = 1.0 / reg_period
_INV_DENOM = 0.0
if abs(_DENOM) > 1e-9:
    _INV_DENOM = 1.0 / _DENOM

# Repulsion Constants
REPULSION_EPSILON = 0.01
REPULSION_FORCE_CAP = 100.0

# Archetype Thresholds
VELOCITY_THRESHOLD = 0.5
MOMENTUM_THRESHOLD = 5.0
COHERENCE_THRESHOLD = 0.3

@cuda.jit
def compute_physics_kernel(prices, volumes,
                           out_center, out_sigma, out_slope,
                           out_z, out_velocity, out_force, out_momentum,
                           out_coherence, out_entropy,
                           out_prob0, out_prob1, out_prob2):
    """
    Fused Physics Kernel:
    1. Rolling Linear Regression (Center, Sigma, Slope)
    2. Z-Score & Volatility
    3. Tidal Forces (Gravity, Momentum)
    4. Wave Function (Probabilities, Entropy, Coherence)
    """
    i = cuda.grid(1)
    n = prices.shape[0]

    if i < n:
        # Default initialization
        out_z[i] = 0.0
        out_velocity[i] = 0.0
        out_force[i] = 0.0
        out_momentum[i] = 0.0
        out_coherence[i] = 1.0
        out_entropy[i] = 0.0
        out_prob0[i] = 1.0
        out_prob1[i] = 0.0
        out_prob2[i] = 0.0

        # Need full regression window
        if i >= reg_period - 1:
            # 1. Rolling Linear Regression (Optimized One Pass)
            sum_y = 0.0
            sum_xy = 0.0
            sum_yy = 0.0

            for k in range(reg_period):
                idx = i - (reg_period - 1) + k
                val = prices[idx]
                x = float(k)

                sum_y += val
                sum_xy += x * val
                sum_yy += val * val

            mean_y = sum_y * _INV_REG_PERIOD

            # Slope calculation using precomputed constants
            slope = (sum_xy - _MEAN_X * sum_y) * _INV_DENOM
            intercept = mean_y - slope * _MEAN_X

            # Center at end of window (x = reg_period - 1)
            # center = slope * (reg_period - 1) + intercept
            # Simplified: center = slope * (reg_period - 1) + (mean_y - slope * mean_x)
            #            = mean_y + slope * (reg_period - 1 - mean_x)
            center = mean_y + slope * ((reg_period - 1) - _MEAN_X)
            out_center[i] = center
            out_slope[i] = slope

            # Compute Sigma (Standard Deviation of Residuals)
            # RSS = SST - slope^2 * DENOM
            # SST = sum_yy - n * mean_y^2
            sst = sum_yy - reg_period * mean_y * mean_y
            rss = sst - slope * slope * _DENOM

            # Clamp for numerical stability
            if rss < 0.0:
                rss = 0.0

            if reg_period > 2:
                sigma = math.sqrt(rss / (reg_period - 2))
            else:
                sigma = 0.0

            if sigma < 1e-6:
                sigma = 1e-6

            out_sigma[i] = sigma

            # 2. Z-Score
            z = (prices[i] - center) / sigma
            out_z[i] = z

            # 3. Velocity & Momentum
            if i > 0:
                velocity = prices[i] - prices[i-1]
            else:
                velocity = 0.0

            out_velocity[i] = velocity

            momentum = (velocity * volumes[i]) / sigma
            out_momentum[i] = momentum

            # Forces (Gravity)
            F_gravity = -GRAVITY_THETA * (z * sigma)

            # Repulsion
            upper_sing = center + SIGMA_ROCHE * sigma
            lower_sing = center - SIGMA_ROCHE * sigma

            dist_upper = abs(prices[i] - upper_sing) / sigma
            dist_lower = abs(prices[i] - lower_sing) / sigma

            F_upper = 0.0
            if z > 0:
                F_upper = 1.0 / (dist_upper**3 + REPULSION_EPSILON)
                if F_upper > REPULSION_FORCE_CAP: F_upper = REPULSION_FORCE_CAP

            F_lower = 0.0
            if z < 0:
                F_lower = 1.0 / (dist_lower**3 + REPULSION_EPSILON)
                if F_lower > REPULSION_FORCE_CAP: F_lower = REPULSION_FORCE_CAP

            repulsion = -F_upper if z > 0 else F_lower

            F_net = F_gravity + momentum + repulsion
            out_force[i] = F_net

            # 4. Wave Function
            E0 = - (z * z) / 2.0
            E1 = - (z - 2.0)**2 / 2.0
            E2 = - (z + 2.0)**2 / 2.0

            max_E = E0
            if E1 > max_E: max_E = E1
            if E2 > max_E: max_E = E2

            p0 = math.exp(E0 - max_E)
            p1 = math.exp(E1 - max_E)
            p2 = math.exp(E2 - max_E)

            total_p = p0 + p1 + p2
            p0 /= total_p
            p1 /= total_p
            p2 /= total_p

            out_prob0[i] = p0
            out_prob1[i] = p1
            out_prob2[i] = p2

            eps = 1e-10
            entropy = - (p0 * math.log(p0 + eps) +
                         p1 * math.log(p1 + eps) +
                         p2 * math.log(p2 + eps))

            out_entropy[i] = entropy
            out_coherence[i] = entropy / 1.09861228867 # ln(3)

@cuda.jit
def detect_archetype_kernel(z_scores, velocities, momentums, coherences,
                            out_roche_snap, out_structural_drive):
    """
    Detects Physics Archetypes based on computed fields.
    """
    i = cuda.grid(1)
    n = z_scores.shape[0]

    if i < n:
        # Roche Snap
        is_roche = abs(z_scores[i]) > 2.0 and abs(velocities[i]) > VELOCITY_THRESHOLD
        out_roche_snap[i] = is_roche

        # Structural Drive
        is_drive = abs(momentums[i]) > MOMENTUM_THRESHOLD and coherences[i] < COHERENCE_THRESHOLD
        out_structural_drive[i] = is_drive
