"""
CUDA-Accelerated Physics Engine
Implements fused kernels for QuantumFieldEngine.
Replacing CPU-bound physics calculations with parallel GPU compute.
"""
import math
import numpy as np
import numba
from numba import cuda

# Physics Constants
reg_period = 21
SIGMA_ROCHE = 2.0
SIGMA_EVENT = 3.0
GRAVITY_THETA = 0.5
PID_KP = 0.5
PID_KI = 0.1
PID_KD = 0.2

# Repulsion Constants
REPULSION_EPSILON = 0.01
REPULSION_FORCE_CAP = 100.0

# Archetype Thresholds
VELOCITY_THRESHOLD = 0.5
MOMENTUM_THRESHOLD = 5.0
COHERENCE_THRESHOLD = 0.3

# Indicator Constants
ADX_PERIOD = 14
HURST_WINDOW = 100
HURST_MIN_WINDOW = 30  # Minimum bars before computing Hurst

@cuda.jit
def compute_physics_kernel(prices, volumes,
                           out_center, out_sigma, out_slope,
                           out_z, out_velocity, out_force, out_momentum,
                           out_coherence, out_entropy,
                           out_prob0, out_prob1, out_prob2,
                           reg_period, mean_x, inv_reg_period, inv_denom, denom):
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

            mean_y = sum_y * inv_reg_period

            # Slope calculation using precomputed constants
            slope = (sum_xy - mean_x * sum_y) * inv_denom
            intercept = mean_y - slope * mean_x

            # Center at end of window (x = reg_period - 1)
            # center = slope * (reg_period - 1) + intercept
            # Simplified: center = slope * (reg_period - 1) + (mean_y - slope * mean_x)
            #            = mean_y + slope * (reg_period - 1 - mean_x)
            center = mean_y + slope * ((reg_period - 1) - mean_x)
            out_center[i] = center
            out_slope[i] = slope

            # Compute Sigma (Standard Deviation of Residuals)
            # RSS = SST - slope^2 * DENOM
            # SST = sum_yy - n * mean_y^2
            sst = sum_yy - reg_period * mean_y * mean_y
            rss = sst - slope * slope * denom

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

@cuda.jit
def compute_dm_tr_kernel(highs, lows, closes,
                          out_tr, out_plus_dm, out_minus_dm):
    """
    Pass 1: Compute raw True Range and Directional Movement per bar.
    Fully parallel — one thread per bar.
    """
    i = cuda.grid(1)
    n = highs.shape[0]

    if i < n:
        if i == 0:
            out_tr[i] = highs[i] - lows[i]
            out_plus_dm[i] = 0.0
            out_minus_dm[i] = 0.0
        else:
            # True Range = max(H-L, |H-prevC|, |L-prevC|)
            hl = highs[i] - lows[i]
            hc = abs(highs[i] - closes[i-1])
            lc = abs(lows[i] - closes[i-1])
            out_tr[i] = max(hl, max(hc, lc))

            # +DM and -DM
            up_move = highs[i] - highs[i-1]
            down_move = lows[i-1] - lows[i]

            if up_move > down_move and up_move > 0:
                out_plus_dm[i] = up_move
            else:
                out_plus_dm[i] = 0.0

            if down_move > up_move and down_move > 0:
                out_minus_dm[i] = down_move
            else:
                out_minus_dm[i] = 0.0

def compute_adx_dmi_cpu(tr_raw, plus_dm_raw, minus_dm_raw, period=14):
    """
    Pass 2: Wilder's smoothed ADX/DMI computation.
    Sequential but fast (single pass over arrays).

    Returns: (adx, dmi_plus, dmi_minus) — all numpy arrays length n
    """
    n = len(tr_raw)
    adx = np.zeros(n)
    dmi_plus = np.zeros(n)
    dmi_minus = np.zeros(n)

    if n < period + 1:
        return adx, dmi_plus, dmi_minus

    # Initial sums (first `period` bars)
    smooth_tr = np.sum(tr_raw[1:period+1])
    smooth_plus = np.sum(plus_dm_raw[1:period+1])
    smooth_minus = np.sum(minus_dm_raw[1:period+1])

    # First DI values
    if smooth_tr > 0:
        dmi_plus[period] = 100.0 * smooth_plus / smooth_tr
        dmi_minus[period] = 100.0 * smooth_minus / smooth_tr

    # First DX
    di_sum = dmi_plus[period] + dmi_minus[period]
    if di_sum > 0:
        dx_first = 100.0 * abs(dmi_plus[period] - dmi_minus[period]) / di_sum
    else:
        dx_first = 0.0

    # Wilder smoothing for remaining bars
    dx_sum = dx_first
    dx_count = 1

    for i in range(period + 1, n):
        # Wilder smoothing: smooth = prev_smooth - (prev_smooth / period) + current
        smooth_tr = smooth_tr - (smooth_tr / period) + tr_raw[i]
        smooth_plus = smooth_plus - (smooth_plus / period) + plus_dm_raw[i]
        smooth_minus = smooth_minus - (smooth_minus / period) + minus_dm_raw[i]

        if smooth_tr > 0:
            dmi_plus[i] = 100.0 * smooth_plus / smooth_tr
            dmi_minus[i] = 100.0 * smooth_minus / smooth_tr

        di_sum = dmi_plus[i] + dmi_minus[i]
        if di_sum > 0:
            dx = 100.0 * abs(dmi_plus[i] - dmi_minus[i]) / di_sum
        else:
            dx = 0.0

        # ADX = Wilder smoothed DX
        if dx_count < period:
            dx_sum += dx
            dx_count += 1
            if dx_count == period:
                adx[i] = dx_sum / period
        else:
            adx[i] = (adx[i-1] * (period - 1) + dx) / period

    return adx, dmi_plus, dmi_minus

@cuda.jit
def compute_hurst_kernel(prices, out_hurst, window_size):
    """
    Rescaled Range (R/S) Hurst exponent per bar.
    Uses 4 sub-window sizes: window/8, window/4, window/2, window.
    Linear regression of log(R/S) vs log(n) gives Hurst.
    """
    i = cuda.grid(1)
    n = prices.shape[0]

    if i < n:
        out_hurst[i] = 0.5  # Default: Brownian

        if i < window_size:
            return

        # Sub-window sizes for R/S regression
        # We'll use 4 sizes: w/8, w/4, w/2, w
        sizes = cuda.local.array(4, dtype=numba.int32)
        sizes[0] = max(window_size // 8, 4)
        sizes[1] = max(window_size // 4, 8)
        sizes[2] = max(window_size // 2, 16)
        sizes[3] = window_size

        log_n = cuda.local.array(4, dtype=numba.float64)
        log_rs = cuda.local.array(4, dtype=numba.float64)

        for s_idx in range(4):
            sz = sizes[s_idx]
            start = i - sz + 1

            # Compute returns within sub-window
            mean_ret = 0.0
            for k in range(start + 1, i + 1):
                mean_ret += (prices[k] - prices[k-1])
            mean_ret /= (sz - 1)

            # Cumulative deviation from mean
            cum_dev = 0.0
            max_dev = -1e30
            min_dev = 1e30
            std_sum = 0.0

            for k in range(start + 1, i + 1):
                ret = (prices[k] - prices[k-1]) - mean_ret
                cum_dev += ret
                if cum_dev > max_dev:
                    max_dev = cum_dev
                if cum_dev < min_dev:
                    min_dev = cum_dev
                std_sum += ret * ret

            R = max_dev - min_dev
            S = math.sqrt(std_sum / (sz - 1)) if sz > 1 else 1e-10
            S = max(S, 1e-10)

            rs = R / S
            log_n[s_idx] = math.log(float(sz))
            log_rs[s_idx] = math.log(max(rs, 1e-10))

        # Linear regression: log(R/S) = H * log(n) + c
        # Hurst = slope
        sum_x = 0.0
        sum_y = 0.0
        sum_xy = 0.0
        sum_xx = 0.0
        for j in range(4):
            sum_x += log_n[j]
            sum_y += log_rs[j]
            sum_xy += log_n[j] * log_rs[j]
            sum_xx += log_n[j] * log_n[j]

        denom = 4.0 * sum_xx - sum_x * sum_x
        if abs(denom) > 1e-12:
            hurst = (4.0 * sum_xy - sum_x * sum_y) / denom
        else:
            hurst = 0.5

        # Clamp to [0, 1]
        out_hurst[i] = max(0.0, min(1.0, hurst))
