"""
Quantum Risk Engine
Powered by QuantLib
Performs Monte Carlo simulations for event horizon probability
"""
import QuantLib as ql
import numpy as np
from typing import Tuple

class QuantumRiskEngine:
    """
    Monte Carlo engine for Three-Body probability estimation.
    Uses Ornstein-Uhlenbeck process to model mean-reversion dynamics.
    """

    def __init__(self):
        # Parameters
        self.dt = 1.0 / (24.0 * 60.0 * 60.0) # 1 second in days (approx, if theta is annual)
        # Actually, let's keep everything in "bar units" or "seconds" to be consistent.
        # If theta is per second, dt=1.
        self.dt_seconds = 1.0

    def calculate_probabilities(self,
                              price: float,
                              center: float,
                              sigma: float,
                              theta: float = 0.05,
                              horizon_seconds: int = 600,
                              num_paths: int = 500) -> Tuple[float, float]:
        """
        Run Monte Carlo simulation to estimate:
        1. Tunnel Probability (Revert to Center)
        2. Escape Probability (Hit Event Horizon)

        Assumes boundaries:
        - Target: Center
        - Stop: +/- 3 Sigma (Event Horizon)
        """
        if sigma <= 0:
            return 0.5, 0.0

        # Define OU Process
        # dx = theta * (mu - x) * dt + sigma * dW
        # Note: QuantLib OU process sigma is volatility of the process, NOT price distribution width.
        # Stationary std dev = sigma_proc / sqrt(2*theta).
        # We want stationary std dev to match our 'sigma' (Regression Sigma).
        # So sigma_proc = sigma * sqrt(2*theta)

        sigma_proc = sigma * np.sqrt(2 * theta)
        process = ql.OrnsteinUhlenbeckProcess(theta, sigma_proc, price, center)

        # Path Generator
        time_horizon = float(horizon_seconds)
        steps = int(horizon_seconds) # 1 step per second

        # We need a sequence generator
        rng = ql.MersenneTwisterUniformRng(42)
        # Sequence generator for Brownian motion (dimension 1)
        seq_gen = ql.GaussianRandomSequenceGenerator(ql.UniformRandomSequenceGenerator(steps, rng))
        path_gen = ql.GaussianPathGenerator(process, time_horizon, steps, seq_gen, False)

        # Boundaries
        # If price > center (L2), target is center, stop is center + 3*sigma
        # If price < center (L3), target is center, stop is center - 3*sigma

        is_l2 = price > center
        if is_l2:
            target_level = center
            stop_level = center + 3.0 * sigma
        else:
            target_level = center
            stop_level = center - 3.0 * sigma

        hits_target = 0
        hits_stop = 0

        # Simulation Loop
        # Note: In Python this loop is slow. 500 paths * 600 steps = 300k steps.
        # Might take ~0.5s - 1s.

        for i in range(num_paths):
            sample_path = path_gen.next()
            path = sample_path.value()

            # Check path for boundary crossing
            # path is a Path object, iterable

            # Optimization: check min/max of path first?
            # QuantLib Path doesn't expose min/max easily without iteration.

            hit_t = False
            hit_s = False

            for j in range(len(path)):
                p = path[j]

                if is_l2:
                    if p <= target_level:
                        hit_t = True
                        break
                    if p >= stop_level:
                        hit_s = True
                        break
                else:
                    if p >= target_level:
                        hit_t = True
                        break
                    if p <= stop_level:
                        hit_s = True
                        break

            if hit_t:
                hits_target += 1
            elif hit_s:
                hits_stop += 1

        total_finished = hits_target + hits_stop
        # Normalize (ignoring paths that didn't hit either in time horizon)
        if total_finished > 0:
            p_tunnel = hits_target / total_finished
            p_escape = hits_stop / total_finished
        else:
            # Fallback if no boundary hit
            p_tunnel = 0.5
            p_escape = 0.0 # Didn't escape

        return p_tunnel, p_escape
