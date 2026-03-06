"""
Quantum Risk Engine
Powered by Numpy Vectorization (formerly QuantLib)
Performs Monte Carlo simulations for event horizon probability
"""
import numpy as np
from typing import Tuple

class QuantumRiskEngine:
    """
    Monte Carlo engine for Three-Body probability estimation.
    Uses Ornstein-Uhlenbeck process to model mean-reversion dynamics.
    """

    def __init__(self, theta: float = 0.05, horizon_seconds: int = 600, num_paths: int = 500):
        self.theta = theta
        self.horizon_seconds = horizon_seconds
        self.num_paths = num_paths

    def calculate_probabilities(self,
                              price: float,
                              center: float,
                              sigma: float) -> Tuple[float, float]:
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

        dt = 1.0
        steps = int(self.horizon_seconds)

        # OU Parameters for Exact Solution
        # X_{t+1} = X_t * exp(-theta*dt) + center * (1 - exp(-theta*dt)) + sigma_step * epsilon
        # Stationary variance = sigma^2
        # Variance over dt = sigma^2 * (1 - exp(-2*theta*dt))

        decay = np.exp(-self.theta * dt)
        mu_term = center * (1 - decay)
        sigma_step = sigma * np.sqrt(1 - np.exp(-2 * self.theta * dt))

        # Current paths state
        current_prices = np.full(self.num_paths, price)

        # Status of paths: 0 = running, 1 = hit target, 2 = hit stop
        status = np.zeros(self.num_paths, dtype=int)

        is_l2 = price > center
        if is_l2:
            target_level = center
            stop_level = center + 3.0 * sigma
        else:
            target_level = center
            stop_level = center - 3.0 * sigma

        # Check initial condition
        # If starting price is already crossing boundaries
        if is_l2:
            if price <= target_level: # Impossible given is_l2 definition unless ==
                status[:] = 1
            elif price >= stop_level:
                status[:] = 2
        else:
            if price >= target_level: # Impossible given is_l2 definition unless ==
                status[:] = 1
            elif price <= stop_level:
                status[:] = 2

        # If already finished, return immediately
        if np.all(status != 0):
            hits_target = np.sum(status == 1)
            hits_stop = np.sum(status == 2)
            total = hits_target + hits_stop
            return hits_target / total, hits_stop / total

        # Vectorized Simulation
        rng = np.random.default_rng(42)
        active_mask = status == 0

        for t in range(steps):
            if not np.any(active_mask):
                break

            # Update only active paths
            n_active = np.sum(active_mask)
            epsilon = rng.standard_normal(n_active)

            # Vectorized update
            current_prices[active_mask] = current_prices[active_mask] * decay + mu_term + sigma_step * epsilon

            # Check boundaries
            active_prices = current_prices[active_mask]

            if is_l2:
                # target: <= center
                hit_t_mask = active_prices <= target_level
                # stop: >= center + 3sigma
                hit_s_mask = active_prices >= stop_level
            else:
                # target: >= center
                hit_t_mask = active_prices >= target_level
                # stop: <= center - 3sigma
                hit_s_mask = active_prices <= stop_level

            combined_hit = hit_t_mask | hit_s_mask

            if np.any(combined_hit):
                # indices relative to active set
                # We need global indices to update status and active_mask
                # active_mask is a boolean array of shape (num_paths,)
                # just_finished_indices are indices in the *compressed* array of active paths?
                # No, np.where(active_mask)[0] gives global indices of active paths.

                active_global_indices = np.where(active_mask)[0]
                just_finished_global_indices = active_global_indices[combined_hit]

                # Identify type
                t_hits = hit_t_mask[combined_hit]

                # Update status
                # status[idx] = 1 where t_hits is True
                status[just_finished_global_indices[t_hits]] = 1
                status[just_finished_global_indices[~t_hits]] = 2

                # Update active_mask
                active_mask[just_finished_global_indices] = False

        hits_target = np.sum(status == 1)
        hits_stop = np.sum(status == 2)

        total_finished = hits_target + hits_stop
        if total_finished > 0:
            p_tunnel = hits_target / total_finished
            p_escape = hits_stop / total_finished
        else:
            # Fallback if no boundary hit
            p_tunnel = 0.5
            p_escape = 0.0 # Didn't escape

        return p_tunnel, p_escape
