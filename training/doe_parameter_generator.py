"""
Design of Experiments (DOE) Parameter Generator
Generates systematic parameter combinations for walk-forward training

Philosophy:
- NOT random search (inefficient)
- NOT grid search (combinatorial explosion)
- Systematic sampling of parameter space
- Context-aware generation
"""
import numpy as np
from scipy.stats import qmc
from scipy.optimize import minimize
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass
import itertools


@dataclass
class ParameterSet:
    """Single parameter combination with metadata"""
    iteration_id: int
    day_number: int
    context_name: str
    parameters: Dict[str, Any]
    generation_method: str  # 'baseline', 'latin_hypercube', 'mutation', 'crossover', 'response_surface'


# --- DOE Configuration Constants ---
BIAS_THRESHOLD_HIGH = 0.30
BIAS_THRESHOLD_MEDIUM = 0.15
BIAS_STRENGTH_HIGH = 0.5
BIAS_STRENGTH_MEDIUM = 0.2

PARAMS_AFFECTED_BY_EARLY_TREND = {
    'take_profit_ticks', 'trailing_stop_ticks'
}
PARAMS_AFFECTED_BY_LATE_EXIT = {
    'take_profit_ticks', 'trailing_stop_ticks'
}

# Mutation Constants
INT_SKEW_FACTOR = 6
INT_MUTATION_MIN = -3
INT_MUTATION_MAX = 4

FLOAT_SKEW_FACTOR = 0.20
FLOAT_NOISE_RANGE = 0.15

# Iteration Thresholds
ITER_BASELINE_END = 10
ITER_LHS_END = 510
ITER_MUTATION_END = 800
ITER_RSM_END = 900
ITER_CROSSOVER_END = 1000


class DOEParameterGenerator:
    """
    Generates parameter combinations using multiple strategies:

    1. Baseline (iterations 0-9): Known good starting points
    2. Latin Hypercube (iterations 10-509): Systematic space filling
    3. Mutation (iterations 510-799): Variations around best params
    4. Response Surface (iterations 800-899): Quadratic optimization of PnL
    5. Crossover (iterations 900-999): Combine good parameter sets
    """

    def __init__(self, context_detector):
        self.context_detector = context_detector
        self.best_params_history = []  # Stores best params per day [(params, performance)]
        self.latest_regret_analysis = None  # Regret analysis from previous day

        # LHS Cache: Key = day_number, Value = {'samples': np.array, 'ranges': list}
        self._lhs_cache = {}

        # Define parameter ranges for exploration
        self.param_ranges = self._define_parameter_ranges()

        # Log configuration on startup (only in main process)
        import multiprocessing
        if multiprocessing.current_process().name == 'MainProcess' and not getattr(DOEParameterGenerator, '_config_printed', False):
            self._log_parameter_configuration()
            DOEParameterGenerator._config_printed = True

    def _log_parameter_configuration(self):
        """Log the parameters being optimized and their associated modules"""
        print("\n" + "="*60)
        print("DOE PARAMETER CONFIGURATION")
        print("="*60)

        # Define module mappings based on parameter prefixes or explicit lists
        module_map = {
            'Quantum Field (PID)': ['pid_'],
            'Core': [] # Catch-all
        }

        categorized = set()

        for module, prefixes in module_map.items():
            if module == 'Core': continue

            params = [p for p in self.param_ranges.keys() if any(prefix in p for prefix in prefixes)]
            if params:
                print(f"\n[{module}]")
                for p in params:
                    min_v, max_v, p_type = self.param_ranges[p]
                    print(f"  - {p:<30} Range: {min_v} to {max_v} ({p_type})")
                    categorized.add(p)

        # Print Core (remaining)
        core_params = [p for p in self.param_ranges.keys() if p not in categorized]
        if core_params:
            print(f"\n[Core Strategy]")
            for p in core_params:
                min_v, max_v, p_type = self.param_ranges[p]
                print(f"  - {p:<30} Range: {min_v} to {max_v} ({p_type})")

        print("="*60 + "\n")

    def update_regret_analysis(self, analysis: Dict):
        """Store regret analysis to guide parameter generation"""
        self.latest_regret_analysis = analysis

    def _get_mutation_bias(self, param_name: str) -> float:
        """
        Returns a bias factor (-1.0 to 1.0) based on regret analysis.
        Positive: encourage increase.
        Negative: encourage decrease.
        Zero: neutral.
        """
        if not self.latest_regret_analysis:
            return 0.0

        dist = self.latest_regret_analysis.get('patterns', {}).get('regret_distribution', {})
        total = sum(dist.values())
        if total == 0:
            return 0.0

        early_trend = dist.get('closed_too_early_trend', 0) / total
        too_late = dist.get('closed_too_late', 0) / total

        bias = 0.0

        # Mapping parameters to regret types
        # Early trend -> We want to hold longer / target higher
        if param_name in PARAMS_AFFECTED_BY_EARLY_TREND:
            if early_trend > BIAS_THRESHOLD_HIGH:
                bias += BIAS_STRENGTH_HIGH  # Strong push up
            elif early_trend > BIAS_THRESHOLD_MEDIUM:
                bias += BIAS_STRENGTH_MEDIUM

        # Too late -> We want to exit sooner
        if param_name in PARAMS_AFFECTED_BY_LATE_EXIT:
            if too_late > BIAS_THRESHOLD_HIGH:
                bias -= BIAS_STRENGTH_HIGH  # Strong push down
            elif too_late > BIAS_THRESHOLD_MEDIUM:
                bias -= BIAS_STRENGTH_MEDIUM

        # Conflict resolution (if both high, maybe neutral or slight bias based on which is higher)
        return np.clip(bias, -0.8, 0.8)

    def _define_parameter_ranges(self) -> Dict[str, tuple]:
        """
        Define min/max ranges for key parameters

        Returns dict of {param_name: (min_val, max_val, step_type)}
        step_type: 'int', 'float', 'choice'
        """
        return {
            # Trade sizing fallbacks (used when regression data unavailable)
            'stop_loss_ticks':    (10, 25, 'int'),   # fallback SL distance in ticks
            'take_profit_ticks':  (30, 60, 'int'),   # fallback TP distance in ticks
            'trailing_stop_ticks': (5, 20, 'int'),   # fallback trail distance in ticks

            # Quantum Field PID (consumed by quantum_field_engine.batch_compute_states)
            'pid_kp': (0.1, 1.0, 'float'),   # Proportional — reaction strength
            'pid_ki': (0.01, 0.2, 'float'),  # Integral — accumulated bias
            'pid_kd': (0.1, 0.5, 'float'),   # Derivative — dampening
        }

    def generate_baseline_set(self, iteration: int, day: int, context: str) -> ParameterSet:
        """
        Iterations 0-9: Known good baseline configurations

        These are hand-tuned starting points based on domain knowledge
        """
        baselines = [
            # Conservative (iteration 0)
            {'stop_loss_ticks': 15, 'take_profit_ticks': 40, 'trailing_stop_ticks': 10,
             'pid_kp': 0.5, 'pid_ki': 0.10, 'pid_kd': 0.20},
            # Aggressive (iteration 1)
            {'stop_loss_ticks': 10, 'take_profit_ticks': 50, 'trailing_stop_ticks': 7,
             'pid_kp': 0.8, 'pid_ki': 0.05, 'pid_kd': 0.10},
            # Balanced (iteration 2)
            {'stop_loss_ticks': 12, 'take_profit_ticks': 45, 'trailing_stop_ticks': 8,
             'pid_kp': 0.5, 'pid_ki': 0.10, 'pid_kd': 0.20},
            # Wide SL (iteration 3)
            {'stop_loss_ticks': 20, 'take_profit_ticks': 35, 'trailing_stop_ticks': 12,
             'pid_kp': 0.4, 'pid_ki': 0.15, 'pid_kd': 0.30},
            # Quick exit (iteration 4)
            {'stop_loss_ticks': 10, 'take_profit_ticks': 30, 'trailing_stop_ticks': 5,
             'pid_kp': 0.6, 'pid_ki': 0.05, 'pid_kd': 0.10},
            # Wide targets (iteration 5)
            {'stop_loss_ticks': 20, 'take_profit_ticks': 60, 'trailing_stop_ticks': 15,
             'pid_kp': 0.3, 'pid_ki': 0.10, 'pid_kd': 0.40},
            # Tight stops (iteration 6)
            {'stop_loss_ticks': 10, 'take_profit_ticks': 40, 'trailing_stop_ticks': 7,
             'pid_kp': 0.7, 'pid_ki': 0.05, 'pid_kd': 0.15},
            # Standard (iteration 7)
            {'stop_loss_ticks': 15, 'take_profit_ticks': 45, 'trailing_stop_ticks': 10,
             'pid_kp': 0.5, 'pid_ki': 0.10, 'pid_kd': 0.20},
            # Scalper (iteration 8)
            {'stop_loss_ticks': 10, 'take_profit_ticks': 25, 'trailing_stop_ticks': 5,
             'pid_kp': 0.9, 'pid_ki': 0.01, 'pid_kd': 0.05},
            # Swing (iteration 9)
            {'stop_loss_ticks': 25, 'take_profit_ticks': 60, 'trailing_stop_ticks': 15,
             'pid_kp': 0.2, 'pid_ki': 0.20, 'pid_kd': 0.50},
        ]

        params = baselines[iteration % len(baselines)]

        return ParameterSet(
            iteration_id=iteration,
            day_number=day,
            context_name=context,
            parameters=params,
            generation_method='baseline'
        )

    def generate_latin_hypercube_set(self, iteration: int, day: int, context: str,
                                     n_samples: int = 500) -> ParameterSet:
        """
        Iterations 10-509: Latin Hypercube Sampling

        Systematically explores parameter space with good coverage using Stratified LHS.
        """
        # Check cache for this day's batch
        if day not in self._lhs_cache:
            # Generate new batch for the day
            param_names = sorted(list(self.param_ranges.keys()))
            d = len(param_names)

            # Use SciPy's QMC LatinHypercube
            sampler = qmc.LatinHypercube(d=d, seed=day)
            sample = sampler.random(n=n_samples) # Shape (n_samples, d)

            # Scale samples to parameter ranges
            scaled_samples = []
            for i in range(n_samples):
                p_dict = {}
                for j, name in enumerate(param_names):
                    min_val, max_val, param_type = self.param_ranges[name]
                    unit_val = sample[i, j]

                    if param_type == 'int':
                        # Scale [0,1] to [min, max]
                        val = int(min_val + unit_val * (max_val - min_val + 0.999))
                        val = min(val, max_val) # Clamp
                    elif param_type == 'float':
                        val = float(min_val + unit_val * (max_val - min_val))
                    else:
                        val = min_val if unit_val < 0.5 else max_val

                    p_dict[name] = val
                scaled_samples.append(p_dict)

            self._lhs_cache[day] = scaled_samples

        # Retrieve sample
        lhs_idx = iteration - 10
        samples = self._lhs_cache[day]

        # Wrap around if index exceeds generated batch size
        params = samples[lhs_idx % len(samples)]

        return ParameterSet(
            iteration_id=iteration,
            day_number=day,
            context_name=context,
            parameters=params,
            generation_method='latin_hypercube'
        )

    def generate_mutation_set(self, iteration: int, day: int, context: str,
                              best_params: Dict[str, Any], mutation_rate: float = 0.10) -> ParameterSet:
        """
        Iterations 510-799: Mutation around best parameters

        Takes best params and mutates 10% of values
        Incorporates Regret Analysis bias to guide mutation
        """
        np.random.seed(day * 1000 + iteration)

        params = best_params.copy()

        # Identify parameters with bias
        biased_params = []
        param_names = list(self.param_ranges.keys())
        for name in param_names:
            bias = self._get_mutation_bias(name)
            if abs(bias) > 0.1:
                biased_params.append(name)

        # Mutate 10% of parameters + any strongly biased ones
        n_mutations = max(1, int(len(param_names) * mutation_rate))

        # Always include biased parameters in mutation set
        random_params = np.random.choice(param_names, n_mutations, replace=False)
        mutated_params = set(list(random_params) + biased_params)

        for param_name in mutated_params:
            if param_name not in self.param_ranges:
                continue

            min_val, max_val, param_type = self.param_ranges[param_name]
            bias = self._get_mutation_bias(param_name)

            if param_type == 'int':
                current_val = params.get(param_name, (min_val + max_val) // 2)

                # Apply bias: positive bias -> likely positive mutation
                # Skew distribution:
                # bias=0.0 -> [-3, 3] centered at 0
                # bias=0.5 -> [-1, 5] centered at 2
                # bias=-0.5 -> [-5, 1] centered at -2

                skew = int(bias * INT_SKEW_FACTOR)  # Up to +/- 4.8 shift
                base_mutation = np.random.randint(INT_MUTATION_MIN, INT_MUTATION_MAX)
                mutation = base_mutation + skew

                new_val = np.clip(current_val + mutation, min_val, max_val)
                params[param_name] = int(new_val)

            elif param_type == 'float':
                current_val = params.get(param_name, (min_val + max_val) / 2)

                # Skew distribution for floats
                # bias=0.5 -> mean shift +10%
                skew = bias * FLOAT_SKEW_FACTOR # up to +/- 20% shift
                noise = np.random.uniform(-FLOAT_NOISE_RANGE, FLOAT_NOISE_RANGE)

                mutation_pct = noise + skew
                new_val = np.clip(current_val * (1 + mutation_pct), min_val, max_val)
                params[param_name] = float(new_val)

        return ParameterSet(
            iteration_id=iteration,
            day_number=day,
            context_name=context,
            parameters=params,
            generation_method='mutation'
        )

    def generate_response_surface_set(self, iteration: int, day: int, context: str) -> ParameterSet:
        """
        Iterations 800-899: Response Surface Optimization

        Fits a quadratic model to historical performance and optimizes for next point.
        """
        # Need at least some history to fit model
        if len(self.best_params_history) < 5:
             # Fallback to mutation
            if self.best_params_history:
                best = self.best_params_history[-1][0]
            else:
                best = self.generate_baseline_set(0, day, context).parameters
            return self.generate_mutation_set(iteration, day, context, best)

        # Extract features (params) and targets (sharpe)
        # Use only numeric parameters
        numeric_params = sorted([k for k, v in self.param_ranges.items() if v[2] in ('int', 'float')])
        X = []
        y = []

        # Scaling parameters (Min-Max)
        min_vals = np.array([self.param_ranges[k][0] for k in numeric_params])
        max_vals = np.array([self.param_ranges[k][1] for k in numeric_params])
        range_vals = max_vals - min_vals
        range_vals[range_vals == 0] = 1.0 # Avoid div/0

        for params, perf in self.best_params_history:
            row = [params.get(k, 0) for k in numeric_params]
            X.append(row)
            y.append(perf)

        X = np.array(X)
        y = np.array(y)

        # Normalize X to [0, 1] for numerical stability
        X_norm = (X - min_vals) / range_vals

        # Fit Quadratic Model: y = c + w*x + x*Q*x
        # Simplified: Linear + Diagonal Quadratic terms (to reduce degrees of freedom)
        # Design matrix: [1, x1...xn, x1^2...xn^2]
        n_samples, n_features = X_norm.shape
        X_design = np.hstack([np.ones((n_samples, 1)), X_norm, X_norm**2])

        # Solve Ridge Regression: w = (X'X + alpha*I)^-1 X'y
        # Use simple L2 regularization to handle underdetermined system (few samples vs many params)
        alpha = 1e-3 # Regularization strength
        n_coeffs = X_design.shape[1]

        try:
            # Manual Ridge: (A + alpha*I) * w = b, where A = X.T @ X, b = X.T @ y
            A = X_design.T @ X_design + alpha * np.eye(n_coeffs)
            b = X_design.T @ y
            w = np.linalg.solve(A, b)
        except Exception:
            # Fallback if singular or solver fails
            best = self.best_params_history[-1][0]
            return self.generate_mutation_set(iteration, day, context, best)

        # Optimization Function (in normalized space)
        def objective(x_vec_norm):
            # Predict negative performance (minimize negative = maximize positive)
            # x_vec_norm shape (n_features,)
            x_aug = np.hstack([1, x_vec_norm, x_vec_norm**2])
            pred = np.dot(x_aug, w)
            return -pred # Minimize negative PnL/Sharpe

        # Bounds in normalized space are [0, 1]
        bounds = [(0.0, 1.0)] * n_features

        # Start search at best known point (normalized)
        best_known_idx = np.argmax(y)
        x0_norm = X_norm[best_known_idx]

        try:
            # Optimize
            res = minimize(objective, x0_norm, bounds=bounds, method='L-BFGS-B')
            x_opt_norm = res.x
        except Exception:
            x_opt_norm = x0_norm # Fallback to best known

        # Denormalize results
        x_opt = x_opt_norm * range_vals + min_vals

        # Construct result params
        opt_params = self.best_params_history[-1][0].copy() # Base on last best to keep non-numeric
        for i, name in enumerate(numeric_params):
            val = x_opt[i]
            # Snap to int if needed
            if self.param_ranges[name][2] == 'int':
                opt_params[name] = int(round(val))
            else:
                opt_params[name] = float(val)

        return ParameterSet(
            iteration_id=iteration,
            day_number=day,
            context_name=context,
            parameters=opt_params,
            generation_method='response_surface'
        )

    def generate_crossover_set(self, iteration: int, day: int, context: str,
                               parent1: Dict[str, Any], parent2: Dict[str, Any]) -> ParameterSet:
        """
        Iterations 900-999: Crossover between good parameter sets

        Combines parameters from two parent sets
        """
        np.random.seed(day * 1000 + iteration)

        params = {}

        # Combine params: 50% from parent1, 50% from parent2
        all_param_names = set(list(parent1.keys()) + list(parent2.keys()))

        for param_name in all_param_names:
            if np.random.random() < 0.5:
                params[param_name] = parent1.get(param_name, parent2.get(param_name))
            else:
                params[param_name] = parent2.get(param_name, parent1.get(param_name))

        return ParameterSet(
            iteration_id=iteration,
            day_number=day,
            context_name=context,
            parameters=params,
            generation_method='crossover'
        )

    def generate_random_set(self, iteration_id: int) -> Dict[str, Any]:
        """
        Pure random sampling from parameter ranges (Monte Carlo).
        No DOE structure, just uniform sampling.
        """
        params = {}
        for name, (lo, hi, dtype) in self._define_parameter_ranges().items():
            if dtype == 'int':
                params[name] = np.random.randint(lo, hi + 1)
            elif dtype == 'float':
                params[name] = np.random.uniform(lo, hi)

        # Add max_hold_bars (in bars, not seconds — depends on timeframe)
        # Random range [10, 200] bars
        params['max_hold_bars'] = np.random.randint(10, 201)

        return params

    def generate_parameter_set(self, iteration: int, day: int, context: str = 'CORE') -> ParameterSet:
        """
        Master generation function

        Routes to appropriate generation method based on iteration number

        Args:
            iteration: Iteration number (0-999)
            day: Day number in training sequence
            context: Active market context

        Returns:
            ParameterSet with generated parameters
        """
        if iteration < ITER_BASELINE_END:
            # Baseline sets
            return self.generate_baseline_set(iteration, day, context)

        elif iteration < ITER_LHS_END:
            # Latin Hypercube sampling
            return self.generate_latin_hypercube_set(iteration, day, context)

        elif iteration < ITER_MUTATION_END:
            # Mutation around best
            if self.best_params_history:
                # Use most recent best params
                best_params = self.best_params_history[-1][0]
            else:
                # Fall back to baseline
                best_params = self.generate_baseline_set(0, day, context).parameters

            return self.generate_mutation_set(iteration, day, context, best_params)

        elif iteration < ITER_RSM_END:
             # Response Surface Optimization
             return self.generate_response_surface_set(iteration, day, context)

        else:
            # Crossover between top performers
            if len(self.best_params_history) >= 2:
                parent1 = self.best_params_history[-1][0]
                parent2 = self.best_params_history[-2][0]
            else:
                # Not enough history, use baselines
                parent1 = self.generate_baseline_set(0, day, context).parameters
                parent2 = self.generate_baseline_set(1, day, context).parameters

            return self.generate_crossover_set(iteration, day, context, parent1, parent2)

    def update_best_params(self, params: Dict[str, Any], performance: float = 0.0):
        """
        Record best parameters from completed day

        Called after finding optimal params for a day
        """
        self.best_params_history.append((params, performance))

        # Keep only last 20 days of history
        if len(self.best_params_history) > 20:
            self.best_params_history.pop(0)

    def get_exploitation_ratio(self, day: int) -> float:
        """
        Calculate exploration vs exploitation ratio

        Early days: More exploration (60% exploit, 40% explore)
        Later days: More exploitation (90% exploit, 10% explore)

        Args:
            day: Day number in training

        Returns:
            Exploitation ratio (0.6 to 0.9)
        """
        # Start at 60%, increase to 90% over 250 days
        ratio = 0.60 + (day / 250) * 0.30
        return min(ratio, 0.90)


# Example usage and testing
if __name__ == "__main__":
    from core.context_detector import ContextDetector

    print("="*80)
    print("DOE PARAMETER GENERATOR - DEMO")
    print("="*80)

    detector = ContextDetector()
    generator = DOEParameterGenerator(detector)

    # Test different generation methods
    print("\n### BASELINE PARAMETERS (Iteration 0) ###")
    baseline = generator.generate_parameter_set(0, day=1)
    print(f"Method: {baseline.generation_method}")
    for key, val in list(baseline.parameters.items())[:5]:
        print(f"  {key}: {val}")

    print("\n### LATIN HYPERCUBE (Iteration 100) ###")
    lhs = generator.generate_parameter_set(100, day=1)
    print(f"Method: {lhs.generation_method}")
    for key, val in list(lhs.parameters.items())[:5]:
        print(f"  {key}: {val}")

    print("\n### MUTATION (Iteration 600) ###")
    # Record a "best params" first
    generator.update_best_params(baseline.parameters)
    mutation = generator.generate_parameter_set(600, day=1)
    print(f"Method: {mutation.generation_method}")
    for key, val in list(mutation.parameters.items())[:5]:
        print(f"  {key}: {val}")

    print("\n### CROSSOVER (Iteration 900) ###")
    generator.update_best_params(lhs.parameters)  # Add another "best"
    crossover = generator.generate_parameter_set(900, day=1)
    print(f"Method: {crossover.generation_method}")
    for key, val in list(crossover.parameters.items())[:5]:
        print(f"  {key}: {val}")

    print("\n### EXPLOITATION RATIO OVER TIME ###")
    for day in [1, 50, 100, 150, 200, 250, 300]:
        ratio = generator.get_exploitation_ratio(day)
        print(f"  Day {day:>3}: {ratio:.1%} exploitation")

    print("\n" + "="*80)
    print("✅ DOE PARAMETER GENERATOR READY")
    print("="*80)
