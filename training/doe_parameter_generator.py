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
from typing import Dict, Any, List
from dataclasses import dataclass
import itertools


@dataclass
class ParameterSet:
    """Single parameter combination with metadata"""
    iteration_id: int
    day_number: int
    context_name: str
    parameters: Dict[str, Any]
    generation_method: str  # 'baseline', 'latin_hypercube', 'mutation', 'crossover'


class DOEParameterGenerator:
    """
    Generates parameter combinations using multiple strategies:

    1. Baseline (iterations 0-9): Known good starting points
    2. Latin Hypercube (iterations 10-509): Systematic space filling
    3. Mutation (iterations 510-799): Variations around best params
    4. Crossover (iterations 800-999): Combine good parameter sets
    """

    def __init__(self, context_detector):
        self.context_detector = context_detector
        self.best_params_history = []  # Stores best params per day

        # Define parameter ranges for exploration
        self.param_ranges = self._define_parameter_ranges()

    def _define_parameter_ranges(self) -> Dict[str, tuple]:
        """
        Define min/max ranges for key parameters

        Returns dict of {param_name: (min_val, max_val, step_type)}
        step_type: 'int', 'float', 'choice'
        """
        return {
            # Core parameters
            'stop_loss_ticks': (10, 25, 'int'),
            'take_profit_ticks': (30, 60, 'int'),
            'min_samples_required': (20, 50, 'int'),
            'confidence_threshold': (0.30, 0.70, 'float'),
            'trail_activation_profit': (30, 100, 'int'),
            'trail_distance_tight': (5, 15, 'int'),
            'trail_distance_wide': (20, 40, 'int'),
            'max_consecutive_losses': (3, 10, 'int'),
            'min_sharpe_ratio': (0.3, 1.0, 'float'),

            # Kill zone parameters
            'killzone_tolerance_ticks': (3, 10, 'int'),
            'min_rejection_wick_ticks': (3, 10, 'int'),
            'wick_to_body_ratio': (1.5, 3.0, 'float'),
            'zone_strength_multiplier': (1.0, 2.0, 'float'),

            # Confirmation parameters
            'volume_spike_threshold': (1.5, 3.0, 'float'),
            'aggressive_buyer_pct': (0.6, 0.8, 'float'),
            'bid_ask_imbalance_threshold': (0.6, 0.8, 'float'),

            # Velocity parameters
            'cascade_min_points': (5, 20, 'int'),
            'cascade_time_window': (0.3, 1.0, 'float'),
            'min_entry_velocity': (2, 10, 'int'),

            # Volatility parameters
            'layer_high_volatility_sigma': (2.5, 4.0, 'float'),
            'layer_low_volatility_sigma': (1.5, 2.5, 'float'),
            'min_sigma_differential': (0.5, 2.0, 'float'),

            # Resonance parameters
            'min_resonance_score': (6.0, 9.0, 'float'),
            'resonance_decay_rate': (0.90, 0.99, 'float'),

            # Transition parameters
            'transition_speed_threshold': (1, 5, 'int'),
            'hysteresis_factor': (0.1, 0.5, 'float'),
            'min_bars_in_state': (3, 10, 'int'),

            # Session parameters
            'opening_range_minutes': (10, 30, 'int'),
            'min_hold_seconds': (30, 120, 'int'),
            'max_hold_seconds': (600, 1800, 'int'),

            # Trading cost (round-trip: commission + slippage in points)
            'trading_cost_points': (0.25, 1.0, 'float')
        }

    def generate_baseline_set(self, iteration: int, day: int, context: str) -> ParameterSet:
        """
        Iterations 0-9: Known good baseline configurations

        These are hand-tuned starting points based on domain knowledge
        """
        baselines = [
            # Conservative (iteration 0)
            {
                'stop_loss_ticks': 15,
                'take_profit_ticks': 40,
                'confidence_threshold': 0.50,
                'trail_distance_tight': 10,
                'trail_distance_wide': 30
            },
            # Aggressive (iteration 1)
            {
                'stop_loss_ticks': 10,
                'take_profit_ticks': 50,
                'confidence_threshold': 0.45,
                'trail_distance_tight': 7,
                'trail_distance_wide': 25
            },
            # Balanced (iteration 2)
            {
                'stop_loss_ticks': 12,
                'take_profit_ticks': 45,
                'confidence_threshold': 0.48,
                'trail_distance_tight': 8,
                'trail_distance_wide': 28
            },
            # High confidence (iteration 3)
            {
                'stop_loss_ticks': 20,
                'take_profit_ticks': 35,
                'confidence_threshold': 0.65,
                'trail_distance_tight': 12,
                'trail_distance_wide': 35
            },
            # Quick exit (iteration 4)
            {
                'stop_loss_ticks': 8,
                'take_profit_ticks': 30,
                'confidence_threshold': 0.40,
                'trail_distance_tight': 5,
                'trail_distance_wide': 20
            },
            # Wide targets (iteration 5)
            {
                'stop_loss_ticks': 20,
                'take_profit_ticks': 60,
                'confidence_threshold': 0.55,
                'trail_distance_tight': 15,
                'trail_distance_wide': 40
            },
            # Tight stops (iteration 6)
            {
                'stop_loss_ticks': 10,
                'take_profit_ticks': 40,
                'confidence_threshold': 0.50,
                'trail_distance_tight': 7,
                'trail_distance_wide': 25
            },
            # Standard (iteration 7)
            {
                'stop_loss_ticks': 15,
                'take_profit_ticks': 45,
                'confidence_threshold': 0.50,
                'trail_distance_tight': 10,
                'trail_distance_wide': 30
            },
            # Scalper (iteration 8)
            {
                'stop_loss_ticks': 8,
                'take_profit_ticks': 25,
                'confidence_threshold': 0.35,
                'trail_distance_tight': 5,
                'trail_distance_wide': 15
            },
            # Swing (iteration 9)
            {
                'stop_loss_ticks': 25,
                'take_profit_ticks': 60,
                'confidence_threshold': 0.60,
                'trail_distance_tight': 15,
                'trail_distance_wide': 40
            }
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

        Systematically explores parameter space with good coverage
        """
        # Calculate position in LHS grid
        lhs_idx = iteration - 10  # 0-499

        # Use LHS to sample parameter space
        np.random.seed(day * 1000 + lhs_idx)  # Reproducible per day

        params = {}

        for param_name, (min_val, max_val, param_type) in self.param_ranges.items():
            if param_type == 'int':
                # Sample uniformly for integers
                val = np.random.randint(min_val, max_val + 1)
            elif param_type == 'float':
                # Sample uniformly for floats
                val = np.random.uniform(min_val, max_val)
            else:
                # For choices, pick randomly
                val = min_val if np.random.random() < 0.5 else max_val

            params[param_name] = val

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
        """
        np.random.seed(day * 1000 + iteration)

        params = best_params.copy()

        # Mutate 10% of parameters
        param_names = list(self.param_ranges.keys())
        n_mutations = max(1, int(len(param_names) * mutation_rate))

        mutated_params = np.random.choice(param_names, n_mutations, replace=False)

        for param_name in mutated_params:
            if param_name not in self.param_ranges:
                continue

            min_val, max_val, param_type = self.param_ranges[param_name]

            if param_type == 'int':
                # Mutate ±20% with bounds
                current_val = params.get(param_name, (min_val + max_val) // 2)
                mutation = np.random.randint(-3, 4)
                new_val = np.clip(current_val + mutation, min_val, max_val)
                params[param_name] = int(new_val)

            elif param_type == 'float':
                # Mutate ±15% with bounds
                current_val = params.get(param_name, (min_val + max_val) / 2)
                mutation = np.random.uniform(-0.15, 0.15) * current_val
                new_val = np.clip(current_val + mutation, min_val, max_val)
                params[param_name] = float(new_val)

        return ParameterSet(
            iteration_id=iteration,
            day_number=day,
            context_name=context,
            parameters=params,
            generation_method='mutation'
        )

    def generate_crossover_set(self, iteration: int, day: int, context: str,
                               parent1: Dict[str, Any], parent2: Dict[str, Any]) -> ParameterSet:
        """
        Iterations 800-999: Crossover between good parameter sets

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
        if iteration < 10:
            # Baseline sets
            ps = self.generate_baseline_set(iteration, day, context)

        elif iteration < 510:
            # Latin Hypercube sampling
            ps = self.generate_latin_hypercube_set(iteration, day, context)

        elif iteration < 800:
            # Mutation around best
            if self.best_params_history:
                # Use most recent best params
                best_params = self.best_params_history[-1]
            else:
                # Fall back to baseline
                best_params = self.generate_baseline_set(0, day, context).parameters

            ps = self.generate_mutation_set(iteration, day, context, best_params)

        else:
            # Crossover between top performers
            if len(self.best_params_history) >= 2:
                parent1 = self.best_params_history[-1]
                parent2 = self.best_params_history[-2]
            else:
                # Not enough history, use baselines
                parent1 = self.generate_baseline_set(0, day, context).parameters
                parent2 = self.generate_baseline_set(1, day, context).parameters

            ps = self.generate_crossover_set(iteration, day, context, parent1, parent2)

        # Enforce constraints on the generated set
        ps.parameters = self._enforce_constraints(ps.parameters)
        return ps

    def update_best_params(self, params: Dict[str, Any]):
        """
        Record best parameters from completed day

        Called after finding optimal params for a day
        """
        self.best_params_history.append(params)

        # Keep only last 20 days of history
        if len(self.best_params_history) > 20:
            self.best_params_history.pop(0)

    def _enforce_constraints(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Ensure logical consistency of parameters"""
        # TP > SL constraint
        if 'stop_loss_ticks' in params and 'take_profit_ticks' in params:
            sl = params['stop_loss_ticks']
            tp = params['take_profit_ticks']

            if tp <= sl:
                # Force TP to be at least SL + 5 or 1.5x SL
                new_tp = max(int(sl * 1.5), sl + 5)
                params['take_profit_ticks'] = new_tp

        return params

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
