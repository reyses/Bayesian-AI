"""
Design of Experiments (DOE) Parameter Generator
Generates systematic parameter combinations for walk-forward training.

Refactored to use Optuna for PID optimization (Bayesian TPE).
"""
import numpy as np
from typing import Dict, Any, List, Tuple, Optional

# --- DOE Configuration Constants ---
# Kept for backward compatibility if imported, though mostly unused now
PARAMS_AFFECTED_BY_EARLY_TREND = {'pid_kp', 'pid_kd'}
PARAMS_AFFECTED_BY_LATE_EXIT = {'pid_ki'}

class DOEParameterGenerator:
    """
    Generates parameter combinations using Optuna TPE (Tree-structured Parzen Estimator).
    Focuses purely on PID optimization (kp, ki, kd).
    """

    def __init__(self, context_detector):
        self.context_detector = context_detector
        self.param_ranges = self._define_parameter_ranges()

        # Log configuration on startup (only in main process)
        import multiprocessing
        if multiprocessing.current_process().name == 'MainProcess' and not getattr(DOEParameterGenerator, '_config_printed', False):
            self._log_parameter_configuration()
            DOEParameterGenerator._config_printed = True

    def _define_parameter_ranges(self) -> Dict[str, tuple]:
        """
        Define min/max ranges for key parameters.
        Only PID parameters are optimized via search; exits are analytical.

        Returns dict of {param_name: (min_val, max_val, step_type)}
        step_type: 'int', 'float', 'choice'
        """
        return {
            # Quantum Field PID (consumed by quantum_field_engine.batch_compute_states)
            'pid_kp': (0.1, 1.0, 'float'),   # Proportional — reaction strength
            'pid_ki': (0.01, 0.2, 'float'),  # Integral — accumulated bias
            'pid_kd': (0.1, 0.5, 'float'),   # Derivative — dampening
        }

    def _log_parameter_configuration(self):
        """Log the parameters being optimized and their associated modules"""
        print("\n" + "="*60)
        print("DOE PARAMETER CONFIGURATION (OPTUNA PID)")
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

    def optimize_pid(
        self,
        objective_fn,          # callable(pid_kp, pid_ki, pid_kd) -> float (Sharpe)
        n_trials: int = 200,   # number of Optuna trials
        seed: int = 42,
    ) -> dict:
        """
        Run Optuna TPE to find pid_kp, pid_ki, pid_kd that maximize Sharpe
        across all cluster members.

        Returns: {'pid_kp': float, 'pid_ki': float, 'pid_kd': float}
        """
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)  # suppress per-trial output

        def _optuna_objective(trial):
            pid_kp = trial.suggest_float('pid_kp', 0.1, 1.0)
            pid_ki = trial.suggest_float('pid_ki', 0.01, 0.2)
            pid_kd = trial.suggest_float('pid_kd', 0.1, 0.5)
            return objective_fn(pid_kp, pid_ki, pid_kd)

        sampler = optuna.samplers.TPESampler(seed=seed)
        study = optuna.create_study(direction='maximize', sampler=sampler)
        study.optimize(_optuna_objective, n_trials=n_trials, show_progress_bar=False)

        best = study.best_params
        return {
            'pid_kp': best['pid_kp'],
            'pid_ki': best['pid_ki'],
            'pid_kd': best['pid_kd'],
        }
