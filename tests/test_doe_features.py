import unittest
import numpy as np
import sys
import os
from unittest.mock import MagicMock

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from training.doe_parameter_generator import DOEParameterGenerator

class TestDOEFeatures(unittest.TestCase):
    def setUp(self):
        self.detector = MagicMock()
        self.generator = DOEParameterGenerator(self.detector)

    def test_optimize_pid(self):
        """Verify optimize_pid finds parameters maximizing the objective function"""
        print("\nTesting optimize_pid...")

        # Objective: Maximize -(kp - 0.5)^2 - (ki - 0.1)^2 - (kd - 0.3)^2
        # Peak at kp=0.5, ki=0.1, kd=0.3
        target_kp = 0.5
        target_ki = 0.1
        target_kd = 0.3

        def objective_fn(kp, ki, kd):
            score = -((kp - target_kp)**2 + (ki - target_ki)**2 + (kd - target_kd)**2)
            return score

        # Run optimization with small number of trials for speed
        best_params = self.generator.optimize_pid(objective_fn, n_trials=50, seed=42)

        print(f"Best Params Found: {best_params}")

        # Check convergence (allow some tolerance as it's stochastic and limited trials)
        self.assertAlmostEqual(best_params['pid_kp'], target_kp, delta=0.2)
        self.assertAlmostEqual(best_params['pid_ki'], target_ki, delta=0.05)
        self.assertAlmostEqual(best_params['pid_kd'], target_kd, delta=0.1)

    def test_define_parameter_ranges(self):
        """Verify parameter ranges are defined correctly"""
        ranges = self.generator._define_parameter_ranges()
        self.assertIn('pid_kp', ranges)
        self.assertIn('pid_ki', ranges)
        self.assertIn('pid_kd', ranges)

        min_v, max_v, _ = ranges['pid_kp']
        self.assertTrue(min_v < max_v)

if __name__ == '__main__':
    unittest.main()
