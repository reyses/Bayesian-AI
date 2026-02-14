
import unittest
import numpy as np
from unittest.mock import MagicMock
from training.doe_parameter_generator import DOEParameterGenerator

class TestDOERegret(unittest.TestCase):
    def setUp(self):
        self.detector = MagicMock()
        self.generator = DOEParameterGenerator(self.detector)

        self.best_params = {
            'stop_loss_ticks': 15,
            'take_profit_ticks': 40,
            'trail_distance_wide': 20,
            'max_hold_seconds': 600,
            'confidence_threshold': 0.5,
            'trail_activation_profit': 30
        }

    def test_no_regret_analysis(self):
        bias = self.generator._get_mutation_bias('take_profit_ticks')
        self.assertEqual(bias, 0.0)

    def test_early_trend_bias(self):
        regret_analysis = {
            'patterns': {
                'regret_distribution': {
                    'closed_too_early_trend': 40, # 40% -> Strong bias
                    'closed_too_early_spike': 10,
                    'closed_too_late': 10,
                    'optimal': 40
                }
            }
        }
        self.generator.update_regret_analysis(regret_analysis)

        # Check specific params
        bias_tp = self.generator._get_mutation_bias('take_profit_ticks')
        bias_tw = self.generator._get_mutation_bias('trail_distance_wide')
        bias_mh = self.generator._get_mutation_bias('max_hold_seconds')

        # Should be positive bias
        self.assertGreater(bias_tp, 0.3)
        self.assertGreater(bias_tw, 0.3)
        self.assertGreater(bias_mh, 0.3)

        # Unrelated param should have 0 bias
        bias_sl = self.generator._get_mutation_bias('stop_loss_ticks')
        self.assertEqual(bias_sl, 0.0)

    def test_too_late_bias(self):
        regret_analysis = {
            'patterns': {
                'regret_distribution': {
                    'closed_too_early_trend': 10,
                    'closed_too_early_spike': 10,
                    'closed_too_late': 40, # 40% -> Strong negative bias
                    'optimal': 40
                }
            }
        }
        self.generator.update_regret_analysis(regret_analysis)

        # Check specific params
        bias_mh = self.generator._get_mutation_bias('max_hold_seconds')
        bias_tw = self.generator._get_mutation_bias('trail_distance_wide')

        # Should be negative bias
        self.assertLess(bias_mh, -0.3)
        self.assertLess(bias_tw, -0.3)

    def test_mutation_output_shift(self):
        # Set up strong positive bias
        regret_analysis = {
            'patterns': {
                'regret_distribution': {
                    'closed_too_early_trend': 100,
                    'optimal': 0
                }
            }
        }
        self.generator.update_regret_analysis(regret_analysis)

        # Run many mutations and check average shift
        np.random.seed(42)
        shifts = []
        for i in range(100):
            # Force iteration in mutation range
            pset = self.generator.generate_mutation_set(600+i, 1, 'CORE', self.best_params)

            # Check MaxHold (should increase)
            if 'max_hold_seconds' in pset.parameters:
                shifts.append(pset.parameters['max_hold_seconds'] - self.best_params['max_hold_seconds'])

        avg_shift = np.mean(shifts)
        print(f"Average MaxHold shift with positive bias: {avg_shift:.2f}")
        self.assertGreater(avg_shift, 0, "Parameters should shift upwards on average with positive bias")

if __name__ == '__main__':
    unittest.main()
