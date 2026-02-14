import unittest
from unittest.mock import MagicMock, patch
import pandas as pd
import numpy as np
from training.doe_parameter_generator import DOEParameterGenerator
from training.orchestrator import BayesianTrainingOrchestrator, TIMEFRAME_MAP

class TestMultiTimeframeDOE(unittest.TestCase):
    def setUp(self):
        self.detector = MagicMock()
        self.generator = DOEParameterGenerator(self.detector)

    def test_doe_parameter_generation(self):
        """Verify DOE generates timeframe_idx in range [0, 2]"""
        # Test Latin Hypercube (iterations 10+)
        # We run enough iterations to likely hit all 3 values (0, 1, 2)
        # Random seed is fixed per day, so we might need to vary day or iteration
        found_indices = set()
        for i in range(10, 110):
            pset = self.generator.generate_latin_hypercube_set(i, 1, 'CORE')
            tf_idx = pset.parameters.get('timeframe_idx')
            self.assertIsNotNone(tf_idx)
            found_indices.add(tf_idx)

        # We expect to see 0, 1, 2 eventually
        for idx in [0, 1, 2]:
            self.assertIn(idx, found_indices)

    def test_baseline_timeframe(self):
        """Verify Baseline (iterations 0-9) always uses 15s (idx=1)"""
        for i in range(10):
            pset = self.generator.generate_baseline_set(i, 1, 'CORE')
            tf_idx = pset.parameters.get('timeframe_idx')
            self.assertEqual(tf_idx, 1, f"Baseline iteration {i} should use timeframe_idx=1")

    def test_mutation_bounds_timeframe(self):
        """Verify mutation keeps timeframe_idx within [0, 2]"""
        # Start with max value
        params = {'timeframe_idx': 2, 'stop_loss_ticks': 10}

        # Run many mutations to try and force out of bounds
        for i in range(50):
            pset = self.generator.generate_mutation_set(600 + i, 1, 'CORE', params)
            tf_idx = pset.parameters.get('timeframe_idx')
            self.assertIsNotNone(tf_idx)
            self.assertGreaterEqual(tf_idx, 0)
            self.assertLessEqual(tf_idx, 2)

    @patch('training.orchestrator.QuantumBayesianBrain')
    @patch('training.orchestrator.QuantumFieldEngine')
    @patch('training.orchestrator.ContextDetector')
    @patch('training.orchestrator.DOEParameterGenerator')
    def test_orchestrator_optimize_day_timeframes(self, MockDOE, MockContext, MockEngine, MockBrain):
        """Verify optimize_day groups params and calls precompute with correct intervals"""
        # Setup mocks
        config = MagicMock()
        config.checkpoint_dir = "checkpoints_test"
        config.iterations = 10
        config.exploration_mode = False

        # Instantiate orchestrator (uses mocked components)
        orchestrator = BayesianTrainingOrchestrator(config)

        # Helper class to simulate ParameterSet
        class ParamSet:
            def __init__(self, params):
                self.parameters = params

        # Create params with distinct timeframe_idx
        # 0 -> 5s, 1 -> 15s, 2 -> 60s
        param_sets = [
            ParamSet({'timeframe_idx': 0, 'stop_loss_ticks': 10}),
            ParamSet({'timeframe_idx': 1, 'stop_loss_ticks': 20}),
            ParamSet({'timeframe_idx': 2, 'stop_loss_ticks': 30}),
            ParamSet({'timeframe_idx': 0, 'stop_loss_ticks': 15})
        ]

        orchestrator.config.iterations = len(param_sets)

        # Mock param generator to return our list based on iteration
        # Note: generate_parameter_set is called with iteration=0, 1, 2...
        orchestrator.param_generator.generate_parameter_set.side_effect = lambda iteration, day, context: param_sets[iteration]

        # Mock _precompute_day_states to return dummy list
        # We just need it to return something True-ish so we enter optimization
        orchestrator._precompute_day_states = MagicMock(return_value=[{'bar_idx': 0, 'structure_ok': True}])

        # Mock _optimize_cpu_sequential
        # It is called with (precomputed, day_data, params, ...)
        # We need to return a result for EACH param in 'params'
        def side_effect_optimize(precomputed, day_data, params, day, date=None, total_days=None):
            # Return dummy results
            results = [{'sharpe': 1.0, 'trades': []} for _ in params]
            return 0, results

        orchestrator._optimize_cpu_sequential = MagicMock(side_effect=side_effect_optimize)

        # Ensure CUDA is not available to force CPU path (or mock it)
        # We mocked _optimize_cpu_sequential, so we just need optimize_day to choose it.
        # optimize_day checks torch.cuda.is_available(). We can patch torch.
        with patch('torch.cuda.is_available', return_value=False):
            day_data = pd.DataFrame({'price': [100]*30, 'timestamp': range(30)})
            orchestrator.optimize_day(1, '2025-01-01', day_data)

        # Verify _precompute_day_states calls
        # We expect calls with interval='5s', '15s', '60s'
        calls = orchestrator._precompute_day_states.call_args_list
        # Extract 'interval' kwarg or 2nd positional arg
        # Signature: _precompute_day_states(day_data, interval='15s', ...)
        intervals_called = []
        for c in calls:
            args, kwargs = c
            if 'interval' in kwargs:
                intervals_called.append(kwargs['interval'])
            elif len(args) > 1:
                intervals_called.append(args[1])
            else:
                intervals_called.append('15s') # Default

        self.assertIn('5s', intervals_called)
        self.assertIn('15s', intervals_called)
        self.assertIn('60s', intervals_called)
        self.assertEqual(len(set(intervals_called)), 3)

        # Verify optimize calls
        # We have 3 groups (0, 1, 2). So 3 calls to optimize.
        self.assertEqual(orchestrator._optimize_cpu_sequential.call_count, 3)

if __name__ == '__main__':
    unittest.main()
