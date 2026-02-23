
import unittest
from unittest.mock import MagicMock, patch
import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Any
from training.fractal_clustering import FractalClusteringEngine, PatternTemplate
from training.orchestrator_worker import _optimize_template_task
from sklearn.cluster import KMeans

@dataclass
class MockPatternEvent:
    idx: int
    timestamp: float
    price: float
    z_score: float
    velocity: float
    momentum: float
    coherence: float
    state: Any = None
    window_data: pd.DataFrame = None
    oracle_marker: int = 1 # Default to SCALP_LONG so it's not noise

class TestClusteringIntegration(unittest.TestCase):

    def setUp(self):
        # Create dummy patterns
        self.patterns = []
        for i in range(100):
            p = MockPatternEvent(
                idx=i,
                timestamp=1000.0 + i,
                price=100.0,
                z_score=np.random.normal(0, 1),
                velocity=np.random.normal(0, 1),
                momentum=np.random.normal(0, 1),
                coherence=np.random.uniform(0, 1),
                window_data=pd.DataFrame({'close': [100, 101, 102], 'price': [100, 101, 102], 'timestamp': [1000, 1001, 1002]}),
                oracle_marker=1 if i % 2 == 0 else -1
            )
            self.patterns.append(p)

    @patch('training.fractal_clustering.CUDAKMeans', side_effect=KMeans)
    def test_clustering_engine(self, mock_kmeans):
        engine = FractalClusteringEngine(n_clusters=10, max_variance=0.5)

        # Check extraction on a single pattern to ensure attributes are handled
        try:
            feats = engine.extract_features(self.patterns[0])
            self.assertEqual(len(feats), 16)
        except AttributeError as e:
            self.fail(f"extract_features raised AttributeError: {e}")

        # This will use the mocked KMeans (which is sklearn KMeans)
        templates = engine.create_templates(self.patterns)

        self.assertTrue(len(templates) > 0)

        # Check total members
        total_members = sum(t.member_count for t in templates)
        self.assertEqual(total_members, 100)

    @patch('training.orchestrator_worker.simulate_trade_standalone')
    def test_optimize_template_task(self, mock_simulate):
        # Mock simulation result
        mock_outcome = MagicMock()
        mock_outcome.pnl = 10.0
        mock_simulate.return_value = mock_outcome

        # Mock generator
        mock_generator = MagicMock()
        mock_generator.optimize_pid.return_value = {'pid_kp': 0.1, 'pid_ki': 0.01, 'pid_kd': 0.05}

        # Fix: Configure template mock to have float attributes for analytical exits
        template = MagicMock()
        template.mean_mfe_ticks = 10.0
        template.p75_mfe_ticks = 15.0
        template.mean_mae_ticks = 5.0
        template.p25_mae_ticks = 2.0

        subset = self.patterns[:5]
        iterations = 2
        point_value = 2.0

        args = (template, subset, iterations, mock_generator, point_value)

        best_params, best_sharpe = _optimize_template_task(args)

        # Check that we got params back
        self.assertIn('pid_kp', best_params)

    @patch('training.orchestrator_worker.simulate_trade_standalone')
    def test_optimize_template_task_variance(self, mock_simulate):
        # Mock generator
        mock_generator = MagicMock()
        mock_generator.optimize_pid.return_value = {'pid_kp': 0.1, 'pid_ki': 0.01, 'pid_kd': 0.05}

        subset = self.patterns[:2]
        # PnL: [10, 20] -> mean=15, std=5 -> Sharpe=3.0

        out1 = MagicMock()
        out1.pnl = 10.0
        out2 = MagicMock()
        out2.pnl = 20.0

        mock_simulate.side_effect = [out1, out2]

        template = MagicMock()
        template.mean_mfe_ticks = 10.0
        template.p75_mfe_ticks = 15.0
        template.mean_mae_ticks = 5.0
        template.p25_mae_ticks = 2.0

        iterations = 1
        point_value = 2.0

        args = (template, subset, iterations, mock_generator, point_value)

        best_params, best_sharpe = _optimize_template_task(args)

        self.assertAlmostEqual(best_sharpe, 3.0)

if __name__ == '__main__':
    unittest.main()
