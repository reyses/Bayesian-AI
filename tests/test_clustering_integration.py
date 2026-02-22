
import unittest
from unittest.mock import MagicMock, patch
import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Any
from sklearn.cluster import KMeans
from training.fractal_clustering import FractalClusteringEngine, PatternTemplate
from training.orchestrator import _optimize_template_task

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
    timeframe: str = '15s'
    depth: int = 0
    parent_type: str = ''
    parent_chain: Any = None
    oracle_marker: int = 1

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
                oracle_marker=1
            )
            self.patterns.append(p)

    @patch('training.fractal_clustering.CUDAKMeans', KMeans)
    def test_clustering_engine(self):
        engine = FractalClusteringEngine(n_clusters=10, max_variance=0.5)
        templates = engine.create_templates(self.patterns)

        self.assertTrue(len(templates) > 0)

        # Check total members
        total_members = sum(t.member_count for t in templates)
        self.assertEqual(total_members, 100)

        # Check sorting
        counts = [t.member_count for t in templates]
        self.assertEqual(counts, sorted(counts, reverse=True))

        # Check physics_variance
        for t in templates:
            self.assertTrue(hasattr(t, 'physics_variance'))
            self.assertGreaterEqual(t.physics_variance, 0.0)

    @patch('training.orchestrator_worker.simulate_trade_standalone')
    def test_optimize_template_task(self, mock_simulate):
        # Mock simulation result
        mock_outcome = MagicMock()
        mock_outcome.pnl = 10.0
        mock_simulate.return_value = mock_outcome

        # Mock generator
        mock_generator = MagicMock()
        mock_generator.generate_parameter_set.return_value.parameters = {'stop_loss': 10}
        mock_generator.optimize_pid.return_value = {'pid_kp': 0.1, 'pid_ki': 0.01, 'pid_kd': 0.05}

        template = MagicMock()
        template.mean_mfe_ticks = 0.0
        template.p75_mfe_ticks = 0.0
        template.mean_mae_ticks = 0.0
        template.p25_mae_ticks = 0.0
        subset = self.patterns[:5]
        iterations = 2
        point_value = 2.0

        args = (template, subset, iterations, mock_generator, point_value)

        best_params, best_sharpe = _optimize_template_task(args)

        self.assertIn('pid_kp', best_params)
        self.assertEqual(best_params['pid_kp'], 0.1)
        # All returns are 10.0. Standard deviation is 0. Sharpe should be 0.
        self.assertEqual(best_sharpe, 0.0)

    @patch('training.orchestrator_worker.simulate_trade_standalone')
    def test_optimize_template_task_variance(self, mock_simulate):
        # Mock generator to return 1 param set
        mock_generator = MagicMock()
        mock_generator.generate_parameter_set.return_value.parameters = {'stop_loss': 10}
        mock_generator.optimize_pid.return_value = {'pid_kp': 0.1, 'pid_ki': 0.01, 'pid_kd': 0.05}

        subset = self.patterns[:2]
        # PnL: [10, 20] -> mean=15, std=5 -> Sharpe=3.0

        out1 = MagicMock()
        out1.pnl = 10.0
        out2 = MagicMock()
        out2.pnl = 20.0

        mock_simulate.side_effect = [out1, out2]

        template = MagicMock()
        template.mean_mfe_ticks = 0.0
        template.p75_mfe_ticks = 0.0
        template.mean_mae_ticks = 0.0
        template.p25_mae_ticks = 0.0
        iterations = 1
        point_value = 2.0

        args = (template, subset, iterations, mock_generator, point_value)

        best_params, best_sharpe = _optimize_template_task(args)

        self.assertAlmostEqual(best_sharpe, 3.0)

if __name__ == '__main__':
    unittest.main()
