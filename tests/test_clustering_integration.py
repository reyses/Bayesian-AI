
import unittest
from unittest.mock import MagicMock, patch
import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Any, Dict, Optional
from training.fractal_clustering import FractalClusteringEngine, PatternTemplate

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
    oracle_meta: Optional[Dict] = None
    file_source: str = ''

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
                oracle_marker=1,
                oracle_meta={'mfe': float(np.random.uniform(1, 10)), 'mae': float(np.random.uniform(0, 5))},
            )
            self.patterns.append(p)

    def test_clustering_engine(self):
        engine = FractalClusteringEngine(n_clusters=10, max_variance=0.5)

        # New API: fit_hypervolume_tree
        tree = engine.fit_hypervolume_tree(self.patterns, min_group_size=5)
        templates = engine.templates

        self.assertTrue(len(templates) > 0)

        total_members = sum(t.member_count for t in templates)
        self.assertGreater(total_members, 0)

        # Check physics_variance
        for t in templates:
            self.assertTrue(hasattr(t, 'physics_variance'))
            self.assertGreaterEqual(t.physics_variance, 0.0)

    @patch('training.orchestrator_worker.simulate_trade_standalone')
    def test_validate_template_consistency(self, mock_simulate):
        from training.orchestrator_worker import _validate_template_consistency

        # Consistent wins across all members
        mock_outcome = MagicMock()
        mock_outcome.pnl = 10.0
        mock_simulate.return_value = mock_outcome

        template = MagicMock()
        template.mean_mfe_ticks = 0.0
        template.p75_mfe_ticks = 0.0
        template.mean_mae_ticks = 0.0
        template.p25_mae_ticks = 0.0

        is_valid, score, diag = _validate_template_consistency(
            template, self.patterns[:20], 2.0
        )
        # All positive PnL, no variance -> consistent
        self.assertTrue(is_valid)
        self.assertGreater(score, 0.0)
        self.assertEqual(diag['n_trades'], 20)

    @patch('training.orchestrator_worker.simulate_trade_standalone')
    def test_validate_template_inconsistent(self, mock_simulate):
        from training.orchestrator_worker import _validate_template_consistency

        # First half wins, second half loses -> inconsistent
        outcomes = []
        for i in range(20):
            m = MagicMock()
            m.pnl = 10.0 if i < 10 else -10.0
            outcomes.append(m)

        mock_simulate.side_effect = outcomes

        template = MagicMock()
        template.mean_mfe_ticks = 0.0
        template.p75_mfe_ticks = 0.0
        template.mean_mae_ticks = 0.0
        template.p25_mae_ticks = 0.0

        is_valid, score, diag = _validate_template_consistency(
            template, self.patterns[:20], 2.0
        )
        # 100% WR first half, 0% WR second half -> delta = 1.0 -> inconsistent
        self.assertFalse(is_valid)
        self.assertGreater(diag['wr_delta'], 0.2)

    def test_imr_geometric_split_basic(self):
        """Two well-separated clusters should produce 2 labels."""
        engine = FractalClusteringEngine()
        rng = np.random.RandomState(42)
        group_a = rng.normal(loc=-3.0, scale=0.5, size=(50, 16))
        group_b = rng.normal(loc=3.0, scale=0.5, size=(50, 16))
        residuals = np.vstack([group_a, group_b])

        labels, lineage = engine._imr_geometric_split(residuals, min_group_size=10)

        self.assertEqual(len(labels), 100)
        self.assertGreaterEqual(len(np.unique(labels)), 2)
        self.assertEqual(set(np.unique(labels)), set(range(len(np.unique(labels)))))
        self.assertIsInstance(lineage, dict)

    def test_imr_geometric_split_single_cluster(self):
        """Tight cluster should return single group."""
        engine = FractalClusteringEngine()
        rng = np.random.RandomState(42)
        residuals = rng.normal(loc=0.0, scale=0.1, size=(60, 16))

        labels, lineage = engine._imr_geometric_split(residuals, min_group_size=10)

        self.assertEqual(len(np.unique(labels)), 1)
        self.assertTrue(np.all(labels == 0))

    def test_imr_geometric_split_too_few(self):
        """Fewer than 2 * min_group_size -> single cluster."""
        engine = FractalClusteringEngine()
        rng = np.random.RandomState(42)
        residuals = rng.normal(size=(15, 16))

        labels, lineage = engine._imr_geometric_split(residuals, min_group_size=10)

        self.assertEqual(len(np.unique(labels)), 1)
        self.assertEqual(len(labels), 15)

    def test_imr_geometric_split_identical(self):
        """All identical patterns -> single cluster."""
        engine = FractalClusteringEngine()
        residuals = np.ones((40, 16))

        labels, lineage = engine._imr_geometric_split(residuals, min_group_size=10)

        self.assertTrue(np.all(labels == 0))

if __name__ == '__main__':
    unittest.main()
