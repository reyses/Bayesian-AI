
import unittest
from unittest.mock import MagicMock
from core.exploration_mode import UnconstrainedExplorer, ExplorationConfig
from core.three_body_state import ThreeBodyQuantumState

class TestExplorationMode(unittest.TestCase):
    def test_explorer_initialization(self):
        config = ExplorationConfig(max_trades=100)
        explorer = UnconstrainedExplorer(config)
        self.assertEqual(explorer.config.max_trades, 100)
        self.assertEqual(explorer.trades_executed, 0)

    def test_should_fire_logic(self):
        config = ExplorationConfig(max_trades=10, allow_chaos_zone=True)
        explorer = UnconstrainedExplorer(config)

        # Create a mock state
        state = ThreeBodyQuantumState.null_state()

        # 1. Should fire in chaos zone
        decision = explorer.should_fire(state)
        self.assertTrue(decision['should_fire'])
        self.assertIn("UNCONSTRAINED", decision['reason'])

        # 2. Check trade counting
        explorer.record_trade(MagicMock())
        self.assertEqual(explorer.trades_executed, 1)

        # 3. Check max trades limit
        explorer.trades_executed = 10
        decision = explorer.should_fire(state)
        self.assertFalse(decision['should_fire'])
        self.assertIn("Exploration complete", decision['reason'])

if __name__ == '__main__':
    unittest.main()
