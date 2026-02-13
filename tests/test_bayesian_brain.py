"""
Bayesian AI v2.0 - Bayesian Brain Test
Validates BayesianBrain probability learning logic.
"""
import sys
import os
import unittest
import numpy as np

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.state_vector import StateVector
from core.bayesian_brain import BayesianBrain, TradeOutcome

class TestBayesianBrain(unittest.TestCase):
    def test_bayesian_brain_learning(self):
        """Test Bayesian probability learning"""
        print("\n=== TEST: BayesianBrain Learning ===")

        brain = BayesianBrain()

        # Create test state
        state = StateVector(
            L1_bias='bull', L2_regime='trending', L3_swing='higher_highs',
            L4_zone='at_killzone', L5_trend='up', L6_structure='bullish',
            L7_pattern='flag', L8_confirm=True, L9_cascade=True
        )

        # Initial probability (no data)
        initial_prob = brain.get_probability(state)
        # Pessimistic Prior Beta(1, 10) = 1/11 ~ 9%
        expected_prior = 0.09090909090909091 # 1/11
        # The original test used 0.09 as hardcoded approximation.
        # Let's check against the implementation: 1/11 is the theoretical value.
        # But core/bayesian_brain.py explicitly returns 0.09 if state not in table.
        # Wait, let's re-read core/bayesian_brain.py:
        # if state not in self.table: return 0.09
        # So it is exactly 0.09.

        self.assertAlmostEqual(initial_prob, 0.09, delta=1e-9, msg="Initial probability should be exactly 0.09")
        print(f"✓ Initial probability: {initial_prob:.2%}")

        # Simulate 10 wins, 2 losses
        for i in range(10):
            outcome = TradeOutcome(
                state=state, entry_price=21500, exit_price=21520,
                pnl=40, result='WIN', timestamp=float(i), exit_reason='trail_stop'
            )
            brain.update(outcome)

        for i in range(2):
            outcome = TradeOutcome(
                state=state, entry_price=21500, exit_price=21490,
                pnl=-20, result='LOSS', timestamp=float(i), exit_reason='structure_break'
            )
            brain.update(outcome)

        # Updated probability
        # Wins = 10, Losses = 2
        # Prior: alpha=1, beta=10
        # Posterior: alpha = 1 + 10 = 11
        #            beta = 10 + 2 = 12
        # Mean = alpha / (alpha + beta) = 11 / 23 = 0.47826...

        # However, check implementation:
        # wins = data['wins'] + 1
        # total = data['total'] + 11
        # prob = wins / total
        # wins = 10 + 1 = 11
        # total = 12 + 11 = 23
        # 11/23 = 0.47826...

        learned_prob = brain.get_probability(state)
        confidence = brain.get_confidence(state)

        expected_posterior = 11/23
        self.assertAlmostEqual(learned_prob, expected_posterior, delta=1e-9, msg=f"Posterior probability should be {expected_posterior}")

        print(f"✓ After 12 trades (10W-2L):")
        print(f"  Probability: {learned_prob:.2%}")
        print(f"  Confidence: {confidence:.2%}")

        # Confidence logic: min(total / 100.0, 1.0)
        # Total trades = 12
        # Confidence = 0.12
        self.assertAlmostEqual(confidence, 0.12, delta=1e-9, msg="Confidence should be 0.12")

        should_fire = brain.should_fire(state, min_prob=0.40, min_conf=0.10)
        self.assertTrue(should_fire, "Should fire with lower thresholds")

        should_fire_strict = brain.should_fire(state, min_prob=0.80)
        self.assertFalse(should_fire_strict, "Should not fire with strict thresholds")

        # Test statistics
        stats = brain.get_stats(state)
        self.assertEqual(stats['wins'], 10)
        self.assertEqual(stats['losses'], 2)
        print(f"✓ Stats: {stats}")

if __name__ == "__main__":
    unittest.main()
