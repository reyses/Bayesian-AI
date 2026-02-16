"""
Bayesian AI v2.0 - State Vector Test
Validates StateVector hashing and equality logic.
"""
import sys
import os
import unittest
from dataclasses import replace

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.state_vector import StateVector

class TestStateVector(unittest.TestCase):
    def test_state_vector_hashing(self):
        """Test StateVector hashing and equality (from test_phase1.py)"""
        print("\n=== TEST: StateVector Hashing ===")

        state1 = StateVector(
            L1_bias='bull',
            L2_regime='trending',
            L3_swing='higher_highs',
            L4_zone='at_killzone',
            L5_trend='up',
            L6_structure='bullish',
            L7_pattern='flag',
            L8_confirm=True,
            L9_cascade=True,
            timestamp=123456.0,
            price=21550.0
        )

        state2 = StateVector(
            L1_bias='bull',
            L2_regime='trending',
            L3_swing='higher_highs',
            L4_zone='at_killzone',
            L5_trend='up',
            L6_structure='bullish',
            L7_pattern='flag',
            L8_confirm=True,
            L9_cascade=True,
            timestamp=999999.0,  # Different timestamp
            price=99999.0        # Different price
        )

        # Should be equal (timestamp/price not part of hash)
        self.assertEqual(state1, state2, "States should be equal despite different metadata")
        self.assertEqual(hash(state1), hash(state2), "Hashes should match")

        print("[OK] StateVector hashing works correctly")
        print(f"  Hash: {hash(state1)}")
        print(f"  State dict: {state1.to_dict()}")

    def test_state_vector_strict_equality(self):
        """Test StateVector strict field equality (from verify_phase1_fixes.py)"""
        print("\n=== TEST: StateVector Strict Equality ===")

        s1 = StateVector(
            L1_bias='bull', L2_regime='trending', L3_swing='sideways',
            L4_zone='mid_range', L5_trend='up', L6_structure='bullish',
            L7_pattern='none', L8_confirm=False, L9_cascade=False,
            timestamp=100, price=100
        )
        s2 = StateVector(
            L1_bias='bull', L2_regime='trending', L3_swing='sideways',
            L4_zone='mid_range', L5_trend='up', L6_structure='bullish',
            L7_pattern='none', L8_confirm=False, L9_cascade=False,
            timestamp=200, price=200 # Metadata differs
        )
        s3 = StateVector(
            L1_bias='bear', # Field differs
            L2_regime='trending', L3_swing='sideways',
            L4_zone='mid_range', L5_trend='up', L6_structure='bullish',
            L7_pattern='none', L8_confirm=False, L9_cascade=False,
            timestamp=100, price=100
        )

        self.assertEqual(s1, s2, "StateVector should ignore metadata in equality check")
        self.assertNotEqual(s1, s3, "StateVector should detect field differences")

        # Verify hash consistency
        self.assertEqual(hash(s1), hash(s2))
        self.assertNotEqual(hash(s1), hash(s3))
        print("[OK] StateVector strict equality works correctly")

if __name__ == "__main__":
    unittest.main()
