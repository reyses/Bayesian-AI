import sys
import os
import unittest
import pandas as pd
import numpy as np

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.state_vector import StateVector
from core.layer_engine import LayerEngine

class TestPhase1Fixes(unittest.TestCase):
    def test_state_vector_equality(self):
        """Test StateVector strict field equality"""
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

    def test_layer_engine_L2_logic(self):
        """Test LayerEngine L2 threshold fix (Random Walk vs Trend)"""
        engine = LayerEngine(use_gpu=False)

        # Scenario 1: Random Walk (Vol 10)
        # Avg Range = 10.
        # 5-day Box should be ~22 (sqrt(5)*10).
        # Ratio ~2.2.
        # Should be 'chopping' (Threshold 3.0).

        # Generate random walk data
        np.random.seed(42)
        dates = pd.date_range(start='2025-01-01', periods=30, freq='D')

        # Create a "Chopping" scenario (High volatility but low displacement)
        # Day 1: 100-110 (up). Day 2: 110-100 (down). Day 3: 100-110...
        closes = []
        highs = []
        lows = []
        opens = []

        for i in range(30):
            base = 100
            o = base
            c = base
            h = base + 10
            l = base - 10
            # Wait, daily range is h-l = 20.
            # Avg Range = 20.
            # 5-day Box: max(h) - min(l) = 110 - 90 = 20.
            # Ratio 1.0.
            # 20 > 20 * 3.0 (60). False. 'chopping'. Correct.
            closes.append(c); highs.append(h); lows.append(l); opens.append(o)

        data_chop = pd.DataFrame({
            'open': opens, 'high': highs, 'low': lows, 'close': closes, 'volume': 1000
        }, index=dates)

        engine.daily_data = data_chop
        res = engine._compute_L2_30d()
        self.assertEqual(res, 'chopping', "Perfect chop should be 'chopping'")

        # Scenario 2: Strong Trend
        # Daily Range = 10.
        # Move 10 per day.
        # 5 days: 0-10, 10-20, ... 40-50.
        # Box: 0-50 = 50.
        # Avg Range = 10.
        # Ratio 5.0.
        # 50 > 10 * 3.0 (30). True. 'trending'. Correct.

        closes = []
        highs = []
        lows = []
        opens = []
        price = 100
        for i in range(30):
            price += 10
            o = price
            c = price + 5
            h = price + 10
            l = price
            # Daily Range: 10.
            closes.append(c); highs.append(h); lows.append(l); opens.append(o)

        data_trend = pd.DataFrame({
            'open': opens, 'high': highs, 'low': lows, 'close': closes, 'volume': 1000
        }, index=dates)

        engine.daily_data = data_trend
        res = engine._compute_L2_30d()
        self.assertEqual(res, 'trending', "Strong trend should be 'trending'")

        # Scenario 3: Reverting Move (The edge case)
        # Avg Range = 10.
        # We want Box ~ 25 (2.5x). Should be 'chopping' now (was 'trending' with 1.5x).

        closes = []
        highs = []
        lows = []
        opens = []

        for i in range(25): # Filler
            closes.append(100); highs.append(110); lows.append(100); opens.append(100)

        # Last 5 days
        # Day 26
        closes.append(110); highs.append(110); lows.append(100); opens.append(100)
        # Day 27
        closes.append(120); highs.append(120); lows.append(110); opens.append(110)
        # Day 28
        closes.append(125); highs.append(125); lows.append(115); opens.append(115)
        # Day 29
        closes.append(120); highs.append(120); lows.append(110); opens.append(110)
        # Day 30
        closes.append(110); highs.append(110); lows.append(100); opens.append(100)

        # Box: 125-100 = 25.
        # Avg Range: 10.
        # Ratio 2.5.

        data_revert = pd.DataFrame({
            'open': opens, 'high': highs, 'low': lows, 'close': closes, 'volume': 1000
        }, index=dates)

        engine.daily_data = data_revert
        res = engine._compute_L2_30d()
        self.assertEqual(res, 'chopping', "Reverting move (2.5x range) should be 'chopping' with new threshold")

if __name__ == '__main__':
    unittest.main()
