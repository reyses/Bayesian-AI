import unittest
import numpy as np
import pandas as pd
from core.quantum_field_engine import QuantumFieldEngine, PATTERN_NONE, PATTERN_COMPRESSION, PATTERN_WEDGE, PATTERN_BREAKDOWN

class TestPatternRecognition(unittest.TestCase):
    def setUp(self):
        self.engine = QuantumFieldEngine()

    def test_compression_pattern(self):
        # Create data that compresses
        # Recent range (5) < 0.7 * Prev range (5)
        # Bars 0-4: Big range (20)
        # Bars 5-9: Small range (4)
        # 4 < 20 * 0.7 (14) -> Compression at index 9

        highs = np.array([120, 120, 120, 120, 120, 112, 112, 112, 112, 112, 112, 112], dtype=float)
        lows =  np.array([100, 100, 100, 100, 100, 108, 108, 108, 108, 108, 108, 108], dtype=float)

        patterns = self.engine._detect_geometric_patterns(highs, lows)

        self.assertEqual(patterns[9], PATTERN_COMPRESSION)
        self.assertEqual(patterns[8], PATTERN_NONE) # Not yet complete window (needs 9 prior bars + current = 10 bars total? No.)
        # Logic:
        # Index 9:
        # Rec: 5-9. Prev: 0-4.
        # Index 8:
        # Rec: 4-8. Prev: -1 to 3 (wrapped).
        # patterns[:9] are set to NONE. So index 8 is NONE.
        pass

    def test_wedge_pattern(self):
        # Wedge: Higher Lows AND Lower Highs over 5 bars (compare idx vs idx-4)
        highs = np.full(20, 120.0)
        lows = np.full(20, 100.0)

        # At index 10 (11th bar)
        # idx-4 is index 6.
        highs[6] = 120.0
        lows[6] = 100.0

        highs[10] = 115.0 # Lower High
        lows[10] = 105.0  # Higher Low

        patterns = self.engine._detect_geometric_patterns(highs, lows)
        self.assertEqual(patterns[10], PATTERN_WEDGE)

    def test_breakdown_pattern(self):
        # Breakdown: Low < min(prev 4 lows)
        highs = np.full(20, 120.0)
        lows = np.full(20, 100.0)

        # At index 10
        # Prev 4 lows (6,7,8,9) are 100.
        lows[10] = 95.0

        patterns = self.engine._detect_geometric_patterns(highs, lows)
        self.assertEqual(patterns[10], PATTERN_BREAKDOWN)

    def test_batch_compute_integration(self):
        # Test that batch_compute_states actually populates pattern_type
        # Need enough data for regression (21) + some for pattern
        n = 30
        df = pd.DataFrame({
            'timestamp': pd.date_range(start='2024-01-01', periods=n, freq='15s'),
            'close': np.linspace(100, 110, n),
            'high': np.linspace(101, 111, n),
            'low': np.linspace(99, 109, n),
            'volume': np.full(n, 1000)
        })

        # Trigger Breakdown at end
        df.loc[n-1, 'low'] = 90.0 # Breakdown

        results = self.engine.batch_compute_states(df)

        # Check last result
        last_state = results[-1]['state']
        self.assertEqual(last_state.pattern_type, PATTERN_BREAKDOWN)

if __name__ == '__main__':
    unittest.main()
