
import numpy as np
import pandas as pd
import unittest
import sys
import os

# Ensure core is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.quantum_field_engine import QuantumFieldEngine

class TestCPUPhysics(unittest.TestCase):
    def test_cpu_fallback(self):
        print("Initializing Engine...")
        # This should log a warning about CUDA not available, which is expected/good
        engine = QuantumFieldEngine(regression_period=21)

        # Force CPU usage explicitly to test the logic
        engine.use_gpu = False
        print(f"Engine initialized. use_gpu={engine.use_gpu}")

        # Create dummy data
        n = 100
        dates = pd.date_range(start='2024-01-01', periods=n, freq='15min')
        prices = np.linspace(100, 110, n) + np.random.randn(n) * 0.1
        volumes = np.random.randint(100, 1000, n).astype(float)

        df = pd.DataFrame({
            'timestamp': dates,
            'open': prices,
            'high': prices + 0.1,
            'low': prices - 0.1,
            'close': prices,
            'volume': volumes
        })

        print("Running batch_compute_states...")
        results = engine.batch_compute_states(df, use_cuda=False)

        self.assertGreater(len(results), 0)
        self.assertEqual(len(results), n - 21) # First 21 are skipped

        # Check first result
        res = results[0]
        state = res['state']

        self.assertIsNotNone(state)

        # Check fields are populated (not default 0.0 for everything)
        # Center should be close to price
        self.assertNotEqual(state.center_position, 0.0)
        self.assertTrue(90 < state.center_position < 120)

        # Sigma should be positive (or small epsilon)
        self.assertGreater(state.sigma_fractal, 0.0)

        # Z-score should be reasonable
        self.assertTrue(-10 < state.z_score < 10)

        # Probabilities should sum to ~1
        total_prob = state.P_at_center + state.P_near_upper + state.P_near_lower
        self.assertAlmostEqual(total_prob, 1.0, places=5)

        print(f"Computed {len(results)} states on CPU.")
        print(f"Sample State: Z={state.z_score:.2f}, Center={state.center_position:.2f}, Sigma={state.sigma_fractal:.4f}")
        print("CPU Fallback Test Passed!")

if __name__ == '__main__':
    unittest.main()
