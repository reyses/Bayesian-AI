
import sys
import unittest
from unittest.mock import MagicMock, patch
import pandas as pd
import numpy as np

# Mock numba.cuda before importing core modules
mock_numba = MagicMock()
mock_numba.__spec__ = MagicMock() # Fix for pandas_ta import check
mock_cuda = MagicMock()
mock_cuda.is_available.return_value = True

sys.modules['numba'] = mock_numba
sys.modules['numba.cuda'] = mock_cuda
# Ensure 'from numba import cuda' works as expected
mock_numba.cuda = mock_cuda

# Mock torch
sys.modules['torch'] = MagicMock()
sys.modules['torch.nn'] = MagicMock()
sys.modules['torch.nn.functional'] = MagicMock()

# Mock the kernels
mock_compute_physics = MagicMock()
mock_detect_archetype = MagicMock()

# Mock core.cuda_physics
sys.modules['core.cuda_physics'] = MagicMock()
sys.modules['core.cuda_physics'].compute_physics_kernel = mock_compute_physics
sys.modules['core.cuda_physics'].detect_archetype_kernel = mock_detect_archetype

# Mock core.cuda_pattern_detector
sys.modules['core.cuda_pattern_detector'] = MagicMock()
sys.modules['core.cuda_pattern_detector'].detect_patterns_cuda = MagicMock(return_value=(np.zeros(100), np.zeros(100)))
sys.modules['core.cuda_pattern_detector'].NUMBA_AVAILABLE = True

# Now import the engine
from core.quantum_field_engine import QuantumFieldEngine

class TestQuantumFieldEngineGPU(unittest.TestCase):
    def setUp(self):
        # Reset mocks
        mock_compute_physics.reset_mock()
        mock_detect_archetype.reset_mock()
        mock_cuda.device_array.reset_mock()
        mock_cuda.to_device.reset_mock()

        # Create dummy data
        dates = pd.date_range(start='2025-01-01', periods=100, freq='15s')
        self.df = pd.DataFrame({
            'timestamp': dates,
            'open': np.random.rand(100),
            'high': np.random.rand(100),
            'low': np.random.rand(100),
            'close': np.random.rand(100),
            'volume': np.random.rand(100)
        })
        self.df['price'] = self.df['close'] # Alias

    def test_batch_compute_calls_kernels(self):
        engine = QuantumFieldEngine()

        # Mock the copy_to_host behavior
        def mock_device_array(*args, **kwargs):
            m = MagicMock()
            arg0 = args[0] if args else 100
            if isinstance(arg0, int):
                size = arg0
            elif hasattr(arg0, 'shape'):
                size = arg0.shape[0]
            else:
                size = 100

            m.copy_to_host.return_value = np.zeros(size, dtype=np.float64)
            return m

        # We need to set side_effect on the mock_cuda object we injected
        mock_cuda.device_array.side_effect = mock_device_array
        mock_cuda.to_device.side_effect = mock_device_array

        # Run batch compute
        results = engine.batch_compute_states(self.df)

        # Verify kernels were called
        # mock_compute_physics[blocks, threads](...)
        self.assertTrue(mock_compute_physics.__getitem__.return_value.called, "compute_physics_kernel should be called via []")

        # Verify call arguments (including new constants)
        call_args = mock_compute_physics.__getitem__.return_value.call_args
        self.assertIsNotNone(call_args)

        # Args should be: d_prices, d_volumes, ..., d_prob2, rp, mean_x, inv_reg_period, inv_denom, denom
        # Total 14 array args + 5 scalar args = 19
        self.assertEqual(len(call_args[0]), 19, "compute_physics_kernel should be called with 19 arguments")

        # Verify reg_period passed correctly (default 21)
        self.assertEqual(call_args[0][14], 21)

        self.assertTrue(mock_detect_archetype.__getitem__.return_value.called, "detect_archetype_kernel should be called via []")

        # Verify results structure
        self.assertIsInstance(results, list)
        if results:
            self.assertIn('state', results[0])
            self.assertIn('structure_ok', results[0])

            # Check a value
            s = results[0]['state']
            # Since mock returns zeros, z_score should be 0.0
            self.assertEqual(s.z_score, 0.0)

if __name__ == '__main__':
    unittest.main()
