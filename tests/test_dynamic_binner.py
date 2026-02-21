
import unittest
import numpy as np
import pickle
import os
from core.dynamic_binner import DynamicBinner, VariableBins

class TestDynamicBinner(unittest.TestCase):
    def test_variable_bins_transform(self):
        """Test VariableBins.transform with various inputs."""
        edges = np.array([0.0, 10.0, 20.0, 30.0])
        # Centers: 5.0, 15.0, 25.0
        vb = VariableBins(edges)

        # Test values within bins
        self.assertEqual(vb.transform(5.0), 5.0)
        self.assertEqual(vb.transform(15.0), 15.0)
        self.assertEqual(vb.transform(25.0), 25.0)

        # Test values near edges
        # 0.0 -> bin 0 (center 5.0)
        self.assertEqual(vb.transform(0.0), 5.0)
        # 9.9 -> bin 0
        self.assertEqual(vb.transform(9.9), 5.0)
        # 10.0 -> bin 1 (center 15.0) because side='right' in np.searchsorted means >= for edges[1] effectively?
        # bisect_right([0, 10, 20, 30], 10.0) -> index 2. 2-1 = 1. Center[1] = 15.0.
        # np.searchsorted([0, 10, 20, 30], 10.0, side='right') -> 2. 2-1 = 1. Center[1] = 15.0.
        self.assertEqual(vb.transform(10.0), 15.0)

        # Test values out of bounds
        # -5.0 -> bin 0
        self.assertEqual(vb.transform(-5.0), 5.0)
        # 35.0 -> bin 2 (center 25.0)
        self.assertEqual(vb.transform(35.0), 25.0)

        # Test equality with np.searchsorted implementation for random values
        # This ensures strict compatibility
        np.random.seed(42) # Ensure determinism
        for _ in range(100):
            val = np.random.uniform(-10, 40)

            # Old implementation logic
            idx_old = np.searchsorted(vb.edges, val, side='right') - 1
            idx_old = max(0, min(idx_old, vb.n_bins - 1))
            res_old = float(vb.centers[idx_old])

            res_new = vb.transform(val)
            self.assertEqual(res_new, res_old, f"Mismatch for value {val}")

    def test_variable_bins_transform_array(self):
        """Test VariableBins.transform_array."""
        edges = np.array([0.0, 10.0, 20.0])
        vb = VariableBins(edges)

        values = np.array([-5.0, 5.0, 10.0, 15.0, 25.0])
        expected = np.array([5.0, 5.0, 15.0, 15.0, 15.0]) # Centers are 5.0, 15.0

        result = vb.transform_array(values)
        np.testing.assert_array_equal(result, expected)

    def test_dynamic_binner_fit_transform(self):
        """Test full DynamicBinner workflow."""
        binner = DynamicBinner(min_bins=2, max_bins=5)
        np.random.seed(42) # Ensure determinism
        data = {
            'var1': np.random.normal(0, 1, 1000),
            'var2': np.random.uniform(0, 100, 1000)
        }
        binner.fit(data)

        self.assertTrue(binner.is_fitted)
        self.assertIn('var1', binner.variables)
        self.assertIn('var2', binner.variables)

        # Test transform
        val1 = binner.transform('var1', 0.5)
        self.assertIsInstance(val1, float)

        val2 = binner.transform('var2', 50.0)
        self.assertIsInstance(val2, float)

        # Unknown variable
        self.assertEqual(binner.transform('unknown', 123.0), 123.0)

    def test_serialization(self):
        """Test save/load."""
        binner = DynamicBinner()
        data = {'v': np.linspace(0, 10, 100)}
        binner.fit(data)

        filename = "temp_binner.pkl"
        try:
            binner.save(filename)
            loaded = DynamicBinner.load(filename)

            self.assertTrue(loaded.is_fitted)
            self.assertIn('v', loaded.variables)
            np.testing.assert_array_equal(loaded.variables['v'].edges, binner.variables['v'].edges)
        finally:
            if os.path.exists(filename):
                os.remove(filename)

if __name__ == "__main__":
    unittest.main()
