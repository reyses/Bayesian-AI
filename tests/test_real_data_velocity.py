
import pytest
import pandas as pd
import numpy as np
import os
import sys
import time
from training.databento_loader import DatabentoLoader
from cuda.velocity_gate import get_velocity_gate

class TestRealDataVelocity:
    def test_velocity_gate_performance_real_data(self):
        """
        Loads the real Databento file and passes the full DataFrame to VelocityGate.
        Verifies that the optimization effectively handles large inputs by not processing the whole history.
        """
        # 1. Setup paths
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        file_path = os.path.join(project_root, 'DATA', 'RAW', 'glbx-mdp3-20250730.trades.0000.dbn.zst')

        if not os.path.exists(file_path):
            pytest.skip("Real data file not found in DATA/RAW")

        # 2. Load Data
        print(f"Loading data from {file_path}...")
        try:
            df = DatabentoLoader.load_data(file_path)
        except Exception as e:
            pytest.fail(f"Failed to load data: {e}")

        print(f"Loaded {len(df)} rows.")
        assert len(df) > 1000, "Dataset too small for meaningful performance test"

        # 3. Initialize Velocity Gate (force CPU to measure data prep overhead, or GPU if available)
        # We really want to verify that the 'slicing' happens before any heavy lifting
        gate = get_velocity_gate(use_gpu=True)

        # 4. Measure Performance
        # Pass the ENTIRE dataframe.
        # Without optimization, this would try to convert all N rows to float32 and copy to GPU.
        # With optimization, it should slice last 200 first.

        start_time = time.time()
        result = gate.detect_cascade(df)
        duration = time.time() - start_time

        print(f"Detection took: {duration:.4f} seconds")

        # 5. Assertions
        # 0.1s is generous; it should be much faster (e.g. 0.001s) if slicing works
        assert duration < 0.1, f"VelocityGate was too slow ({duration:.4f}s) on large dataset. Optimization might be missing."

        # Ensure it didn't crash and returned a bool
        assert isinstance(result, bool)

if __name__ == "__main__":
    # Allow running directly
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    pytest.main([__file__])
