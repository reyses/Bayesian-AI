
import pytest
import pandas as pd
import numpy as np
import os
import sys

# Add project root to path
sys.path.append(os.getcwd())

from cuda_modules.confirmation import get_confirmation_engine

class TestCUDAConfirmationEngine:
    def setup_method(self):
        # By default use CPU for reliable testing in CI
        self.engine_cpu = get_confirmation_engine(use_gpu=False)

    def test_confirmation_cpu_active_pattern(self):
        """Test confirmation logic on CPU: Volume spike confirmation"""
        # Logic: current volume > mean of last 3 * 1.2 (inclusive of current)
        # Setup volumes: [100, 100, 100] -> Previous avg 100.
        # If we use 140: Mean of (100, 100, 140) is 113.33.
        # Threshold: 113.33 * 1.2 = 136.
        # 140 > 136 -> Should confirm.

        volumes = [100, 100, 100, 140]
        df = pd.DataFrame({
            'volume': np.array(volumes, dtype=np.int32),
            'price': np.random.random(4)
        })

        # Pattern MUST be active for confirmation to run
        confirmed = self.engine_cpu.confirm(df, L7_pattern_active=True)
        assert confirmed is True

    def test_no_confirmation_cpu_active_pattern(self):
        """Test no confirmation when volume is low"""
        # Setup volumes: [100, 100, 100] -> Mean 100. Threshold 120.
        # Next volume: 110 -> Should NOT confirm.

        volumes = [100, 100, 100, 110]
        df = pd.DataFrame({
            'volume': np.array(volumes, dtype=np.int32),
            'price': np.random.random(4)
        })

        confirmed = self.engine_cpu.confirm(df, L7_pattern_active=True)
        assert confirmed is False

    def test_confirmation_cpu_inactive_pattern(self):
        """Test no confirmation if L7 pattern is not active, regardless of volume"""
        volumes = [100, 100, 100, 500] # Massive volume spike
        df = pd.DataFrame({
            'volume': np.array(volumes, dtype=np.int32),
            'price': np.random.random(4)
        })

        # Pattern INACTIVE
        confirmed = self.engine_cpu.confirm(df, L7_pattern_active=False)
        assert confirmed is False

    def test_insufficient_data(self):
        """Test behavior with insufficient data (<3 bars)"""
        volumes = [100, 500]
        df = pd.DataFrame({
            'volume': np.array(volumes, dtype=np.int32),
            'price': np.random.random(2)
        })

        confirmed = self.engine_cpu.confirm(df, L7_pattern_active=True)
        assert confirmed is False

    def test_gpu_fallback_init(self):
        """Verify initialization with use_gpu=True does not crash"""
        try:
            engine_gpu = get_confirmation_engine(use_gpu=True)
            assert engine_gpu is not None
        except Exception as e:
            pytest.fail(f"Initialization with use_gpu=True failed: {e}")

if __name__ == "__main__":
    sys.exit(pytest.main(["-v", __file__]))
