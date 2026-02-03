
"""
Bayesian-AI - CUDA Confirmation Test
Tests the confirmation engine logic.
"""
import pytest
import pandas as pd
import numpy as np
import os
import sys

# Add project root to path
sys.path.append(os.getcwd())

from cuda_modules.confirmation import get_confirmation_engine
import cuda_modules.confirmation

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
        """Verify initialization behaves correctly based on environment"""
        # Reset singleton to ensure we test initialization logic
        cuda_modules.confirmation._confirmation_engine = None

        try:
            from numba import cuda
            can_use_cuda = cuda.is_available()
        except:
            can_use_cuda = False

        if can_use_cuda:
            # Should succeed
            engine = get_confirmation_engine(use_gpu=True)
            assert engine.use_gpu is True
        else:
            # Should fail strictly
            with pytest.raises(RuntimeError):
                get_confirmation_engine(use_gpu=True)

            # CPU mode should work
            engine = get_confirmation_engine(use_gpu=False)
            assert engine.use_gpu is False

if __name__ == "__main__":
    sys.exit(pytest.main(["-v", __file__]))
