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

from archive.cuda_modules.confirmation import get_confirmation_engine
import archive.cuda_modules.confirmation
from tests.utils import get_cuda_availability

class TestCUDAConfirmationEngine:
    def setup_method(self):
        if not get_cuda_availability():
            pytest.skip("GPU required for confirmation logic")
        self.engine = get_confirmation_engine(use_gpu=True)

    def test_confirmation_gpu_active_pattern(self):
        """Test confirmation logic on GPU: Volume spike confirmation"""
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
        confirmed = self.engine.confirm(df, L7_pattern_active=True)
        assert confirmed is True

    def test_no_confirmation_gpu_active_pattern(self):
        """Test no confirmation when volume is low"""
        # Setup volumes: [100, 100, 100] -> Mean 100. Threshold 120.
        # Next volume: 110 -> Should NOT confirm.

        volumes = [100, 100, 100, 110]
        df = pd.DataFrame({
            'volume': np.array(volumes, dtype=np.int32),
            'price': np.random.random(4)
        })

        confirmed = self.engine.confirm(df, L7_pattern_active=True)
        assert confirmed is False

    def test_confirmation_gpu_inactive_pattern(self):
        """Test no confirmation if L7 pattern is not active, regardless of volume"""
        volumes = [100, 100, 100, 500] # Massive volume spike
        df = pd.DataFrame({
            'volume': np.array(volumes, dtype=np.int32),
            'price': np.random.random(4)
        })

        # Pattern INACTIVE
        confirmed = self.engine.confirm(df, L7_pattern_active=False)
        assert confirmed is False

    def test_insufficient_data(self):
        """Test behavior with insufficient data (<3 bars)"""
        volumes = [100, 500]
        df = pd.DataFrame({
            'volume': np.array(volumes, dtype=np.int32),
            'price': np.random.random(2)
        })

        confirmed = self.engine.confirm(df, L7_pattern_active=True)
        assert confirmed is False

def test_gpu_fallback_enforcement():
    """Verify initialization behaves correctly based on environment"""
    # Reset singleton to ensure we test initialization logic
    archive.cuda_modules.confirmation._confirmation_engine = None

    cuda_available = get_cuda_availability()

    if cuda_available:
        # Should succeed
        engine = get_confirmation_engine(use_gpu=True)
        assert engine.use_gpu is True
    else:
        # Should fail strictly
        with pytest.raises(RuntimeError):
            get_confirmation_engine(use_gpu=True)

        # CPU mode should also fail
        with pytest.raises(RuntimeError):
            get_confirmation_engine(use_gpu=False)

if __name__ == "__main__":
    sys.exit(pytest.main(["-v", __file__]))
