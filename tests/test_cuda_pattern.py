
"""
Bayesian-AI - CUDA Pattern Test
Tests pattern detection logic.
"""
import pytest
import pandas as pd
import numpy as np
import os
import sys

# Add project root to path
sys.path.append(os.getcwd())

from cuda_modules.pattern_detector import get_pattern_detector, PATTERN_COMPRESSION, PATTERN_WEDGE, PATTERN_BREAKDOWN, PATTERN_NONE
import cuda_modules.pattern_detector

class TestCUDAPatternDetector:
    def setup_method(self):
        # By default use CPU for reliable testing in CI
        self.detector_cpu = get_pattern_detector(use_gpu=False)

    def test_compression_pattern(self):
        """Test compression pattern detection (range contraction)"""
        # Create data where first 5 bars have large range, next 5 have small range
        highs = []
        lows = []

        # Previous range (large): idx 0-4
        for i in range(5):
            highs.append(100.0 + (i % 2) * 10) # Ranges approx 10
            lows.append(90.0)

        # Recent range (small): idx 5-9
        for i in range(5):
            highs.append(100.0 + (i % 2) * 2) # Ranges approx 2
            lows.append(98.0)

        df = pd.DataFrame({
            'high': np.array(highs, dtype=np.float32),
            'low': np.array(lows, dtype=np.float32)
        })

        pattern, conf = self.detector_cpu.detect(df)
        assert pattern == 'compression'
        assert conf == 0.85

    def test_wedge_pattern(self):
        """Test wedge pattern (higher lows, lower highs)"""
        # highs decreasing, lows increasing over 5 bars
        highs = [110, 109, 108, 107, 105]
        lows =  [90,  91,  92,  93,  95]

        df = pd.DataFrame({
            'high': np.array(highs, dtype=np.float32),
            'low': np.array(lows, dtype=np.float32)
        })

        pattern, conf = self.detector_cpu.detect(df)
        assert pattern == 'wedge'
        assert conf == 0.75

    def test_breakdown_pattern(self):
        """Test breakdown pattern (new low below previous support)"""
        # flat support then drop
        highs = [100, 100, 100, 100, 100]
        lows =  [90,  90,  90,  90,  80] # Drop to 80

        df = pd.DataFrame({
            'high': np.array(highs, dtype=np.float32),
            'low': np.array(lows, dtype=np.float32)
        })

        pattern, conf = self.detector_cpu.detect(df)
        assert pattern == 'breakdown'
        assert conf == 0.90

    def test_no_pattern(self):
        """Test no pattern detected"""
        highs = [100, 101, 102, 103, 104]
        lows =  [90,  91,  92,  93,  94]
        df = pd.DataFrame({
            'high': np.array(highs, dtype=np.float32),
            'low': np.array(lows, dtype=np.float32)
        })

        pattern, conf = self.detector_cpu.detect(df)
        assert pattern == 'none'
        assert conf == 0.0

    def test_insufficient_data(self):
        """Test behavior with insufficient data"""
        highs = [100, 101]
        lows =  [90,  91]
        df = pd.DataFrame({
            'high': np.array(highs, dtype=np.float32),
            'low': np.array(lows, dtype=np.float32)
        })

        pattern, conf = self.detector_cpu.detect(df)
        assert pattern == 'none'

    def test_gpu_fallback(self):
        """
        Verify that we can initialize in the correct mode for the environment.
        """
        # Reset singleton to ensure we test initialization logic
        cuda_modules.pattern_detector._pattern_detector = None

        try:
            from numba import cuda
            can_use_cuda = cuda.is_available()
        except:
            can_use_cuda = False

        if can_use_cuda:
            # If CUDA is available, use_gpu=True should work
            try:
                detector = get_pattern_detector(use_gpu=True)
                assert detector.use_gpu is True
            except Exception as e:
                pytest.fail(f"CUDA available but init failed: {e}")
        else:
            # If CUDA is NOT available, use_gpu=True should RAISE RuntimeError (strict mode)
            with pytest.raises(RuntimeError):
                get_pattern_detector(use_gpu=True)

            # But use_gpu=False should work
            detector = get_pattern_detector(use_gpu=False)
            assert detector.use_gpu is False

if __name__ == "__main__":
    sys.exit(pytest.main(["-v", __file__]))
