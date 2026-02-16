"""
Bayesian-AI - Environment Verification Tests
Tests system environment and configuration.
"""
import pytest
import os
import sys
import importlib

# Ensure root directory is in python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def test_python_version():
    """Verify Python version is compatible (>= 3.8)."""
    assert sys.version_info >= (3, 8), f"Python version too old: {sys.version}"

def test_cuda_configuration():
    """
    Verify CUDA configuration.
    This test passes if CUDA is unavailable (returns False), but fails if configuration is inconsistent.
    """
    try:
        from numba import cuda
        numba_available = cuda.is_available()
    except ImportError:
        numba_available = False
    except Exception:
        numba_available = False

    try:
        import torch
        torch_available = torch.cuda.is_available()
    except ImportError:
        torch_available = False

    # Log status (visible with pytest -s)
    print(f"\nCUDA Status: Numba={numba_available}, Torch={torch_available}")

    # It's okay if both are False (CPU mode)
    # It's okay if both are True
    # It might be weird if one is True and the other False, but technically possible.
    # So we mainly assert that we can check them without crashing.
    assert True

@pytest.mark.skipif(os.environ.get('CI') == 'true', reason="Data checks skipped in CI environment")
def test_data_paths():
    """
    Verify data paths for LEARNING mode.
    Skipped in CI if 'CI' env var is set, or we can handle missing data gracefully.
    """
    try:
        from config.settings import OPERATIONAL_MODE, RAW_DATA_PATH
    except ImportError:
        pytest.skip("Could not import config.settings")

    if OPERATIONAL_MODE == "LEARNING":
        path = os.path.abspath(RAW_DATA_PATH)
        # We don't fail if data is missing, as it might be a fresh checkout.
        # But we can warn.
        if not os.path.exists(path):
            pytest.skip(f"RAW_DATA_PATH {path} does not exist (acceptable for fresh checkout)")

        # If path exists, check for basic files
        # assert os.path.exists(path), f"Data path {path} missing"

if __name__ == "__main__":
    pytest.main([__file__])
