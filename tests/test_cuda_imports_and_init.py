
import pytest
import os
import sys

# Add project root to path
sys.path.append(os.getcwd())

class TestCUDAImportsAndInit:
    """
    Verifies that all CUDA modules can be imported and initialized.
    Checks robustness of CPU fallback mechanisms.
    """

    def test_import_velocity_gate(self):
        try:
            from cuda_modules.velocity_gate import get_velocity_gate
            gate = get_velocity_gate(use_gpu=False)
            assert gate is not None
            assert gate.use_gpu is False
        except ImportError as e:
            pytest.fail(f"Failed to import velocity_gate: {e}")
        except Exception as e:
            pytest.fail(f"Failed to init velocity_gate (CPU): {e}")

    def test_import_pattern_detector(self):
        try:
            from cuda_modules.pattern_detector import get_pattern_detector
            detector = get_pattern_detector(use_gpu=False)
            assert detector is not None
            assert detector.use_gpu is False
        except ImportError as e:
            pytest.fail(f"Failed to import pattern_detector: {e}")
        except Exception as e:
            pytest.fail(f"Failed to init pattern_detector (CPU): {e}")

    def test_import_confirmation(self):
        try:
            from cuda_modules.confirmation import get_confirmation_engine
            engine = get_confirmation_engine(use_gpu=False)
            assert engine is not None
            assert engine.use_gpu is False
        except ImportError as e:
            pytest.fail(f"Failed to import confirmation: {e}")
        except Exception as e:
            pytest.fail(f"Failed to init confirmation (CPU): {e}")

    def test_gpu_request_robustness(self):
        """
        Requesting GPU on a system (likely without CUDA in CI) should not crash.
        It should initialize, though .use_gpu might be False depending on hardware presence.
        """
        try:
            from cuda_modules.velocity_gate import get_velocity_gate
            from cuda_modules.pattern_detector import get_pattern_detector
            from cuda_modules.confirmation import get_confirmation_engine

            # These might print "GPU not available" to stdout, which is fine.
            gate = get_velocity_gate(use_gpu=True)
            detector = get_pattern_detector(use_gpu=True)
            engine = get_confirmation_engine(use_gpu=True)

            assert gate is not None
            assert detector is not None
            assert engine is not None

        except Exception as e:
            pytest.fail(f"Initialization with use_gpu=True crashed: {e}")

if __name__ == "__main__":
    sys.exit(pytest.main(["-v", __file__]))
