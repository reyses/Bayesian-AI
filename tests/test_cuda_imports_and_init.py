
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
        Requesting GPU on a system (likely without CUDA in CI) SHOULD fail if drivers are missing.
        Strict configuration now disables silent fallback.
        """
        try:
            from numba import cuda
            # In this sandbox environment, importing numba.cuda alone can trigger DynamicLibNotFoundError
            # if drivers are missing. We catch that here to determine availability.
            cuda_available = cuda.is_available()
        except Exception:
            cuda_available = False

        # Override for testing environment: if we are in a sandbox without drivers,
        # we treat it as "CUDA NOT available" regardless of what numba says initially.
        # This ensures our test expectation matches reality.
        # Check if we can actually run a dummy kernel or check device
        if cuda_available:
            try:
                cuda.get_current_device()
            except Exception:
                cuda_available = False

        from cuda_modules.velocity_gate import get_velocity_gate
        from cuda_modules.pattern_detector import get_pattern_detector
        from cuda_modules.confirmation import get_confirmation_engine

        if cuda_available:
            # If CUDA is really available, it should succeed
            try:
                gate = get_velocity_gate(use_gpu=True)
                assert gate.use_gpu is True
            except RuntimeError:
                pytest.fail("CUDA available but VelocityGate failed to init with GPU")
        else:
            # If CUDA is NOT available, it MUST raise RuntimeError
            # We expect these to fail because we are enforcing "No Silent Fallback"

            # Important: get_velocity_gate uses a singleton pattern. If it was already
            # initialized in a previous test (e.g. imports above), it might return the existing instance.
            # We must test the classes directly to ensure we are testing the constructor logic.
            from cuda_modules.velocity_gate import CUDAVelocityGate
            from cuda_modules.pattern_detector import CUDAPatternDetector
            from cuda_modules.confirmation import CUDAConfirmationEngine

            with pytest.raises(RuntimeError, match="CUDA requested.*but not available"):
                CUDAVelocityGate(use_gpu=True)

            with pytest.raises(RuntimeError, match="CUDA requested.*but not available"):
                CUDAPatternDetector(use_gpu=True)

            with pytest.raises(RuntimeError, match="CUDA requested.*but not available"):
                CUDAConfirmationEngine(use_gpu=True)

if __name__ == "__main__":
    sys.exit(pytest.main(["-v", __file__]))
