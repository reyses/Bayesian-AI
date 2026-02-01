import sys
import os
import unittest

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

class TestBuildIntegrity(unittest.TestCase):
    def test_operational_mode_access(self):
        """Verify OPERATIONAL_MODE is accessible without circular imports"""
        try:
            from config.settings import OPERATIONAL_MODE
            self.assertEqual(OPERATIONAL_MODE, "LEARNING")
        except ImportError as e:
            self.fail(f"Could not import OPERATIONAL_MODE: {e}")

    def test_engine_imports(self):
        """Verify EngineCore can import settings"""
        try:
            import engine_core
            # Just ensure no import error
        except Exception as e:
            self.fail(f"EngineCore import failed: {e}")

    def test_orchestrator_imports(self):
        """Verify Orchestrator can import settings via DatabentoLoader"""
        try:
            from training.orchestrator import TrainingOrchestrator
            # Just ensure no import error
        except Exception as e:
            self.fail(f"Orchestrator import failed: {e}")

if __name__ == '__main__':
    unittest.main()
