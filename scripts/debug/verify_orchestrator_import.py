import sys
import os

sys.path.append(os.getcwd())

print("Importing BayesianTrainingOrchestrator...")
try:
    from training.orchestrator import BayesianTrainingOrchestrator
    print("Import successful.")
except Exception as e:
    print(f"Import failed: {e}")
    sys.exit(1)
