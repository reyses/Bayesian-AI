"""
Bayesian-AI - Manifest Integrity Check
Validates that the workflow manifest matches the actual file structure.
"""
import json
import os
import sys
import importlib
import time

# Ensure root directory is in python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import traceback

def log(msg):
    print(f"[INTEGRITY] {msg}")

def check_file_existence(manifest_path):
    log("Checking file existence from manifest...")
    try:
        with open(manifest_path, 'r') as f:
            manifest = json.load(f)
    except FileNotFoundError:
        log(f"FAIL: Manifest file not found at {manifest_path}")
        return False

    all_files = []
    # Collect files from pipeline
    for stage, files in manifest.get('pipeline', {}).items():
        all_files.extend(files)

    # Collect files from layers
    for layer, filepath in manifest.get('layers', {}).items():
        all_files.append(filepath)

    # Unique files
    all_files = list(set(all_files))

    missing_files = []
    for filepath in all_files:
        if not os.path.exists(filepath):
            missing_files.append(filepath)
            log(f"FAIL: Missing file {filepath}")
        else:
            log(f"OK: Found {filepath}")

    if missing_files:
        return False
    return True

def check_imports():
    log("Verifying imports...")
    modules_to_check = [
        'core.bayesian_brain',
        'core.state_vector',
        'core.layer_engine',
        'core.data_aggregator',
        'cuda_modules.velocity_gate',
        'cuda_modules.pattern_detector',
        'cuda_modules.confirmation',
        'execution.wave_rider',
        'training.databento_loader',
        'training.orchestrator',
        'config.symbols',
        'engine_core'
    ]

    success = True
    for module in modules_to_check:
        try:
            importlib.import_module(module)
            log(f"OK: Imported {module}")
        except Exception as e:
            log(f"FAIL: Could not import {module}: {e}")
            traceback.print_exc()
            success = False
    return success

def check_cuda():
    log("Checking CUDA availability...")
    try:
        from numba import cuda
        if cuda.is_available():
            log("OK: CUDA is available.")
            try:
                gpu = cuda.get_current_device()
                log(f"OK: GPU Detected: {gpu.name}")
            except Exception as e:
                log(f"WARNING: Could not get GPU name: {e}")
        else:
            log("WARNING: CUDA is NOT available. System will run in CPU fallback mode.")
            # Not a fail condition for the build itself, but important note
    except ImportError:
        log("FAIL: Numba not installed or CUDA support missing.")
        return False
    except Exception as e:
        log(f"WARNING: Error checking CUDA (likely missing drivers): {e}")
        # Not a fail condition for build integrity in CI/fallback
        return True
    return True

def check_bayesian_io():
    log("Verifying BayesianBrain I/O...")
    test_file = "test_prob_table.pkl"
    try:
        from core.bayesian_brain import BayesianBrain, TradeOutcome
        from core.state_vector import StateVector
        import pickle

        brain = BayesianBrain()

        # Create a dummy state and outcome
        state = StateVector.null_state()
        outcome = TradeOutcome(
            state=state,
            entry_price=100.0,
            exit_price=110.0,
            pnl=10.0,
            result='WIN',
            timestamp=time.time(),
            exit_reason='test'
        )

        brain.update(outcome)

        # Save
        brain.save(test_file)

        if not os.path.exists(test_file):
            log("FAIL: Probability table file not created.")
            return False

        # Load
        brain2 = BayesianBrain()
        brain2.load(test_file)

        # Verify
        if len(brain2.table) != 1:
            log("FAIL: Loaded table has incorrect size.")
            return False

        prob = brain2.get_probability(state)
        if prob < 0.5: # Should be > 0.5 since we added a WIN
             log(f"FAIL: Probability incorrect after load. Got {prob}")
             return False

        # Cleanup
        if os.path.exists(test_file):
            os.remove(test_file)
        log("OK: BayesianBrain save/load verification passed.")
        return True

    except Exception as e:
        log(f"FAIL: BayesianBrain I/O error: {e}")
        traceback.print_exc()
        if os.path.exists(test_file):
            os.remove(test_file)
        return False

def check_databento_loader():
    log("Verifying DatabentoLoader...")
    try:
        from training.databento_loader import DatabentoLoader
        if not hasattr(DatabentoLoader, 'load_data'):
            log("FAIL: DatabentoLoader missing 'load_data' method.")
            return False
        log("OK: DatabentoLoader class and method found.")
        return True
    except Exception as e:
        log(f"FAIL: DatabentoLoader check error: {e}")
        return False

def main():
    log("Starting Integrity Check...")

    checks = [
        check_file_existence('config/workflow_manifest.json'),
        check_imports(),
        check_cuda(),
        check_bayesian_io(),
        check_databento_loader()
    ]

    if all(checks):
        log("Integrity Check COMPLETE: ALL PASS")
        sys.exit(0)
    else:
        log("Integrity Check COMPLETE: FAILURES DETECTED")
        sys.exit(1)

if __name__ == "__main__":
    main()
