import sys
import os
import importlib.util

def check_import(module_name):
    try:
        spec = importlib.util.find_spec(module_name)
        if spec is None:
            print(f"[ERROR] Module '{module_name}' not found.")
            return False
        print(f"[OK] Module '{module_name}' found.")
        return True
    except Exception as e:
        print(f"[ERROR] Error checking '{module_name}': {e}")
        return False

def verify_environment():
    print("Verifying build environment...")
    all_passed = True

    # Check dependencies
    if not check_import("numba"): all_passed = False
    if not check_import("llvmlite"): all_passed = False
    if not check_import("PyInstaller"): all_passed = False

    # Check CUDA (optional but good to know)
    try:
        from numba import cuda
        if cuda.is_available():
            print("[OK] CUDA is available.")
        else:
            print("[WARNING] CUDA is NOT available. Executable may fall back to CPU.")
    except ImportError:
        print("[WARNING] Could not check CUDA availability (numba not found?).")

    # Check config directory
    if os.path.isdir("config"):
        print("[OK] 'config' directory found.")
    else:
        print("[ERROR] 'config' directory NOT found.")
        all_passed = False

    # Check/Create probability_table.pkl
    pkl_path = "probability_table.pkl"
    if os.path.exists(pkl_path):
        print(f"[OK] '{pkl_path}' exists.")
    else:
        print(f"[INFO] '{pkl_path}' not found. Generating default table...")
        try:
            # Add current directory to path so we can import core
            sys.path.append(os.getcwd())
            from core.bayesian_brain import BayesianBrain
            brain = BayesianBrain()
            brain.save(pkl_path)
            if os.path.exists(pkl_path):
                 print(f"[OK] '{pkl_path}' generated successfully.")
            else:
                 print(f"[ERROR] Failed to generate '{pkl_path}'.")
                 all_passed = False
        except Exception as e:
            print(f"[ERROR] Failed to generate '{pkl_path}': {e}")
            all_passed = False

    if all_passed:
        print("\nEnvironment verification PASSED.")
        return True
    else:
        print("\nEnvironment verification FAILED.")
        return False

if __name__ == "__main__":
    success = verify_environment()
    sys.exit(0 if success else 1)
