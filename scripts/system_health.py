#!/usr/bin/env python3
"""
System Health Check Script
Consolidated verification for Python environment, dependencies, CUDA, and Data.
"""
import sys
import os
import shutil
import subprocess
import importlib.util

def check_python():
    print("=== Python Environment ===")
    print(f"Python Version: {sys.version}")
    if sys.version_info < (3, 8):
        print("WARNING: Python version < 3.8. Some features may not work.")
        return False
    print("PASS: Python version OK.")
    return True

def check_dependencies():
    print("\n=== Dependencies ===")
    required = ['numpy', 'pandas', 'torch', 'numba']
    missing = []
    for lib in required:
        if importlib.util.find_spec(lib) is None:
            missing.append(lib)
        else:
            try:
                module = importlib.import_module(lib)
                print(f"  - {lib}: {module.__version__}")
            except ImportError:
                missing.append(lib)

    if missing:
        print(f"FAIL: Missing dependencies: {', '.join(missing)}")
        return False
    print("PASS: Core dependencies installed.")
    return True

def check_nvidia_smi():
    if shutil.which("nvidia-smi") is None:
        return False
    try:
        result = subprocess.run(["nvidia-smi"], capture_output=True, text=True)
        return result.returncode == 0
    except Exception:
        return False

def check_cuda():
    print("\n=== CUDA & GPU ===")

    # 1. NVIDIA Driver
    driver_ok = check_nvidia_smi()
    if driver_ok:
        print("PASS: NVIDIA Driver detected via nvidia-smi.")
    else:
        print("WARNING: nvidia-smi not found or failed. Driver might be missing.")

    # 2. PyTorch CUDA
    try:
        import torch
        if torch.cuda.is_available():
            print(f"PASS: PyTorch CUDA available ({torch.version.cuda}).")
            print(f"  - Device Count: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                print(f"  - Device {i}: {props.name} | VRAM: {props.total_memory / 1e9:.2f} GB")

            # Simple functional test
            try:
                t = torch.tensor([1.0], device='cuda')
                print("  - Tensor allocation on GPU successful.")
            except Exception as e:
                print(f"  - FAIL: Tensor allocation failed: {e}")

        else:
            print("FAIL: PyTorch CUDA NOT available.")
            if driver_ok:
                print("  - Driver is present but PyTorch cannot see GPU.")
    except ImportError:
        print("FAIL: PyTorch not installed.")

    # 3. Numba CUDA
    try:
        from numba import cuda
        if cuda.is_available():
            print("PASS: Numba CUDA available.")
            try:
                device = cuda.get_current_device()
                name = device.name.decode('utf-8') if isinstance(device.name, bytes) else device.name
                print(f"  - Numba Device: {name}")
            except Exception as e:
                print(f"  - WARNING: Could not get Numba device info: {e}")
        else:
            print("WARNING: Numba CUDA NOT available.")
    except ImportError:
        print("WARNING: Numba not installed.")

    return True

def check_data():
    print("\n=== Data Availability ===")
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    raw_data_dir = os.path.join(project_root, 'DATA', 'RAW')

    if os.path.exists(raw_data_dir):
        files = os.listdir(raw_data_dir)
        if files:
            print(f"PASS: DATA/RAW exists and contains {len(files)} files.")
            # Optional: check for specific expected files if needed
        else:
            print("WARNING: DATA/RAW exists but is empty.")
    else:
        print(f"WARNING: DATA/RAW not found at {raw_data_dir}")

    # Check Testing DATA
    test_data_dir = os.path.join(project_root, 'tests', 'Testing DATA')
    if os.path.exists(test_data_dir) and os.listdir(test_data_dir):
        print(f"PASS: tests/Testing DATA exists.")
    else:
        print("WARNING: tests/Testing DATA not found or empty.")

    return True

def main():
    print("Starting System Health Check...")

    check_python()
    check_dependencies()
    check_cuda()
    check_data()

    print("\nSystem Health Check Complete.")

if __name__ == "__main__":
    main()
