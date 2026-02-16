#!/usr/bin/env python3
"""
GPU Health Check Script
Verifies PyTorch CUDA stack functionality.
"""
import sys
import time
import shutil
import subprocess

try:
    import torch
except ImportError:
    print("ERROR: PyTorch not installed.")
    print("Run: pip install -r requirements.txt")
    if '--strict' in sys.argv:
        sys.exit(1)
    sys.exit(0)

def check_nvidia_smi():
    """Check if nvidia-smi is available."""
    if shutil.which("nvidia-smi") is None:
        return False
    try:
        result = subprocess.run(["nvidia-smi"], capture_output=True, text=True)
        return result.returncode == 0
    except Exception:
        return False

def check_gpu():
    print("=== PyTorch GPU Health Check ===")

    print(f"PyTorch Version: {torch.__version__}")
    print(f"CUDA Version (PyTorch): {torch.version.cuda}")

    driver_ok = check_nvidia_smi()
    cuda_ok = torch.cuda.is_available()

    if not cuda_ok:
        print("\nFAIL: CUDA is not available to PyTorch.")
        if driver_ok:
            print("  - NVIDIA Driver is DETECTED via nvidia-smi.")
            print("  - PyTorch cannot see the GPU.")
            print("  - Likely Cause: CPU-only PyTorch version installed.")
            print("\nSUGGESTED FIX:")
            print("  Run: python scripts/fix_cuda.py")
        else:
            print("  - NVIDIA Driver is NOT DETECTED via nvidia-smi.")
            print("  - Please install NVIDIA drivers first.")
            print("  - If drivers are installed, ensure 'nvidia-smi' is in PATH.")

        return False

    try:
        device_count = torch.cuda.device_count()
        print(f"CUDA Devices Found: {device_count}")

        for i in range(device_count):
            props = torch.cuda.get_device_properties(i)
            print(f"  Device {i}: {props.name} | VRAM: {props.total_memory / 1e9:.2f} GB")

        # Functional Test
        print("\nRunning Functional Test...")
        device = torch.device('cuda')

        # Allocate
        t0 = time.time()
        a = torch.randn(2000, 2000, device=device)
        b = torch.randn(2000, 2000, device=device)
        print(f"  Allocation: {time.time()-t0:.4f}s")

        # Compute
        t0 = time.time()
        c = torch.matmul(a, b)
        torch.cuda.synchronize() # Wait for completion
        print(f"  Computation (2000x2000 matmul): {time.time()-t0:.4f}s")

        # Verification
        if c.shape == (2000, 2000):
            print("  Result Shape: OK")
        else:
            print(f"  Result Shape: FAIL {c.shape}")
            return False

        print("\nPASS: GPU stack is operational.")
        return True

    except Exception as e:
        print(f"\nFAIL: Exception during GPU operations: {e}")
        return False

if __name__ == '__main__':
    success = check_gpu()
    if '--strict' in sys.argv and not success:
        sys.exit(1)
    sys.exit(0)
