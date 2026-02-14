#!/usr/bin/env python3
"""
CUDA Fix Script
Attempts to repair PyTorch installation by forcing CUDA version.
"""
import subprocess
import sys
import shutil
import os

def check_nvidia_smi():
    """Check if nvidia-smi is available and works."""
    print("Checking NVIDIA Driver status...")
    if shutil.which("nvidia-smi") is None:
        print("WARNING: 'nvidia-smi' command not found. Ensure NVIDIA drivers are installed and in PATH.")
        return False

    try:
        result = subprocess.run(["nvidia-smi"], capture_output=True, text=True)
        if result.returncode != 0:
            print(f"WARNING: 'nvidia-smi' failed with exit code {result.returncode}.")
            print("Output:\n" + result.stderr)
            return False

        print("PASS: NVIDIA Driver detected via nvidia-smi.")
        return True
    except Exception as e:
        print(f"WARNING: Unexpected error checking nvidia-smi: {e}")
        return False

def reinstall_torch():
    print("\n=== Reinstalling PyTorch with CUDA 12.1 support ===")

    # Uninstall existing torch packages
    pkgs = ["torch", "torchvision", "torchaudio"]
    print(f"Uninstalling existing packages: {', '.join(pkgs)}...")
    subprocess.call([sys.executable, "-m", "pip", "uninstall", "-y"] + pkgs)

    # Install with correct index-url
    # We prioritize the CUDA index URL to avoid pulling CPU versions from PyPI
    cmd = [
        sys.executable, "-m", "pip", "install",
        "torch", "torchvision", "torchaudio",
        "--index-url", "https://download.pytorch.org/whl/cu121"
    ]

    print(f"Running: {' '.join(cmd)}")
    try:
        subprocess.check_call(cmd)
        print("\nSUCCESS: PyTorch reinstallation complete.")
        print("Please run 'python scripts/gpu_health_check.py' to verify CUDA access.")
    except subprocess.CalledProcessError as e:
        print(f"\nERROR: Failed to install PyTorch: {e}")
        sys.exit(1)

if __name__ == "__main__":
    if check_nvidia_smi():
        reinstall_torch()
    else:
        print("\nABORTING: PyTorch reinstallation stopped because NVIDIA drivers appear missing or broken.")
        print("Please install NVIDIA drivers first from https://www.nvidia.com/Download/index.aspx")
        print("If you are sure drivers are installed, run:\n")
        print(f"  {sys.executable} -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
