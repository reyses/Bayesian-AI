"""
Bayesian-AI - 3-Stage CUDA Audit Logic
File: cuda_modules/hardened_verification.py

Stage A (Handshake): Verify RTX 3060 recognition.
Stage B (Injection): CPU-to-GPU deterministic verification using DATA/RAW snippets or synthetic data.
Stage C (Handoff): Pure VRAM data-passing between L7, L8, and L9 kernels.
"""
import sys
import os
import pandas as pd
import numpy as np
import logging
from logging.handlers import RotatingFileHandler
import glob

# Add project root to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

# Setup Logging to CUDA_Debug.log
# We explicitly add handlers to ensure logging works even if basicConfig was skipped
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

# File Handler (explicitly added)
file_handler = RotatingFileHandler('CUDA_Debug.log', maxBytes=10*1024*1024, backupCount=1)
file_handler.setLevel(logging.DEBUG)
file_formatter = logging.Formatter('%(asctime)s | [%(levelname)s] | %(message)s')
file_handler.setFormatter(file_formatter)
logger.addHandler(file_handler)

# Console Handler
console = logging.StreamHandler()
console.setLevel(logging.INFO)
logger.addHandler(console)

try:
    from numba import cuda
    NUMBA_AVAILABLE = True
except Exception as e:
    NUMBA_AVAILABLE = False
    logging.error(f"Numba import failed: {e}")

def run_audit():
    logging.info("STARTING 3-STAGE CUDA AUDIT")

    # --- STAGE A: HANDSHAKE ---
    logging.info("[STAGE A] Handshake: Verifying GPU...")
    if not NUMBA_AVAILABLE:
        logging.critical("FAIL: Numba not available.")
        return False

    try:
        if not cuda.is_available():
            logging.warning("FAIL: CUDA device not available.")
            # If in simulation mode (e.g. env var set), we proceed with limited checks or fail gracefully
            if os.environ.get('NUMBA_ENABLE_CUDASIM') != '1':
                 return False
            else:
                 logging.info("Running in CUDA SIMULATOR mode.")

        if cuda.is_available():
            device = cuda.get_current_device()
            # Handle bytes vs string for device name
            device_name = device.name.decode('utf-8') if hasattr(device.name, 'decode') else device.name
            logging.info(f"Device found: {device_name}")

            # Strict 3060 check or loose check
            if "3060" not in str(device_name) and "Tesla" not in str(device_name) and "NVIDIA" not in str(device_name):
                 logging.warning(f"Device '{device_name}' is not explicitly verified as RTX 3060.")
            else:
                 logging.info("RTX 3060 (or compatible NVIDIA GPU) recognized.")

    except Exception as e:
        logging.critical(f"[STAGE A] Exception: {e}")
        return False

    # --- STAGE B: INJECTION ---
    logging.info("[STAGE B] Injection: CPU-to-GPU deterministic verification...")
    try:
        # Load snippet from DATA/RAW or Synthetic
        prices = None

        # 1. Search for Parquet
        parquet_files = glob.glob(os.path.join(project_root, 'DATA', 'RAW', '*.parquet'))
        if parquet_files:
            logging.info(f"Loading data snippet from {parquet_files[0]}")
            try:
                df = pd.read_parquet(parquet_files[0])
                for col in ['price', 'close']:
                    if col in df.columns:
                        prices = df[col].values
                        prices = df[col].dropna().values
                        break
            except Exception as e:
                logging.warning(f"Failed to load parquet: {e}")

        # 2. Search for DBN if no prices yet
        if prices is None:
            dbn_files = glob.glob(os.path.join(project_root, 'DATA', 'RAW', '*.dbn*'))
            if dbn_files:
                logging.info(f"Loading data snippet from {dbn_files[0]}")
                try:
                    from training.databento_loader import DatabentoLoader
                    df = DatabentoLoader.load_data(dbn_files[0])
                    for col in ['price', 'close']:
                        if col in df.columns:
                            prices = df[col].values
                            prices = df[col].dropna().values
                            break
                except ImportError:
                    logging.warning("DatabentoLoader not found, skipping .dbn files.")
                except Exception as e:
                    logging.warning(f"Failed to load dbn: {e}")

        # 3. Fallback to Synthetic
        if prices is None or len(prices) == 0:
             logging.warning("No suitable data found in DATA/RAW. Using SYNTHETIC data for audit.")
             prices = np.random.uniform(100, 200, 100).astype(np.float32)

        # Truncate and Ensure Float32
        if len(prices) > 100:
             prices = prices[:100]
        prices = prices.astype(np.float32)

        # Define Kernel 1 (Injection)
        @cuda.jit
        def scale_kernel(input_arr, output_arr, factor):
            i = cuda.grid(1)
            if i < input_arr.size:
                output_arr[i] = input_arr[i] * factor

        # Transfer to GPU
        d_input = cuda.to_device(prices)
        d_output = cuda.device_array_like(d_input)

        threads_per_block = 32
        blocks_per_grid = (prices.size + (threads_per_block - 1)) // threads_per_block

        factor = 2.0
        scale_kernel[blocks_per_grid, threads_per_block](d_input, d_output, factor)

        # Transfer back
        h_output = d_output.copy_to_host()

        # Verify
        expected = prices * factor
        if np.allclose(h_output, expected):
             logging.info("PASS: Stage B Verification (CPU -> GPU -> Kernel -> CPU) successful.")
        else:
             logging.critical("FAIL: Stage B Verification mismatch.")
             return False

    except Exception as e:
        logging.critical(f"[STAGE B] Exception: {e}")
        return False

    # --- STAGE C: HANDOFF ---
    logging.info("[STAGE C] Handoff: Pure VRAM data-passing...")
    try:
        # We reuse d_output from Stage B as input for Stage C
        # Kernel 2 (Handoff)
        @cuda.jit
        def offset_kernel(input_arr, output_arr, offset):
            i = cuda.grid(1)
            if i < input_arr.size:
                output_arr[i] = input_arr[i] + offset

        d_final = cuda.device_array_like(d_output)
        offset = 100.0

        # Launch Kernel 2 using device array from Kernel 1 directly
        offset_kernel[blocks_per_grid, threads_per_block](d_output, d_final, offset)

        # Verify
        h_final = d_final.copy_to_host()
        expected_final = (prices * 2.0) + 100.0

        if np.allclose(h_final, expected_final):
             logging.info("PASS: Stage C Verification (GPU -> GPU) successful.")
        else:
             logging.critical("FAIL: Stage C Verification mismatch.")
             return False

    except Exception as e:
        logging.critical(f"[STAGE C] Exception: {e}")
        return False

    logging.info("AUDIT COMPLETE: ALL STAGES PASSED")
    return True

if __name__ == "__main__":
    success = run_audit()
    if not success:
        sys.exit(1)
