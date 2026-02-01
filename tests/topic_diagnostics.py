import sys
import os

# Ensure root directory is in python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from numba import cuda

def run_diagnostics():
    print("Algorithm V2 - Topic Diagnostics")
    print("================================")

    is_available = cuda.is_available()
    print(f"CUDA Available: {is_available}")

    if is_available:
        try:
            device = cuda.get_current_device()
            # device.name is bytes in some versions, string in others. Handle both.
            name = device.name
            if isinstance(name, bytes):
                name = name.decode('utf-8')

            print(f"GPU Detected: {name}")

            # Requirement says "RTX 3060 detection"
            if '3060' in name:
                print("SUCCESS: RTX 3060 Detected.")
            else:
                print(f"INFO: Detected GPU {name}")

        except Exception as e:
            print(f"WARNING: CUDA available but failed to get device info: {e}")
    else:
        print("WARNING: CUDA not available. System running in CPU fallback mode.")

    # Check Data Path for LEARNING mode
    try:
        from config.settings import OPERATIONAL_MODE, RAW_DATA_PATH
        import pathlib
        print(f"Operational Mode: {OPERATIONAL_MODE}")
        if OPERATIONAL_MODE == "LEARNING":
            print(f"Checking Data Path: {RAW_DATA_PATH}")
            path = pathlib.Path(RAW_DATA_PATH)
            if not path.exists():
                print(f"FAIL: {RAW_DATA_PATH} does not exist.")
            else:
                files = ["ohlcv-1s.parquet", "trades.parquet"]
                missing = []
                for f in files:
                    if not (path / f).exists():
                        missing.append(f)
                if missing:
                    print(f"FAIL: Missing files in {RAW_DATA_PATH}: {missing}")
                else:
                    print(f"PASS: Required files found in {RAW_DATA_PATH}")
    except ImportError:
        print("WARNING: Could not import config.settings")

    print("DIAGNOSTICS COMPLETE")
    sys.exit(0)

if __name__ == "__main__":
    run_diagnostics()
