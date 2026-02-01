import os
import sys
import pathlib

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config.settings import OPERATIONAL_MODE, RAW_DATA_PATH

def check_data_availability():
    """Check: Perform a pathlib check to confirm required .parquet files exist in DATA/RAW when MODE == LEARNING."""
    print(f"Checking data availability for mode: {OPERATIONAL_MODE}")

    if OPERATIONAL_MODE == "LEARNING":
        raw_path = pathlib.Path(RAW_DATA_PATH)
        if not raw_path.exists():
            print(f"FAIL: {RAW_DATA_PATH} does not exist.")
            sys.exit(1)

        required_files = ["ohlcv-1s.parquet", "trades.parquet"]
        missing = []
        for f in required_files:
            file_path = raw_path / f
            if not file_path.exists():
                missing.append(f)
            else:
                print(f"OK: Found {f}")

        if missing:
            print(f"FAIL: Missing files: {missing}")
            sys.exit(1)
        else:
            print("PASS: All required files found.")
            sys.exit(0)
    else:
        print("SKIP: Not in LEARNING mode.")
        sys.exit(0)

if __name__ == "__main__":
    check_data_availability()
