"""
Bayesian-AI - Test Data Setup & Validation
Prepares test data by copying files and running initial integrity checks.
"""
import sys
import os
import shutil
import subprocess
import pandas as pd

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from training.databento_loader import DatabentoLoader

def run_validation_tests():
    """Runs data integrity and velocity tests."""
    print("\nRunning Validation Tests...")

    tests = [
        "tests/test_real_data_velocity.py",
        "tests/test_databento_loading.py"
    ]

    for test in tests:
        print(f"Executing {test}...")
        result = subprocess.run([sys.executable, "-m", "pytest", test, "-v"], capture_output=True, text=True)
        if result.returncode != 0:
            print(f"FAIL: {test}")
            print(result.stdout)
            print(result.stderr)
            return False
        else:
            print(f"PASS: {test}")

    return True

def setup_test_data():
    print("Bayesian-AI - Setup Test Data")
    print("=============================")

    # Define paths
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    data_raw_dir = os.path.join(root_dir, 'DATA', 'RAW')
    tests_dir = os.path.join(root_dir, 'tests')

    dbn_filename = 'glbx-mdp3-20250101-20260209.ohlcv-1s.dbn.zst'
    source_dbn_path = os.path.join(root_dir, 'DATA', 'OHLCV_raw', dbn_filename) # Corrected path
    dest_dbn_path = os.path.join(data_raw_dir, dbn_filename)
    trades_parquet_path = os.path.join(data_raw_dir, 'trades.parquet')
    ohlcv_parquet_path = os.path.join(data_raw_dir, 'ohlcv-1s.parquet')

    # Create DATA/RAW directory
    os.makedirs(data_raw_dir, exist_ok=True)
    print(f"Created directory: {data_raw_dir}")

    # Copy DBN file
    if not os.path.exists(source_dbn_path):
        print(f"ERROR: Source file not found: {source_dbn_path}")
        sys.exit(1)

    shutil.copy2(source_dbn_path, dest_dbn_path)
    print(f"Copied {source_dbn_path} to {dest_dbn_path}")

    # Process data
    print(f"Loading {dest_dbn_path}...")
    try:
        # DatabentoLoader.load_data returns a DataFrame with 'timestamp' (float seconds), 'price', 'volume', 'type'
        df = DatabentoLoader.load_data(dest_dbn_path)
        print(f"Loaded {len(df)} rows.")

        print(f"Saving trades to {trades_parquet_path}...")
        df.to_parquet(trades_parquet_path)

        print("Generating OHLCV...")
        # Ensure timestamp is datetime for resampling
        # timestamp is in seconds (float) as per DatabentoLoader implementation
        df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
        df_idx = df.set_index('datetime')

        # Resample to 1s
        ohlcv = df_idx['price'].resample('1s').ohlc()
        ohlcv['volume'] = df_idx['volume'].resample('1s').sum()

        # Reset index to get timestamp column back
        ohlcv = ohlcv.reset_index()

        # Add timestamp as float seconds
        ohlcv['timestamp'] = ohlcv['datetime'].astype('int64') // 10**9

        print(f"Saving OHLCV to {ohlcv_parquet_path}...")
        ohlcv.to_parquet(ohlcv_parquet_path)

        print("Data setup complete.")

        # Run validation after setup
        if not run_validation_tests():
            print("WARNING: Validation tests failed after data setup.")
            sys.exit(1)

    except Exception as e:
        print(f"Error setting up data: {e}")
        # Print full traceback for debugging
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    setup_test_data()
