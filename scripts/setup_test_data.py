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
from training.orchestrator import TIMEFRAME_MAP
import numpy as np

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

    dbn_filename = 'glbx-mdp3-20250730.trades.0000.dbn.zst'

    # Check possible locations for source file
    source_locations = [
        os.path.join(tests_dir, dbn_filename),
        os.path.join(tests_dir, 'Testing DATA', dbn_filename)
    ]

    source_dbn_path = None
    for loc in source_locations:
        if os.path.exists(loc):
            source_dbn_path = loc
            break

    if not source_dbn_path:
        # Default to tests/ for error message if not found anywhere
        source_dbn_path = os.path.join(tests_dir, dbn_filename)

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

        # Generate additional resampled files based on TIMEFRAME_MAP
        _generate_resampled_data(ohlcv, data_raw_dir)

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

def _generate_resampled_data(ohlcv_df, output_dir):
    """Helper to generate resampled parquet files for all orchestrator timeframes."""
    print("\nGenerating resampled data for TIMEFRAME_MAP intervals...")

    # Use the 1s OHLCV as base input (matching Orchestrator logic)
    base_df = ohlcv_df.copy()

    # Ensure timestamp is datetime
    if 'timestamp' in base_df.columns and not pd.api.types.is_datetime64_any_dtype(base_df['timestamp']):
            base_df['timestamp'] = pd.to_datetime(base_df['timestamp'], unit='s')

    # Add _1s_idx (global index)
    base_df['_1s_idx'] = np.arange(len(base_df))

    base_df = base_df.set_index('timestamp')

    for interval_name in sorted(list(set(TIMEFRAME_MAP.values()))):
        if interval_name == '1s': # Skip base
            continue

        print(f"  Resampling to {interval_name}...", end='', flush=True)

        # Pandas 2.2+ deprecation fix: 'm' -> 'min'
        pd_freq = interval_name
        if pd_freq.endswith('m') and not pd_freq.endswith('min'):
            pd_freq = pd_freq.replace('m', 'min')

        # Aggregation logic matching orchestrator
        agg_dict = {
            '_1s_idx': 'last',
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }

        # Resample
        # Handle potential missing columns if ohlcv structure changes
        valid_agg = {k: v for k, v in agg_dict.items() if k in base_df.columns or k == 'close'}
        # close is guaranteed by ohlcv.ohlc()

        resampled = base_df.resample(pd_freq).agg(valid_agg).dropna(subset=['close'])

        # Reset index to get timestamp back
        resampled = resampled.reset_index()

        # Convert timestamp back to float seconds
        # Ensure datetime64[ns] before casting to int64 for consistent division
        if not pd.api.types.is_datetime64_ns_dtype(resampled['timestamp']):
            resampled['timestamp'] = resampled['timestamp'].astype('datetime64[ns]')

        resampled['timestamp'] = resampled['timestamp'].astype('int64') // 10**9

        # Save
        filename = f'ohlcv-{interval_name}.parquet'
        filepath = os.path.join(output_dir, filename)
        resampled.to_parquet(filepath)
        print(f" Saved to {filename}")

if __name__ == "__main__":
    setup_test_data()
