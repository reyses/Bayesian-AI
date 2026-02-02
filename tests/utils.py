import pandas as pd
import os
import sys

def load_test_data():
    """
    Loads the sample data from DATA/RAW/ohlcv-1s.parquet.
    Returns a DataFrame with datetime index.
    """
    # Determine the project root
    # Assuming this file is in tests/utils.py, root is ..
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, '..'))

    data_path = os.path.join(project_root, 'DATA', 'RAW', 'ohlcv-1s.parquet')

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Test data not found at {data_path}")

    df = pd.read_parquet(data_path)

    # Ensure datetime is the index
    if 'datetime' in df.columns:
        df['datetime'] = pd.to_datetime(df['datetime'])
        df.set_index('datetime', inplace=True)

    return df
