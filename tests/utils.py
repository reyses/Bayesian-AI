import pandas as pd
import os
import sys
import glob
from training.databento_loader import DatabentoLoader

def find_test_data_file(filename):
    """
    Finds a test data file.
    It first checks in DATA/RAW, and if DATA/RAW is empty, it checks in tests/Testing DATA.
    """
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    raw_data_dir = os.path.join(project_root, 'DATA', 'RAW')
    
    # Prefer DATA/RAW if it has any files
    if os.path.exists(raw_data_dir) and os.listdir(raw_data_dir):
        file_path = os.path.join(raw_data_dir, filename)
        if os.path.exists(file_path):
            return file_path

    # Fallback to tests/Testing DATA
    testing_data_path = os.path.join(project_root, 'tests', 'Testing DATA', filename)
    if os.path.exists(testing_data_path):
        return testing_data_path
        
    return None

def get_test_data_files():
    """
    Returns a list of paths to test data files.
    It first checks in DATA/RAW, then in tests/Testing DATA.
    """
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    raw_data_dir = os.path.join(project_root, 'DATA', 'RAW')
    testing_data_dir = os.path.join(project_root, 'tests', 'Testing DATA')

    data_files = []
    if os.path.exists(raw_data_dir) and os.listdir(raw_data_dir):
        data_files.extend(glob.glob(os.path.join(raw_data_dir, "*.dbn*")))
        data_files.extend(glob.glob(os.path.join(raw_data_dir, "*.parquet")))

    if not data_files and os.path.exists(testing_data_dir):
        data_files.extend(glob.glob(os.path.join(testing_data_dir, "*.dbn*")))
        data_files.extend(glob.glob(os.path.join(testing_data_dir, "*.parquet")))
        
    return data_files

def load_test_data():
    """
    Loads the sample data from glbx-mdp3-20251230-20260129.ohlcv-1s.dbn.zst.
    Returns a DataFrame with datetime index.
    """
    data_path = find_test_data_file('glbx-mdp3-20251230-20260129.ohlcv-1s.dbn.zst')

    if not data_path or not os.path.exists(data_path):
        raise FileNotFoundError(f"Test data not found in DATA/RAW or tests/Testing DATA")

    df = DatabentoLoader.load_data(data_path)

    # Ensure datetime is the index
    if 'timestamp' in df.columns and not isinstance(df.index, pd.DatetimeIndex):
        df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
        df.set_index('datetime', inplace=True)
    elif 'datetime' in df.columns:
        df['datetime'] = pd.to_datetime(df['datetime'])
        df.set_index('datetime', inplace=True)

    return df
