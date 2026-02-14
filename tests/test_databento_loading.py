"""
Bayesian-AI - Databento Loading Test
Tests data loading from Databento files.
"""
import pytest
import pandas as pd
import os
import sys

# Ensure project root is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from training.databento_loader import DatabentoLoader
from tests.utils import find_test_data_file

# List of test files
TEST_FILES = [
    'glbx-mdp3-20250730.trades.0000.dbn.zst'
]

class TestDatabentoLoading:
    @pytest.mark.parametrize("filename", TEST_FILES)
    def test_load_real_databento_file(self, filename):
        """
        Tests loading of a real .dbn.zst file to ensure the loader can
        handle actual data correctly. This is an integration test.
        """
        file_path = find_test_data_file(filename)

        # Check if the file exists before running the test
        if not file_path or not os.path.exists(file_path):
            pytest.skip(f"Test data file {filename} not found in DATA/RAW or tests/Testing DATA")

        # Load the data
        df = DatabentoLoader.load_data(file_path)

        # 1. Check if the DataFrame is not empty
        assert not df.empty, "The loaded DataFrame should not be empty."

        # 2. Check for required columns
        required_cols = ['timestamp', 'price', 'volume', 'type']
        for col in required_cols:
            assert col in df.columns, f"DataFrame should have column '{col}'"

        # 3. Check data types
        assert pd.api.types.is_float_dtype(df['timestamp']), "'timestamp' column should be float"
        assert pd.api.types.is_float_dtype(df['price']), "'price' column should be float"
        assert pd.api.types.is_integer_dtype(df['volume']), "'volume' column should be integer"
        assert pd.api.types.is_string_dtype(df['type']), "'type' column should be object/string"

        # 4. Check for trade filtering (assuming default behavior)
        # The 'type' column should only contain 'T' if filtering is working
        assert (df['type'] == 'T').all(), "All records should be of type 'T' (trade)"

        # 5. Check for a reasonable number of rows (optional, but good for a sanity check)
        assert len(df) > 100, "Should have a reasonable number of rows"
