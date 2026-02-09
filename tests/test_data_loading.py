import os
import shutil
import unittest
import glob
import pandas as pd
from training.orchestrator import load_data_from_directory

class TestDashboardDataLoading(unittest.TestCase):
    def setUp(self):
        self.test_dir = "tests/temp_data_loading_test"
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
        os.makedirs(self.test_dir)
        self.addCleanup(shutil.rmtree, self.test_dir, ignore_errors=True)

        # Copy a small dbn file from DATA/RAW if available
        self.raw_dir = "DATA/RAW"
        dbn_files = glob.glob(os.path.join(self.raw_dir, "*.dbn.zst"))

        if dbn_files:
            # Sort by size to pick smallest
            dbn_files.sort(key=lambda x: os.path.getsize(x))
            self.test_dbn = dbn_files[0]
            shutil.copy(self.test_dbn, self.test_dir)
            print(f"Copied smallest file: {self.test_dbn} ({os.path.getsize(self.test_dbn)/1024:.1f} KB) to {self.test_dir}")
            self.has_dbn = True
        else:
            print("WARNING: No .dbn.zst files found in DATA/RAW. Skipping conversion test.")
            self.has_dbn = False

    def test_load_data_creates_parquet(self):
        if not self.has_dbn:
            return

        print("\n--- Running load_data_from_directory ---")
        files = load_data_from_directory(self.test_dir)

        # Check return type
        self.assertIsInstance(files, list)
        self.assertTrue(len(files) > 0)
        print(f"Returned files: {files}")

        self.assertTrue(files[0].endswith(".parquet"))
        self.assertTrue(os.path.exists(files[0]))

        # Check if we can load it with pandas
        try:
            df = pd.read_parquet(files[0])
            self.assertIsInstance(df, pd.DataFrame)
            print(f"Successfully loaded DataFrame. Shape: {df.shape}")
            self.assertTrue('timestamp' in df.columns)
        except Exception as e:
            self.fail(f"Failed to load parquet file: {e}")

        # Run again, should return parquet immediately (cached check)
        print("\n--- Running load_data_from_directory (cached) ---")
        files2 = load_data_from_directory(self.test_dir)
        self.assertEqual(files, files2)

if __name__ == "__main__":
    unittest.main()
