
import unittest
import pandas as pd
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import tempfile
import shutil
import os
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from training.fractal_atlas_builder import AtlasBuilder

class TestFractalAtlas(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.input_file = Path(self.test_dir) / "test_input.parquet"
        self.output_dir = Path(self.test_dir) / "ATLAS"

        # Generate dummy data: 2 days of 1s data
        # Start: 2025-01-01 00:00:00
        # End:   2025-01-03 00:00:00 (exclusive)
        dates = pd.date_range(start="2025-01-01", periods=172800, freq="1s")

        n = len(dates)
        opens = np.random.randn(n) + 100
        highs = opens + np.abs(np.random.randn(n))
        lows = opens - np.abs(np.random.randn(n))
        closes = opens + np.random.randn(n) * 0.1
        volumes = np.random.randint(1, 100, size=n)

        # Timestamps as float seconds (unix)
        # Robust conversion: (dates - epoch) / 1s
        timestamps = (dates - pd.Timestamp("1970-01-01")) // pd.Timedelta("1s")

        df = pd.DataFrame({
            'timestamp': timestamps,
            'open': opens,
            'high': highs,
            'low': lows,
            'close': closes,
            'volume': volumes
        })

        # Save as parquet
        df.to_parquet(self.input_file)

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_atlas_generation(self):
        print(f"Testing Atlas Generation with input: {self.input_file}")

        builder = AtlasBuilder(self.input_file, self.output_dir)
        builder.run()

        # Verify output structure
        expected_resolutions = ['1s', '5s', '15s', '1m', '5m', '15m', '1h', '4h', '1D', '1W']

        for res in expected_resolutions:
            res_dir = self.output_dir / res
            self.assertTrue(res_dir.exists(), f"Missing directory for {res}")

            # Check for partition files
            files = list(res_dir.glob("*.parquet"))
            self.assertTrue(len(files) > 0, f"No parquet files found for {res}")

            # Verify content of the first file
            p = files[0]
            df = pd.read_parquet(p)
            self.assertFalse(df.empty, f"Output file empty for {res}")

            expected_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
            for col in expected_cols:
                self.assertIn(col, df.columns, f"Missing column {col} in {res}")

            # Basic sanity check on rows
            # 1s input has 172800 rows.
            # 1m output should have ~2880 rows.
            if res == '1m':
                # Sum of rows across all partition files
                total_rows = sum(pd.read_parquet(f).shape[0] for f in files)
                # Allow small deviation due to start/end alignment
                self.assertTrue(2800 <= total_rows <= 2885, f"Expected ~2880 rows for 1m, got {total_rows}")

if __name__ == '__main__':
    unittest.main()
