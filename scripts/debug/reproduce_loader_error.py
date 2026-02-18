from training.databento_loader import DatabentoLoader
import os

filepath = "tests/Testing DATA/glbx-mdp3-20250730.trades.0000.dbn.zst"

try:
    print(f"Loading using DatabentoLoader from: {filepath}")
    df = DatabentoLoader.load_data(filepath)
    print("Success!")
    print(df.head())
except Exception as e:
    print(f"Caught exception: {e}")
    import traceback
    traceback.print_exc()
