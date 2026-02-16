"""
Script to reproduce loader errors in DatabentoLoader.
Usage: python scripts/reproduce_loader_error.py
"""
import os
import sys

# Add project root to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from training.databento_loader import DatabentoLoader

filepath = os.path.join(project_root, "tests", "Testing DATA", "glbx-mdp3-20250730.trades.0000.dbn.zst")

try:
    print(f"Loading using DatabentoLoader from: {filepath}")
    df = DatabentoLoader.load_data(filepath)
    print("Success!")
    print(df.head())
except Exception as e:
    print(f"Caught exception: {e}")
    import traceback
    traceback.print_exc()
