"""
Script to inspect Databento .dbn or .data files.
Usage: python scripts/inspect_databento.py
"""
import databento as db
import pandas as pd
import sys
import os

# Add project root to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

# Correct path resolution
filepath = os.path.join(project_root, "tests", "Testing DATA", "glbx-mdp3-20250730.trades.0000.dbn.zst")

try:
    print(f"Inspecting file: {filepath}")

    # Try loading directly
    store = db.DBNStore.from_file(filepath)
    print("DBNStore loaded.")

    # Try iterator first to see one record
    # record = next(store)
    # print(f"First record: {record}")

    # Try to_df
    df = store.to_df()
    print("DataFrame columns:", df.columns)
    print("DataFrame head:\n", df.head())

except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
