import databento as db
import pandas as pd
import sys

filepath = "tests/Testing DATA/glbx-mdp3-20250730.trades.0000.dbn.zst"

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
