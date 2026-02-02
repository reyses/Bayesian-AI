
import os
import databento as db
from tests.utils import find_test_data_file

try:
    path = find_test_data_file('glbx-mdp3-20251230-20260129.ohlcv-1s.dbn.zst')
    if not path:
        print("File not found")
    else:
        print(f"File found at: {path}")
        data = db.DBNStore.from_file(path)
        df = data.to_df()
        print("Columns:", df.columns.tolist())
except Exception as e:
    print(e)
