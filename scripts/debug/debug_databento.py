import argparse
import databento as db
import pandas as pd
import sys
import logging
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from core.logger import setup_logger

def main():
    parser = argparse.ArgumentParser(description="Debug Databento data loading.")
    parser.add_argument(
        "--filepath",
        type=str,
        default="tests/Testing DATA/glbx-mdp3-20250730.trades.0000.dbn.zst",
        help="Path to the DBN file to inspect."
    )
    args = parser.parse_args()

    logger = setup_logger("debug_databento", "debug_outputs/debug_databento.log", console=True)

    try:
        logger.info(f"Inspecting file: {args.filepath}")
        # Try loading directly
        store = db.DBNStore.from_file(args.filepath)
        logger.info("DBNStore loaded.")

        # Try to_df
        df = store.to_df()
        logger.info(f"DataFrame columns: {df.columns}")
        logger.info(f"DataFrame head:\n{df.head()}")

    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)

if __name__ == "__main__":
    main()
