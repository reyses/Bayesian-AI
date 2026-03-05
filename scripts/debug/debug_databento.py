import databento as db
import pandas as pd
import sys
import argparse
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Inspect Databento DBN files.")
    parser.add_argument(
        "--filepath",
        type=str,
        default="tests/Testing DATA/glbx-mdp3-20250730.trades.0000.dbn.zst",
        help="Path to the DBN file to inspect"
    )
    args = parser.parse_args()

    filepath = args.filepath

    try:
        logger.info(f"Inspecting file: {filepath}")
        # Try loading directly
        store = db.DBNStore.from_file(filepath)
        logger.info("DBNStore loaded.")

        # Try iterator first to see one record
        # record = next(store)
        # logger.info(f"First record: {record}")

        # Try to_df
        df = store.to_df()
        logger.info(f"DataFrame columns: {df.columns}")
        logger.info(f"DataFrame head:\n{df.head()}")

    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)

if __name__ == "__main__":
    main()
