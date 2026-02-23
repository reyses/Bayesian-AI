import argparse
import logging
import databento as db
import pandas as pd
import sys
import traceback

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Debug Databento DBN file loading.")
    parser.add_argument(
        "--file",
        type=str,
        required=True,
        help="Path to the .dbn.zst file to inspect."
    )
    args = parser.parse_args()

    filepath = args.file

    try:
        logger.info(f"Inspecting file: {filepath}")
        # Try loading directly
        store = db.DBNStore.from_file(filepath)
        logger.info("DBNStore loaded.")

        # Try to_df
        df = store.to_df()
        logger.info(f"DataFrame columns: {df.columns}")
        logger.info(f"DataFrame head:\n{df.head()}")

    except Exception as e:
        logger.error(f"Error: {e}")
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    main()
