import argparse
import databento as db
import pandas as pd
import sys
import logging
import traceback

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Inspect a Databento file directly.")
    parser.add_argument(
        "filepath",
        nargs='?',
        default="DATA/RAW/trades.parquet",
        help="Path to the databento file to inspect (default: DATA/RAW/trades.parquet)"
    )
    args = parser.parse_args()

    filepath = args.filepath

    try:
        logger.info(f"Inspecting file: {filepath}")

        # Determine if it's a DBN file or Parquet
        if filepath.endswith('.parquet'):
            logger.info("Loading as Parquet file...")
            df = pd.read_parquet(filepath)
        else:
            logger.info("Loading as DBN file...")
            store = db.DBNStore.from_file(filepath)
            logger.info("DBNStore loaded.")
            df = store.to_df()

        logger.info(f"DataFrame columns: {df.columns.tolist()}")
        logger.info(f"DataFrame head:\n{df.head()}")

    except Exception as e:
        logger.error(f"Error: {e}")
        logger.debug(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main()
