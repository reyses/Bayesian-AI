import argparse
import logging
import sys
import os
import traceback

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from training.databento_loader import DatabentoLoader

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Verify DatabentoLoader functionality.")
    parser.add_argument(
        "filepath",
        nargs='?',
        default="DATA/RAW/trades.parquet",
        help="Path to the databento file to load (default: DATA/RAW/trades.parquet)"
    )
    args = parser.parse_args()

    filepath = args.filepath

    try:
        logger.info(f"Loading using DatabentoLoader from: {filepath}")
        df = DatabentoLoader.load_data(filepath)
        logger.info("Success!")
        logger.info(f"DataFrame Head:\n{df.head()}")
    except Exception as e:
        logger.error(f"Caught exception: {e}")
        logger.debug(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main()
