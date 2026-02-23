import argparse
import logging
import traceback
import sys
import os

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from training.databento_loader import DatabentoLoader

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Verify DatabentoLoader implementation.")
    parser.add_argument(
        "--file",
        type=str,
        required=True,
        help="Path to the .dbn.zst file to load."
    )
    args = parser.parse_args()

    filepath = args.file

    try:
        logger.info(f"Loading using DatabentoLoader from: {filepath}")
        df = DatabentoLoader.load_data(filepath)
        logger.info("Success!")
        logger.info(f"Head:\n{df.head()}")
    except Exception as e:
        logger.exception(f"Caught exception while loading data from {filepath}: {e}")

if __name__ == "__main__":
    main()
