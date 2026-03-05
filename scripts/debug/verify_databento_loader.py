from training.databento_loader import DatabentoLoader
import os
import argparse
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Verify DatabentoLoader functionality.")
    parser.add_argument(
        "--filepath",
        type=str,
        default="tests/Testing DATA/glbx-mdp3-20250730.trades.0000.dbn.zst",
        help="Path to the DBN file to load"
    )
    args = parser.parse_args()

    filepath = args.filepath

    try:
        logger.info(f"Loading using DatabentoLoader from: {filepath}")
        df = DatabentoLoader.load_data(filepath)
        logger.info("Success!")
        logger.info(f"\n{df.head()}")
    except Exception as e:
        logger.error(f"Caught exception: {e}", exc_info=True)

if __name__ == "__main__":
    main()
