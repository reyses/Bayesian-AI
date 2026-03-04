import argparse
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from training.databento_loader import DatabentoLoader
from core.logger import setup_logger

def main():
    parser = argparse.ArgumentParser(description="Verify loading logic for Databento.")
    parser.add_argument(
        "--filepath",
        type=str,
        default="tests/Testing DATA/glbx-mdp3-20250730.trades.0000.dbn.zst",
        help="Path to the test DBN file to attempt loading."
    )
    args = parser.parse_args()

    logger = setup_logger("verify_databento_loader", "debug_outputs/verify_databento_loader.log", console=True)

    try:
        logger.info(f"Loading using DatabentoLoader from: {args.filepath}")
        df = DatabentoLoader.load_data(args.filepath)
        logger.info("Success!")
        logger.info(f"\n{df.head()}")
    except Exception as e:
        logger.error(f"Caught exception: {e}", exc_info=True)

if __name__ == "__main__":
    main()
