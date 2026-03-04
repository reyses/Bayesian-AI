import argparse
import os
import glob
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from core.logger import setup_logger

def find_test_data_file(filename, logger):
    project_root = os.getcwd() # Assuming running from root
    raw_data_dir = os.path.join(project_root, 'DATA', 'RAW')
    testing_data_dir = os.path.join(project_root, 'tests', 'Testing DATA')

    logger.info(f"Checking {raw_data_dir}")
    if os.path.exists(raw_data_dir) and os.listdir(raw_data_dir):
        logger.info("DATA/RAW has files")
        file_path = os.path.join(raw_data_dir, filename)
        if os.path.exists(file_path):
            return file_path
    else:
        logger.warning("DATA/RAW empty or missing")

    logger.info(f"Checking {testing_data_dir}")
    testing_data_path = os.path.join(testing_data_dir, filename)
    if os.path.exists(testing_data_path):
        return testing_data_path

    return None

def main():
    parser = argparse.ArgumentParser(description="Find test data files in common directories.")
    parser.add_argument(
        "--filename",
        type=str,
        default='glbx-mdp3-20250730.trades.0000.dbn.zst',
        help="Name of the file to search for."
    )
    args = parser.parse_args()

    logger = setup_logger("debug_utils", "debug_outputs/debug_utils.log", console=True)

    found = find_test_data_file(args.filename, logger)
    if found:
        logger.info(f"Found: {found}")
    else:
        logger.error(f"File {args.filename} not found.")

if __name__ == "__main__":
    main()
