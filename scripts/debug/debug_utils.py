import os
import argparse
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def find_test_data_file(filename):
    project_root = os.getcwd() # Assuming running from root

    # Check DATA/RAW
    raw_data_dir = os.path.join(project_root, 'DATA', 'RAW')
    logger.info(f"Checking {raw_data_dir}")
    if os.path.exists(raw_data_dir) and os.listdir(raw_data_dir):
        logger.info("DATA/RAW has files")
        file_path = os.path.join(raw_data_dir, filename)
        if os.path.exists(file_path):
            return file_path
    else:
        logger.warning("DATA/RAW empty or missing")

    # Check tests/Testing DATA
    testing_data_dir = os.path.join(project_root, 'tests', 'Testing DATA')
    logger.info(f"Checking {testing_data_dir}")
    testing_data_path = os.path.join(testing_data_dir, filename)
    if os.path.exists(testing_data_path):
        return testing_data_path

    return None

def main():
    parser = argparse.ArgumentParser(description="Find a test data file.")
    parser.add_argument("filename", nargs='?', default="ohlcv-1s.parquet", help="Filename to search for.")
    args = parser.parse_args()

    logger.info(f"Searching for: {args.filename}")
    found = find_test_data_file(args.filename)

    if found:
        logger.info(f"Found: {found}")
    else:
        logger.error(f"File not found: {args.filename}")

if __name__ == "__main__":
    main()
