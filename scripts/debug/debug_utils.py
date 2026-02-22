import os
import glob
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def find_test_data_file(filename):
    project_root = os.getcwd() # Assuming running from root
    raw_data_dir = os.path.join(project_root, 'DATA', 'RAW')
    testing_data_dir = os.path.join(project_root, 'tests', 'Testing DATA')

    logger.debug(f"Checking {raw_data_dir}")
    if os.path.exists(raw_data_dir) and os.listdir(raw_data_dir):
        logger.debug("DATA/RAW has files")
        file_path = os.path.join(raw_data_dir, filename)
        if os.path.exists(file_path):
            return file_path
    else:
        logger.debug("DATA/RAW empty or missing")

    logger.debug(f"Checking {testing_data_dir}")
    testing_data_path = os.path.join(testing_data_dir, filename)
    if os.path.exists(testing_data_path):
        return testing_data_path

    return None

if __name__ == "__main__":
    # Example usage
    filename = 'trades.parquet'
    found = find_test_data_file(filename)
    if found:
        logger.info(f"Found: {found}")
    else:
        logger.warning(f"File {filename} not found.")
