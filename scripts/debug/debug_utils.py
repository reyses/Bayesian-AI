import os
import glob

def find_test_data_file(filename):
    project_root = os.getcwd() # Assuming running from root
    raw_data_dir = os.path.join(project_root, 'DATA', 'RAW')
    testing_data_dir = os.path.join(project_root, 'tests', 'Testing DATA')

    print(f"Checking {raw_data_dir}")
    if os.path.exists(raw_data_dir) and os.listdir(raw_data_dir):
        print("DATA/RAW has files")
        file_path = os.path.join(raw_data_dir, filename)
        if os.path.exists(file_path):
            return file_path
    else:
        print("DATA/RAW empty or missing")

    print(f"Checking {testing_data_dir}")
    testing_data_path = os.path.join(testing_data_dir, filename)
    if os.path.exists(testing_data_path):
        return testing_data_path

    return None

filename = 'glbx-mdp3-20250730.trades.0000.dbn.zst'
found = find_test_data_file(filename)
print(f"Found: {found}")
