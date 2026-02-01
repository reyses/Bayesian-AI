import json
import os
import sys
import importlib

# Ensure root directory is in python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def run_test():
    print("Algorithm V2 - Topic Build Verification")
    print("=======================================")

    # Load Manifest
    manifest_path = 'config/workflow_manifest.json'
    if not os.path.exists(manifest_path):
        print(f"FAIL: {manifest_path} not found")
        sys.exit(1)

    with open(manifest_path, 'r') as f:
        manifest = json.load(f)

    print(f"Project: {manifest.get('project')}")
    if manifest.get('project') != "Algorithm V2":
         print(f"FAIL: Project name mismatch. Expected 'Algorithm V2', got '{manifest.get('project')}'")
         sys.exit(1)

    # Collect files
    files_to_check = set()
    modules_to_check = set()

    for stage, file_list in manifest.get('pipeline', {}).items():
        for fpath in file_list:
            files_to_check.add(fpath)
            # Convert path to module format (e.g., core/state_vector.py -> core.state_vector)
            if fpath.endswith('.py'):
                module_name = fpath[:-3].replace('/', '.')
                modules_to_check.add(module_name)

    for layer, fpath in manifest.get('layers', {}).items():
        files_to_check.add(fpath)
        if fpath.endswith('.py'):
             module_name = fpath[:-3].replace('/', '.')
             modules_to_check.add(module_name)

    # Verify Files
    missing_files = []
    for fpath in files_to_check:
        if not os.path.exists(fpath):
            missing_files.append(fpath)

    if missing_files:
        print(f"FAIL: Missing files: {missing_files}")
        sys.exit(1)
    else:
        print(f"PASS: All {len(files_to_check)} manifest files exist.")

    # Verify Imports
    failed_imports = []
    # Add config.symbols as it is important
    modules_to_check.add('config.symbols')
    modules_to_check.add('config.settings')

    for module in modules_to_check:
        try:
            importlib.import_module(module)
        except Exception as e:
            print(f"ERROR importing {module}: {e}")
            failed_imports.append(module)

    if failed_imports:
        print(f"FAIL: Failed to import modules: {failed_imports}")
        sys.exit(1)
    else:
        print(f"PASS: All {len(modules_to_check)} modules imported successfully.")

    # Verify OPERATIONAL_MODE accessibility
    try:
        from config.settings import OPERATIONAL_MODE
        if OPERATIONAL_MODE not in ["LEARNING", "EXECUTE"]:
             print(f"FAIL: Invalid OPERATIONAL_MODE: {OPERATIONAL_MODE}")
             sys.exit(1)
        print(f"PASS: OPERATIONAL_MODE is valid: {OPERATIONAL_MODE}")
    except ImportError:
        print("FAIL: Could not import OPERATIONAL_MODE from config.settings")
        sys.exit(1)

    print("BUILD INTEGRITY CHECK PASSED")
    sys.exit(0)

if __name__ == "__main__":
    run_test()
