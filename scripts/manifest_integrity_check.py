"""
Manifest Integrity Checker
Validates that all files listed in config/workflow_manifest.json exist.
"""
import os
import sys
import json

def check_manifest():
MANIFEST_PATH = 'config/workflow_manifest.json'

def check_manifest():
    if not os.path.exists(MANIFEST_PATH):
    if not os.path.exists(manifest_path):
        print(f"ERROR: Manifest file not found at {manifest_path}")
        return 1

    try:
    try:
        with open(MANIFEST_PATH, 'r') as f:
    except json.JSONDecodeError as e:
        print(f"ERROR: Failed to parse manifest: {e}")
        return 1

    missing_files = []

    # Check 'pipeline' section (lists of files)
    if 'pipeline' in manifest:
        for stage, files_list in manifest['pipeline'].items():
            if not isinstance(files_list, list):
                print(f"WARNING: Pipeline stage '{stage}' in manifest does not contain a list of files. Skipping.")
                continue
            for filepath in files_list:
                if not os.path.exists(filepath):
                    missing_files.append(f"Pipeline [{stage}]: {filepath}")

    # Check 'layers' section (key-value pairs)
    if 'layers' in manifest:
        for layer, filepath in manifest['layers'].items():
            if not os.path.exists(filepath):
                missing_files.append(f"Layer [{layer}]: {filepath}")

    # Check 'resources' section (key-value pairs)
    if 'resources' in manifest:
        for resource, path in manifest['resources'].items():
            if not isinstance(path, str):
                print(f"WARNING: Resource '{resource}' in manifest does not have a string path. Skipping.")
                continue
            if not os.path.exists(path):
                missing_files.append(f"Resource [{resource}]: {path}")

    if missing_files:
        print("FAIL: The following files are missing from the manifest:")
        for missing in missing_files:
            print(f"  - {missing}")
        return 1

    print("PASS: All manifest files exist.")
    return 0

if __name__ == "__main__":
    sys.exit(check_manifest())
