#!/usr/bin/env python3
"""
Manifest Integrity Checker
Validates that all files listed in config/workflow_manifest.json exist.
"""
import os
import sys
import json

MANIFEST_PATH = 'config/workflow_manifest.json'

def check_manifest():
    if not os.path.exists(MANIFEST_PATH):
        print(f"ERROR: Manifest file not found at {MANIFEST_PATH}")
        return 1

    try:
        with open(MANIFEST_PATH, 'r') as f:
            manifest = json.load(f)
    except json.JSONDecodeError as e:
        print(f"ERROR: Failed to parse manifest: {e}")
        return 1
    except Exception as e:
        print(f"ERROR: Failed to read manifest: {e}")
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
            if not isinstance(filepath, str):
                print(f"WARNING: Layer '{layer}' in manifest does not have a string filepath. Skipping.")
                continue
            if not os.path.exists(filepath):
                missing_files.append(f"Layer [{layer}]: {filepath}")

    # Check 'resources' section (key-value pairs)
    if 'resources' in manifest:
        for resource, path in manifest['resources'].items():
            if not isinstance(path, str):
                print(f"WARNING: Resource '{resource}' in manifest does not have a string path. Skipping.")
                continue

            # Auto-create directory resources if missing (e.g., DATA/RAW)
            if not os.path.exists(path):
                # If it looks like a directory (no extension), try creating it
                if '.' not in os.path.basename(path):
                    try:
                        print(f"INFO: Creating missing resource directory: {path}")
                        os.makedirs(path, exist_ok=True)
                    except OSError as e:
                        print(f"WARNING: Failed to create resource directory {path}: {e}")
                        missing_files.append(f"Resource [{resource}]: {path}")
                else:
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
