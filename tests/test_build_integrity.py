"""
Bayesian-AI - Build Integrity Test
Tests build integrity and configuration.
"""
import json
import os
import sys
import importlib
import pytest

# Ensure root directory is in python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def test_manifest_structure():
    """Verify manifest file structure and project name."""
    manifest_path = 'config/workflow_manifest.json'
    assert os.path.exists(manifest_path), f"{manifest_path} not found"

    with open(manifest_path, 'r') as f:
        manifest = json.load(f)

    assert manifest.get('project') == "Bayesian-AI", \
        f"Project name mismatch. Expected 'Bayesian-AI', got '{manifest.get('project')}'"

def test_file_existence():
    """Verify all files listed in manifest exist."""
    manifest_path = 'config/workflow_manifest.json'
    if not os.path.exists(manifest_path):
        pytest.skip("Manifest not found")

    with open(manifest_path, 'r') as f:
        manifest = json.load(f)

    files_to_check = set()
    for stage, file_list in manifest.get('pipeline', {}).items():
        for fpath in file_list:
            files_to_check.add(fpath)

    for layer, fpath in manifest.get('layers', {}).items():
        files_to_check.add(fpath)

    missing_files = []
    for fpath in files_to_check:
        if not os.path.exists(fpath):
            missing_files.append(fpath)

    assert not missing_files, f"Missing files: {missing_files}"

def test_module_imports():
    """Verify all python modules listed in manifest can be imported."""
    manifest_path = 'config/workflow_manifest.json'
    if not os.path.exists(manifest_path):
        pytest.skip("Manifest not found")

    with open(manifest_path, 'r') as f:
        manifest = json.load(f)

    modules_to_check = set()

    # Helper to convert path to module
    def path_to_module(path):
        if path.endswith('.py'):
            return path[:-3].replace('/', '.')
        return None

    for stage, file_list in manifest.get('pipeline', {}).items():
        for fpath in file_list:
            mod = path_to_module(fpath)
            if mod: modules_to_check.add(mod)

    for layer, fpath in manifest.get('layers', {}).items():
        mod = path_to_module(fpath)
        if mod: modules_to_check.add(mod)

    # Add critical config modules
    modules_to_check.add('config.symbols')
    modules_to_check.add('config.settings')

    failed_imports = []
    for module in modules_to_check:
        try:
            importlib.import_module(module)
        except Exception as e:
            # Check if it's just a cuda missing error (allowable in CI/non-gpu envs if handled)
            # But for build integrity, imports should generally succeed or handle ImportError internally
            failed_imports.append((module, str(e)))

    assert not failed_imports, f"Failed to import modules: {failed_imports}"

def test_operational_mode():
    """Verify OPERATIONAL_MODE configuration."""
    try:
        from config.settings import OPERATIONAL_MODE
        assert OPERATIONAL_MODE in ["LEARNING", "EXECUTE"], \
            f"Invalid OPERATIONAL_MODE: {OPERATIONAL_MODE}"
    except ImportError:
        pytest.fail("Could not import OPERATIONAL_MODE from config.settings")

if __name__ == "__main__":
    pytest.main([__file__])
