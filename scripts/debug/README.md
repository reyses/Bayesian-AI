# Debug Scripts

This directory contains verification and debug scripts to ensure the integrity of the Bayesian-AI system.

## Scripts

### `verify_engine_resilience.py`
**Purpose:** Verifies that the `QuantumFieldEngine` can handle input DataFrames with missing columns (e.g., missing 'open') without crashing.
**Usage:** `python scripts/debug/verify_engine_resilience.py`

### `verify_hypervolume.py`
**Purpose:** Runs a truncated end-to-end training and forward pass cycle to verify the Hypervolume Clustering pipeline (`hypervolume_tree.pkl`) and ensure no regression in critical outputs like `oracle_trade_log.csv`.
**Usage:** `python scripts/debug/verify_hypervolume.py --data-dir DATA/ATLAS_1MONTH --checkpoint-dir checkpoints_verify`

### `verify_databento_loader.py`
**Purpose:** Verifies that the `DatabentoLoader` can correctly load and parse `.dbn.zst` files.
**Usage:** `python scripts/debug/verify_databento_loader.py --file <path_to_file.dbn.zst>`

### `debug_utils.py`
**Purpose:** Shared utility functions for debug scripts (e.g., locating test data).

## Guidelines
- Use `logging` instead of `print` for script output.
- Use `argparse` for handling command-line arguments.
- Ensure scripts return non-zero exit codes on failure.
