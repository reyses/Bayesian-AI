# Agent Instructions

This repository contains "Bayesian-AI", an algorithmic trading system.

## 1. Core Directives

*   **Single Source of Truth**: `requirements.txt` is the ONLY allowed file for defining project dependencies. Do not create separate requirements files (e.g., `requirements_dashboard.txt`, `requirements_notebook.txt`).
*   **Dependency Pinning**: Critical dependencies should be pinned to specific versions in `requirements.txt` to ensure build reproducibility and prevent unexpected breakages from upstream changes.
*   **Notebook Limit**: Only one notebook should be created, unless the User specifies that an additional one is needed.
*   **Artifacts**: Do not modify files in `dist/` or `build/` directly. Edit the source code and rebuild.

## 2. Development Workflow

To run the workflow locally, ensure you have Python 3.10+ installed.

### Local Setup
1.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

### Running Tests
*   **Phase 1 Tests**:
    ```bash
    python tests/test_phase1.py
    ```
    Validates core components (StateVector, BayesianBrain, LayerEngine).
*   **Full System Tests**:
    ```bash
    python tests/test_full_system.py
    ```
    Validates full system integration.

### Building Executable
*   **Build Script**:
    ```bash
    python scripts/build_executable.py
    ```
    The output will be in `dist/Bayesian_AI_Engine/`.

## 3. Training Workflow

To train the Bayesian probability model using real Databento data:

### Data Setup
1.  **Create Directory**:
    ```bash
    mkdir -p data/raw
    ```
2.  **Copy Files**:
    Place your `.dbn.zst` files in `data/raw/`.
3.  **Setup & Verify**:
    ```bash
    python scripts/setup_test_data.py
    ```

### Pipeline Execution
1.  **Verify Data Loading & Velocity**:
    ```bash
    python -m pytest tests/test_real_data_velocity.py -v
    python -m pytest tests/test_databento_loading.py -v
    ```
2.  **Run Training Loop**:
    ```bash
    # Start small (e.g., 10 iterations)
    python training/orchestrator.py \
      --data-dir data/raw \
      --iterations 10 \
      --output models/
    ```
3.  **Inspect Results**:
    ```bash
    python scripts/inspect_results.py models/probability_table.pkl
    ```

## 4. CI/CD & Automation

### Unified Pipeline
A unified CI/CD workflow is defined in `.github/workflows/unified_test_pipeline.yml`. It runs on every push and pull request to the `main` or `master` branches.
*   **Steps**: Installs dependencies, runs integrity/math checks, executes tests (Phase 1, Full System, CUDA Audit), generates status report, and builds the executable.

### Status Reporting
An automated status report is included in the unified workflow.
*   **Script**: `scripts/generate_status_report.py`
*   **Output**: `CURRENT_STATUS.md`
*   **Purpose**: Provides a living snapshot of project health, code stats, and validation checks.

## 5. Debugging & Logging

When troubleshooting issues, please include the following generated files if available.

### Log Files
*   **`CUDA_Debug.log`**: Captures root-level CUDA initialization and verification events.
*   **`notebooks/CUDA_Debug.log`**: Captures GPU kernel execution details and pattern detection logs from notebook runs.
*   **`debug_outputs/*.log`**: Specific test execution logs (e.g., `tests/test_phase0.py` generates detailed logs here).
*   **`CURRENT_STATUS.md`**: The latest system health report.

### Git Policy
**NOTE:** The files listed above are auto-generated but are **whitelisted** in `.gitignore`. **Do** commit these files unless explicitly instructed not to, as they provide critical debugging history.

## 6. Technical Context

*   **Dependencies**: `numpy` must be kept `<2.0` for compatibility with `numba`. `numba` and `llvmlite` are required.
*   **Hardware**: The system supports CUDA acceleration via `numba.cuda` and `cupy`. The CI environment does not have GPUs, so the code automatically falls back to CPU mode.
*   **Configuration**: The `config/` directory contains system settings and is bundled with the executable.
*   **Data**: `probability_table.pkl` is required for the engine to function and will be generated if missing.
