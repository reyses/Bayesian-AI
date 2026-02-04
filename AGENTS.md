# Agent Instructions

This repository contains "Bayesian-AI", an algorithmic trading system.

## Workflow

A unified CI/CD workflow is defined in `.github/workflows/unified_test_pipeline.yml`. It runs on every push and pull request to the `main` or `master` branches.

The workflow performs the following steps:
1.  **Installs Dependencies**: From `requirements.txt`.
2.  **Runs Tests**:
    *   Integrity and Math verification.
    *   `tests/test_phase1.py`: Validates core components (StateVector, BayesianBrain, LayerEngine).
    *   `tests/test_full_system.py`: Validates full system integration.
    *   CUDA Audit & Sentinel Bridge (Feedback Loop).
    *   Training Validation.
3.  **Generates Status Report**: Runs `scripts/generate_status_report.py` and updates `CURRENT_STATUS.md`.
4.  **Builds Executable**: Runs `scripts/build_executable.py` to package the application using PyInstaller.

## Status Reporting

An automated status report is included in the unified workflow.
- **Script**: `scripts/generate_status_report.py`
- **Output**: `CURRENT_STATUS.md`
- **Purpose**: Provides a living snapshot of project health, code stats, and validation checks.

## Training Workflow

To train the Bayesian probability model using real Databento data:

1.  **Create Data Directory & Copy Files**:
    ```bash
    mkdir -p data/raw
    # Copy your .dbn.zst files here
    cp /path/to/databento/downloads/*.dbn.zst data/raw/
    ```

2.  **Run Pipeline**:

    *   **Setup & Verify Data**:
        ```bash
        python scripts/setup_test_data.py
        ```

    *   **Run Training**:
        ```bash
        python -m pytest tests/test_real_data_velocity.py -v
        python -m pytest tests/test_databento_loading.py -v
        ```

    *   **Run Training**:
        ```bash
        # Start small (10 iterations)
        python training/orchestrator.py \
          --data-dir data/raw \
          --iterations 10 \
          --output models/
        ```

    *   **Inspect Results**:
        ```bash
        python scripts/inspect_results.py models/probability_table.pkl
        ```

## Local Development

To run the workflow locally, ensure you have Python 3.10+ installed.

1.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

2.  **Run Tests**:
    ```bash
    python tests/test_phase1.py
    python tests/test_full_system.py
    ```

3.  **Build Executable**:
    ```bash
    python scripts/build_executable.py
    ```
    The output will be in `dist/Bayesian_AI_Engine/`.

4.  **Generate Status Report**:
    ```bash
    python scripts/generate_status_report.py
    ```

## Important Notes

*   **Dependencies**: `numpy` must be kept `<2.0` for compatibility with `numba`. `numba` and `llvmlite` are required.
*   **CUDA**: The system supports CUDA acceleration via `numba.cuda` and `cupy`. The CI environment does not have GPUs, so the code falls back to CPU mode.
*   **Configuration**: `config/` directory is bundled with the executable.
*   **Data**: `probability_table.pkl` is required and will be generated if missing.
