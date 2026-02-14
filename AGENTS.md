# Agent Instructions

This repository contains "Bayesian-AI", a sophisticated algorithmic trading system leveraging quantum probability fields and Bayesian inference.

## 1. Core Directives

*   **Single Source of Truth**: `requirements.txt` is the ONLY allowed file for defining project dependencies. Do not create separate requirements files.
*   **Dependency Pinning**: Critical dependencies must be pinned in `requirements.txt`.
*   **Notebook Limit**: Only one notebook is permitted unless explicitly authorized.
*   **Artifacts**: Do not modify files in `dist/` or `build/` directly. Edit the source code and rebuild.
*   **Documentation**: The primary documentation source is `docs/TECHNICAL_MANUAL.md`.

## 2. Development Workflow

### Local Setup
1.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

### Running Tests
Use `pytest` to run the test suite.
*   **Run All Tests**:
    ```bash
    python -m pytest
    ```
*   **Key Test Modules**:
    -   `tests/test_bayesian_brain.py`: Validates core decision logic.
    -   `tests/test_quantum_field_engine.py`: Tests the vectorized math engine.
    -   `tests/test_integration_quantum.py`: Verifies system integration.

### Building Executable
*   **Build Script**:
    ```bash
    python scripts/build_executable.py
    ```
    The output will be in `dist/Bayesian_AI_Engine/`.

## 3. Training Workflow

The system learns by optimizing parameters over historical data using a Bayesian approach.

### Data Setup
1.  **Directory Structure**:
    Ensure `DATA/RAW` exists. If missing, create it:
    ```bash
    mkdir -p DATA/RAW
    ```
2.  **Data Files**:
    Place `.dbn.zst` (Databento compressed) files in `DATA/RAW`.

### Pipeline Execution
Run the `BayesianTrainingOrchestrator`:
```bash
python training/orchestrator.py --data-dir DATA/RAW --iterations 50 --output checkpoints/
```
*   **--data-dir**: Path to raw data files.
*   **--iterations**: Number of optimization loops per day.
*   **--output**: Directory for saving models and logs.

## 4. Project Structure

*   **`core/`**: The heart of the system. Contains `QuantumFieldEngine` (vectorized math), `BayesianBrain`, and `ThreeBodyQuantumState`.
*   **`training/`**: Orchestration logic (`BayesianTrainingOrchestrator`), DOE (Design of Experiments), and parameter generation.
*   **`tests/`**: Comprehensive test suite using `pytest`.
*   **`scripts/`**: Utility scripts for building, CUDA health checks (`gpu_health_check.py`), and status reporting.
*   **`docs/`**: Documentation, primarily `TECHNICAL_MANUAL.md`.

## 5. CI/CD & Automation

### Unified Pipeline
The workflow is defined in `.github/workflows/unified_test_pipeline.yml`. It runs automatically on push/PR to `main`.

### Status Reporting
*   **Script**: `scripts/generate_status_report.py`
*   **Output**: Console output and potentially updated docs.
*   **Purpose**: Snapshots project health and validation status.

## 6. Technical Context

*   **Dependencies**:
    -   `numpy < 2`: Required for compatibility.
    -   `torch`: Used for tensor operations, configured for CUDA (`cu121`) where available.
    -   `numba`: Used for JIT compilation acceleration.
    -   `pandas` & `pandas_ta`: Data manipulation and technical analysis.
*   **Hardware**: The system is optimized for NVIDIA GPUs (CUDA) but will fallback to CPU if necessary. Use `scripts/gpu_health_check.py` to verify GPU status.
