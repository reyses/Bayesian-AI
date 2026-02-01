# Agent Instructions

This repository contains "ProjectX v2.0", an algorithmic trading system.

## Workflow

A CI/CD workflow is defined in `.github/workflows/ci.yml`. It runs on every push and pull request to the `main` or `master` branches.

The workflow performs the following steps:
1.  **Installs Dependencies**: From `requirements.txt`.
2.  **Runs Tests**:
    *   `tests/test_phase1.py`: Validates core components (StateVector, BayesianBrain, LayerEngine).
    *   `tests/test_full_system.py`: Validates full system integration.
3.  **Builds Executable**: Runs `build_executable.py` to package the application using PyInstaller.

## Local Development

To run the workflow locally, ensure you have Python 3.10 installed.

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
    python build_executable.py
    ```
    The output will be in `dist/Bayesian_AI_Engine/`.

## Important Notes

*   **Dependencies**: `numpy` must be kept `<2.0` for compatibility with `numba`. `numba` and `llvmlite` are required.
*   **CUDA**: The system supports CUDA acceleration via `numba.cuda` and `cupy`. The CI environment does not have GPUs, so the code falls back to CPU mode.
*   **Configuration**: `config/` directory is bundled with the executable.
*   **Data**: `probability_table.pkl` is required and will be generated if missing.
