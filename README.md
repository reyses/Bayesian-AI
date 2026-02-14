# Bayesian-AI

> **Algorithmic Trading System with 9-Layer Temporal Hierarchy & Bayesian Inference**

[![Status](https://img.shields.io/badge/Status-Active-success.svg)](CURRENT_STATUS.md)

Bayesian-AI is a high-frequency trading system that utilizes a Fractal Three-Body Quantum model to capture market conditions across multiple timeframes (from 90 days down to 1 second) and applies Bayesian probability to estimate win rates.

**Active Engine:** Fractal Three-Body Quantum (PyTorch/CUDA)
**Legacy Engine:** 9-Layer Hierarchy (Deprecated/Archived)

## 📖 Documentation

*   **[Technical Manual](docs/TECHNICAL_MANUAL.md)**: Comprehensive system logic, architecture, and module reference.
*   **[Dashboard Guide](docs/DASHBOARD_GUIDE.md)**: Instructions for using the interactive validation, debugging, and learning dashboard.
*   **[Current Status](CURRENT_STATUS.md)**: Live project health, code statistics, and validation metrics.
*   **[Agent Instructions](AGENTS.md)**: Guidelines for AI agents working on this codebase.

## 🚀 Quick Start (Local Development)

To run the workflow locally, ensure you have Python 3.10+ installed.

1.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

2.  **Run Tests**:
    ```bash
    python tests/test_phase1.py
    python tests/test_integration_quantum.py
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

## 🔧 Troubleshooting

### CUDA Not Detected / Running on CPU
If you have an NVIDIA GPU but the system says "CUDA: NOT AVAILABLE" or runs slowly on CPU:

1.  **Check GPU Health**:
    ```bash
    python scripts/gpu_health_check.py
    ```
2.  **Fix PyTorch Installation**:
    If the above check fails or reports CPU-only torch, run the fix script to force a CUDA-enabled reinstallation:
    ```bash
    python scripts/fix_cuda.py
    ```
    This will uninstall existing PyTorch packages and reinstall the correct CUDA 12.1 version.

## 🏗 System Architecture

The system operates on a **LOAD -> TRANSFORM -> ANALYZE -> VISUALIZE** pipeline.

*   **Fractal Three-Body Quantum**: Uses Roche limits, wave functions, and tunnel probabilities to model market state.
*   **Bayesian Brain**: Learns probability distributions of "WIN" outcomes for unique quantum states.
*   **CUDA Acceleration**: Offloads quantum wave function calculations and simulations to the GPU via PyTorch.
*   **Design of Experiments (DOE)**: Advanced parameter optimization using Latin Hypercube Sampling (LHS), Response Surface Optimization, and PnL-prioritized Regret Analysis.

For detailed architecture, see the [Technical Manual](docs/TECHNICAL_MANUAL.md).

## 🏛 Legacy Architecture (Archived)

The legacy **9-Layer Hierarchy** trading engine has been moved to `archive/` (`archive/layer_engine.py`, `archive/cuda_modules/`). This system is **deprecated** and no longer active in the main execution loop.

*   **9-Layer Hierarchy**: Decomposes market state into Static (L1-L4) and Fluid (L5-L9) layers.
*   **Status**: Archived / Deprecated.
