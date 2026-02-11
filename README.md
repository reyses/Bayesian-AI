# Bayesian-AI

> **Algorithmic Trading System with 9-Layer Temporal Hierarchy & Bayesian Inference**

[![Status](https://img.shields.io/badge/Status-Active-success.svg)](CURRENT_STATUS.md)

Bayesian-AI is a high-frequency trading system that utilizes a Fractal Three-Body Quantum model to capture market conditions across multiple timeframes (from 90 days down to 1 second) and applies Bayesian probability to estimate win rates.

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

## 🏗 System Architecture

The system operates on a **LOAD -> TRANSFORM -> ANALYZE -> VISUALIZE** pipeline.

*   **Fractal Three-Body Quantum**: Uses Roche limits, wave functions, and tunnel probabilities to model market state.
*   **Bayesian Brain**: Learns probability distributions of "WIN" outcomes for unique quantum states.
*   **CUDA Acceleration**: Offloads quantum wave function calculations and simulations to the GPU via PyTorch.

For detailed architecture, see the [Technical Manual](docs/TECHNICAL_MANUAL.md).

## 🏛 Legacy Architecture (Deprecated)

The repository also includes the legacy **9-Layer Hierarchy** trading engine (`core/layer_engine.py`, `cuda_modules/`). This system is **deprecated** and no longer active in the main execution loop.

*   **9-Layer Hierarchy**: Decomposes market state into Static (L1-L4) and Fluid (L5-L9) layers.
*   **Status**: Deprecated / Maintenance Mode.
