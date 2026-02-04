# Bayesian-AI

> **Algorithmic Trading System with 9-Layer Temporal Hierarchy & Bayesian Inference**

[![Status](https://img.shields.io/badge/Status-Active-success.svg)](CURRENT_STATUS.md)

Bayesian-AI is a high-frequency trading system that utilizes a 9-layer hierarchical state model to capture market conditions across multiple timeframes (from 90 days down to 1 second) and applies Bayesian probability to estimate win rates.

## 📖 Documentation

*   **[Technical Manual](docs/TECHNICAL_MANUAL.md)**: Comprehensive system logic, architecture, and module reference.
*   **[Debug Dashboard Guide](docs/DEBUG_DASHBOARD_GUIDE.md)**: Instructions for using the interactive validation notebook.
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

*   **9-Layer Hierarchy**: Decomposes market state into Static (L1-L4) and Fluid (L5-L9) layers.
*   **Bayesian Brain**: Learns probability distributions of "WIN" outcomes for unique market states.
*   **CUDA Acceleration**: Offloads high-frequency pattern detection (L7), confirmation (L8), and velocity checks (L9) to the GPU via Numba.

For detailed architecture, see the [Technical Manual](docs/TECHNICAL_MANUAL.md).
