# Bayesian-AI

> **Algorithmic Trading System with 9-Layer Temporal Hierarchy & Bayesian Inference**

[![Status](https://img.shields.io/badge/Status-Active-success.svg)](docs/TECHNICAL_MANUAL.md#part-4-current-project-status)

Bayesian-AI is a high-frequency trading system that utilizes a Fractal Three-Body Quantum model to capture market conditions across multiple timeframes (from 90 days down to 1 second) and applies Bayesian probability to estimate win rates.

**Active Engine:** Fractal Three-Body Quantum (PyTorch/CUDA)
**Legacy Engine:** 9-Layer Hierarchy (Deprecated/Archived)

## 📖 Documentation

*   **[Technical Manual](docs/TECHNICAL_MANUAL.md)**: The single source of truth. Contains:
    *   **System Logic & Parameters**: Physics Engine, Math ("Nightmare Protocol"), Learning Cycle.
    *   **Dashboard Guide**: Interactive validation and debugging.
    *   **Legacy Reference**: Archived architecture for historical context.
    *   **Current Status**: Live project health, code statistics, and validation metrics.
*   **[Agent Instructions](AGENTS.md)**: Guidelines for AI agents working on this codebase.

## 🚀 Quick Start (Local Development)

To run the workflow locally, ensure you have Python 3.10+ installed.

1.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

2.  **Run Tests**:
    ```bash
    python scripts/run_tests.py
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

### System Health & CUDA Check
Run the unified health check script to verify Python environment, dependencies, and CUDA availability:

```bash
python scripts/system_health.py
```

### CUDA Not Detected / Running on CPU
If you have an NVIDIA GPU but the system says "CUDA: NOT AVAILABLE" or runs slowly on CPU:

1.  **Run Health Check**:
    ```bash
    python scripts/system_health.py
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
