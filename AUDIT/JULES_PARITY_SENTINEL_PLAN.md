# Jules Parity Sentinel: Implementation Plan

**Date:** 2026-02-18
**Author:** Jules (The Parity Sentinel)
**Status:** DRAFT

## 1. Executive Summary

This report outlines the strategy to enforce strict "Air-Gap" separation between CUDA production code and CPU validation logic within the `Bayesian-AI` repository. The objective is to eliminate "fallback pollution" in production kernels and ensure 1:1 mathematical parity via rigorous, independent CPU verification suites.

## 2. Current Architecture Assessment

The active engine, `core/quantum_field_engine.py`, currently implements a hybrid architecture using PyTorch:
- **Mixed Logic:** The class `QuantumFieldEngine` contains both high-level orchestration and low-level mathematical operations.
- **Internal Fallback:** Methods like `_calculate_wave_function` use `if self.use_gpu:` blocks to switch between PyTorch (CUDA) and NumPy (CPU) paths internally.
- **Violation:** This violates the "Parity Sentinel" directive which prohibits fallback logic within production source files.

The legacy `cuda_modules/` directory uses Numba but is deprecated.

## 3. Implementation Strategy

To comply with the "Jules" directives, we must refactor the codebase to separate the "Kernel" logic from the "Engine" orchestration.

### Phase 1: Test Suite Architecture & Refactoring

We will restructure the core engine to isolate mathematical kernels.

#### 3.1. Directory Structure Changes
The following structure is proposed:

```
core/
├── quantum_field_engine.py       # Orchestrator (loads kernels)
└── kernels/                      # NEW: Pure Mathematical Kernels
    ├── __init__.py
    └── quantum_math.py           # PyTorch/CUDA Implementation (Production)

tests/
└── cpu_validation/               # NEW: Independent Verification Suite
    ├── __init__.py
    ├── mirror_logic.py           # NumPy Implementation (Reference)
    └── test_parity.py            # Validation Gate (Epsilon Check)
```

#### 3.2. Refactoring `quantum_field_engine.py`
- **Extraction:** Move mathematical methods (`_calculate_wave_function`, `_calculate_tunneling`, `batch_compute_states` logic) into `core/kernels/quantum_math.py`.
- **Constraint:** `core/kernels/quantum_math.py` must use **PyTorch** exclusively. It shall not import `numpy` or check for CUDA availability. It assumes it is running on the compute device provided (Device Agnostic or CUDA-specific).
- **Orchestration:** `QuantumFieldEngine` will handle data loading and device management, passing tensors to the kernels.

#### 3.3. Creating the CPU Mirror
- **Mirror Logic:** implementation of `tests/cpu_validation/mirror_logic.py` using **NumPy** exclusively.
- **Strict Parity:** This file must replicate the math of `core/kernels/quantum_math.py` exactly, line-by-line.

### Phase 2: Agent Operating Procedures (AOPs)

We will append the following directives to `AGENTS.md` (or a new `CONTRIBUTING.md`):

```markdown
## AI AGENT OPERATING PROCEDURES (AOPS) - JULES PROTOCOL

1.  **SCOPE**: Modifications to `core/kernels/` are restricted to high-performance (CUDA/PyTorch) optimizations.
2.  **PROHIBITION**: Do not inject `if torch.cuda.is_available():` or `try: import cupy` blocks into `core/kernels/`. Fallback logic is strictly prohibited in production kernels.
3.  **RESPONSIBILITY**: If you modify a function in `core/kernels/`, you MUST signal Jules to update the corresponding CPU mirror in `tests/cpu_validation/`.
4.  **NULL STATE**: Any PR introducing CPU-compatibility code (fallbacks) into the production kernel module will be rejected.
```

### Phase 3: Jules PR Review Logic (Verification)

We will implement a verification workflow (GitHub Action or Script) to enforce parity.

#### 3.4. Validation Gate: `tests/cpu_validation/test_parity.py`
This script will:
1.  Initialize random tensors/arrays.
2.  Run the **Production Kernel** (`core.kernels.quantum_math`) on the available device (CPU in CI, CUDA if available).
3.  Run the **Mirror Logic** (`tests.cpu_validation.mirror_logic`).
4.  Compare outputs: $|Kernel_{output} - Mirror_{output}| < \epsilon$ (where $\epsilon \approx 1e^{-7}$ for float32).

**Note on CI Limitations:**
Since GitHub Actions runners are CPU-only, the "Production Kernel" (PyTorch) will execute on CPU during CI tests. This is acceptable as PyTorch ensures mathematical consistency across devices. The "Parity Sentinel" value ensures that we have a *logic* verification independent of the PyTorch framework (via the NumPy mirror), protecting against framework-specific bugs or unintended logic changes in the kernel.

## 4. Execution Plan

1.  **Setup:** Create `tests/cpu_validation/`.
2.  **Extraction:** Refactor `core/quantum_field_engine.py` -> `core/kernels/quantum_math.py`.
3.  **Mirroring:** Create `tests/cpu_validation/mirror_logic.py`.
4.  **Testing:** Implement `tests/cpu_validation/test_parity.py`.
5.  **Documentation:** Update `AGENTS.md`.

## 5. Conclusion

This plan establishes the "Parity Sentinel" architecture, enforcing a rigorous separation of concerns and ensuring that the high-performance production code remains clean and focused, while safety and correctness are guaranteed by an independent CPU-based validation suite.
