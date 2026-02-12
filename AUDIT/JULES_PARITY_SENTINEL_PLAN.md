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
- **Extraction:** Move pure mathematical methods into `core/kernels/quantum_math.py` to ensure complete separation of logic. This includes:
    - `_calculate_wave_function`
    - `_calculate_tunneling`
    - `_calculate_force_fields`
    - `_detect_geometric_patterns`
    - `batch_compute_states` (logic portion)
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
1.  **Initialize Deterministic Tensors:** Use fixed seeds (`torch.manual_seed`, `np.random.seed`) to generate random input tensors/arrays. This prevents flaky tests and ensures reproducibility.
2.  **Run Production Kernel:** Execute `core.kernels.quantum_math` on the available device (CPU in CI, CUDA if available).
3.  **Run Mirror Logic:** Execute `tests.cpu_validation.mirror_logic`.
4.  **Compare Outputs:** Validate parity using robust floating-point comparison:
    - `numpy.allclose(kernel_out, mirror_out, atol=1e-7, rtol=1e-5)`
    - This accounts for minor precision differences between PyTorch (float32) and NumPy (float64 default) or device-specific variations.

#### 3.5. Automated Parity Check (CI Hook)
To prevent drift between production kernels and the validation mirror, a CI step or pre-commit hook should be implemented. This hook will:
- Detect if any file in `core/kernels/` has been modified in a PR.
- If modified, verify that `tests/cpu_validation/mirror_logic.py` has also been modified in the same commit/PR.
- Fail the build if the mirror is not updated, enforcing the "Responsibility" AOP.

**Note on CI Limitations:**
Since GitHub Actions runners are CPU-only, the "Production Kernel" (PyTorch) will execute on CPU during CI tests. This is acceptable as PyTorch ensures mathematical consistency across devices. The "Parity Sentinel" value ensures that we have a *logic* verification independent of the PyTorch framework (via the NumPy mirror), protecting against framework-specific bugs or unintended logic changes in the kernel.

## 4. Execution Plan

1.  **Setup:** Create `tests/cpu_validation/`.
2.  **Extraction:** Refactor `core/quantum_field_engine.py` -> `core/kernels/quantum_math.py`.
3.  **Mirroring:** Create `tests/cpu_validation/mirror_logic.py`.
4.  **Testing:** Implement `tests/cpu_validation/test_parity.py` with deterministic seeds and `allclose` validation.
5.  **Automation:** Configure the CI hook for parity checks.
6.  **Documentation:** Update `AGENTS.md`.

## 5. Conclusion

This plan establishes the "Parity Sentinel" architecture, enforcing a rigorous separation of concerns and ensuring that the high-performance production code remains clean and focused, while safety and correctness are guaranteed by an independent CPU-based validation suite.
