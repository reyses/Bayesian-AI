# Repository Audit Report: Architecture Inconsistency

**Date:** 2026-02-06
**Auditor:** Jules (AI Software Engineer)

## 1. Executive Summary

The repository currently houses two distinct and conflicting trading architectures:
1.  **Legacy System (Active):** A "9-Layer Hierarchy" system based on `StateVector`, `LayerEngine`, and `BayesianBrain`. This system is fully documented, tested, and tied to the main execution entry point (`engine_core.py`).
2.  **Next-Gen System (Inactive/Partial):** A "Fractal Three-Body Quantum" system based on `FractalMarketState`, `QuantumFieldEngine`, and `ThreeBodyQuantumState`. This system appears to be a work-in-progress refactor that has been committed but not fully integrated or documented.

**Conclusion:** The codebase is in a transitional state. The documentation (`README.md`, `TECHNICAL_MANUAL.md`) reflects the Legacy system, while the file structure contains a mix of both. This creates confusion regarding the intended behavior and the "source of truth" for the system's logic.

## 2. Architecture Mismatch

| Feature | Legacy System (Active) | Next-Gen System (Inactive) |
| :--- | :--- | :--- |
| **State Model** | `StateVector` (9-Layer HashMap) | `ThreeBodyQuantumState` (Fractal/Quantum) |
| **Core Engine** | `core/layer_engine.py` | `core/fractal_three_body.py` |
| **Physics** | Simple Velocity/Pattern checks | `QuantumFieldEngine` (Roche Limits, Tunnels) |
| **Learning** | `BayesianBrain` (Simple Laplace) | `QuantumBayesianBrain` (Adaptive Phases) |
| **Entry Point** | `engine_core.py` -> `LayerEngine` | *None (No integrated entry point)* |
| **Status** | **PRODUCTION** | **EXPERIMENTAL** |

## 3. File Categorization

### A. Legacy System (Active)
These files constitute the currently running system:
*   `engine_core.py` (Main Entry Point)
*   `core/layer_engine.py` (State Computation)
*   `core/state_vector.py` (Data Structure)
*   `training/orchestrator.py` (Training Loop)
*   `execution/wave_rider.py` (Execution Logic)
*   `cuda_modules/*` (Pattern/Velocity acceleration for Legacy L7/L9)

### B. Next-Gen System (Inactive)
These files are present but not used by the main engine:
*   `core/fractal_three_body.py`
*   `core/three_body_state.py`
*   `core/quantum_field_engine.py`
*   `core/resonance_cascade.py`
*   `core/adaptive_confidence.py`

### C. Shared / Bridge
*   `core/bayesian_brain.py`: Updated to include `QuantumBayesianBrain` but `engine_core.py` still instantiates the base class.

## 4. Inconsistencies Identified

### Documentation
*   **Critical:** `docs/TECHNICAL_MANUAL.md` describes the 9-Layer Hierarchy in detail but makes zero mention of the Fractal/Quantum architecture.
*   **Critical:** `README.md` points to the 9-Layer Hierarchy.

### Execution
*   `engine_core.py` initializes `LayerEngine` (Legacy). It ignores `FractalMarketState`.
*   `training/orchestrator.py` trains the Legacy model (`probability_table.pkl` based on `StateVector`).

### Testing
*   `tests/test_full_system.py` validates the Legacy system.
*   `tests/test_quantum_system.py` validates the Next-Gen components in isolation.

## 5. Recommendations

1.  **Acknowledge Transition:** Update `README.md` and `CURRENT_STATUS.md` to explicitly label the Quantum system as "Experimental" or "Next-Gen" to avoid confusion.
2.  **Decide Path Forward:**
    *   *Option A (Migration):* Refactor `engine_core.py` to use `FractalMarketState` and `QuantumBayesianBrain`. Deprecate `LayerEngine`.
    *   *Option B (Cleanup):* If the Quantum experiment is abandoned, remove the files to clean up the repo.
    *   *Option C (Parallel):* Create a new entry point (e.g., `engine_quantum.py`) to allow running the new system alongside the old one for A/B testing.
3.  **Audit Logs:** Ensure logging (`CUDA_Debug.log`) captures which engine is running.

## 6. Action Taken
*   Created this Audit Report.
*   Updated `README.md` to mention the Next-Gen architecture.
*   Updated `scripts/generate_status_report.py` to track architecture status programmatically.
