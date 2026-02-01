# Changelog - Bayesian-AI

## [V2.0.0] - Unified System Realignment

### Refactoring & Architecture
- **Global Rename**: Rebranded all modules from legacy "Sniper" nomenclature to "Bayesian-AI".
- **Functional Naming**: Standardized file naming convention (e.g., `velocity_gate.py`, `bayesian_brain.py`).
- **Pipeline Definition**: Formalized **LOAD -> TRANSFORM -> ANALYZE -> VISUALIZE** workflow.

### Logic Shifts
- **Bayesian Probability Engine**:
    - Replaced heuristic-based entry logic with `BayesianBrain`.
    - Implemented Hash-Map based `StateVector` lookup for O(1) decision speed.
    - Added "Win Rate + Confidence" dual-threshold trigger (80% / 30%).
- **9-Layer Temporal Hierarchy**:
    - Defined strict separation between Static (L1-L4) and Fluid (L5-L9) contexts.
    - Offloaded L7, L8, and L9 computation to dedicated CUDA-ready modules.
- **Velocity Gate (L9)**:
    - Implemented `detect_cascade` algorithm for sub-second volatility detection (10+ points in <0.5s).

### Dependencies & Environment
- **NumPy**: Pinned to `<2.0` (specifically `1.26.4`) for Numba compatibility.
- **Numba**: Integrated for JIT compilation of critical paths (`velocity_gate`, `pattern_detector`).
- **Databento**: Added native support for `.dbn` file ingestion via `DatabentoLoader`.

### Removed
- Removed legacy "Sniper" headers and docstrings.
- Deprecated CPU-only cascade detection in favor of Numba-accelerated implementation.
