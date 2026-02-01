# Bayesian-AI - System Logic

## Pipeline Overview

The system follows a strict **LOAD -> TRANSFORM -> ANALYZE -> VISUALIZE** pipeline.

### 1. LOAD
**Ingestion & Normalization**
- **Source**: Databento (`.dbn`) or Parquet (`.parquet`) files.
- **Component**: `training/databento_loader.py` / `training/orchestrator.py`
- **Output**: Standardized DataFrame with `timestamp`, `price`, `volume`, `type`.

### 2. TRANSFORM
**State Generation (The 9-Layer Hierarchy)**
- **Component**: `core/layer_engine.py` (Coordinator)
- **Aggregation**: `core/data_aggregator.py` maintains real-time buffers.
- **Computation**:
    - **Static Context (L1-L4)**: Computed once per session (CPU).
    - **Fluid Context (L5-L9)**: Computed in real-time.
        - L5 (4hr) & L6 (1hr): CPU trend/structure checks.
        - L7 (15m): `cuda/pattern_detector.py` (Pattern Recognition).
        - L8 (5m): `cuda/confirmation.py` (Volume/Structure Confirmation).
        - L9 (1s): `cuda/velocity_gate.py` (Velocity Cascade Detection).

### 3. ANALYZE
**Decision Engine**
- **State Representation**: `core/state_vector.py` creates an immutable hash of the 9-layer state.
- **Probability Lookup**: `core/bayesian_brain.py` maps `StateVector` -> Win Probability & Confidence.
- **Execution**: `execution/wave_rider.py` manages positions, trailing stops, and structure-break exits.

### 4. VISUALIZE
**Reporting & Execution**
- **Component**: `engine_core.py` (Main Loop)
- **Output**: Real-time logs, trade executions, PnL updates.

---

## 9-Layer Hierarchy

| Layer | Timeframe | Type | Compute | Component |
|-------|-----------|------|---------|-----------|
| **L1** | 90-Day | Bias | CPU (Static) | `layer_engine.py` |
| **L2** | 30-Day | Regime | CPU (Static) | `layer_engine.py` |
| **L3** | 1-Week | Swing | CPU (Static) | `layer_engine.py` |
| **L4** | Daily | Zone | CPU (Static) | `layer_engine.py` |
| **L5** | 4-Hour | Trend | CPU (Fluid) | `layer_engine.py` |
| **L6** | 1-Hour | Structure | CPU (Fluid) | `layer_engine.py` |
| **L7** | 15-Min | Pattern | **CUDA** | `pattern_detector.py` |
| **L8** | 5-Min | Confirm | **CUDA** | `confirmation.py` |
| **L9** | 1-Sec | Velocity | **CUDA** | `velocity_gate.py` |

## Data Pathing

- **Training**: `TrainingOrchestrator` loads historical data (Parquet/DBN) -> Feeds `BayesianEngine` -> Updates `probability_table.pkl`.
- **Live**: `BayesianEngine` loads `probability_table.pkl` -> Processes live ticks -> Queries Bayesian Brain -> Executes via `WaveRider`.
