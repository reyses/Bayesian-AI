# Agent Instructions

This repository contains "Bayesian-AI", a sophisticated algorithmic trading system leveraging quantum probability fields and Bayesian inference.

## 1. Core Directives

*   **Single Source of Truth**: `requirements.txt` is the ONLY allowed file for defining project dependencies. Do not create separate requirements files.
*   **Dependency Pinning**: Critical dependencies must be pinned in `requirements.txt`.
*   **Notebook Limit**: Only one notebook is permitted unless explicitly authorized.
*   **Artifacts**: Do not modify files in `dist/` or `build/` directly. Edit the source code and rebuild.
*   **Documentation**: The primary documentation source is `docs/TECHNICAL_MANUAL.md`.

## 2. Development Workflow

### Local Setup
1.  **Python Version**: Ensure you have Python 3.10+ installed.
2.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

### Running Tests
Use `pytest` to run the test suite.
*   **Run All Tests**:
    ```bash
    python -m pytest
    ```
*   **Key Test Modules**:
    -   `tests/test_bayesian_brain.py`: Validates core decision logic.
    -   `tests/test_quantum_field_engine.py`: Tests the vectorized math engine.
    -   `tests/test_integration_quantum.py`: Verifies system integration.

### Building Executable
*   **Build Script**:
    ```bash
    python scripts/build_executable.py
    ```
    The output will be in `dist/Bayesian_AI_Engine/`.

## 3. Training Workflow

The system learns by optimizing parameters over historical data using a Bayesian approach.

### Data Setup
1.  **Directory Structure**:
    Ensure `DATA/RAW` exists. If missing, create it:
    ```bash
    mkdir -p DATA/RAW
    ```
    *Note: The system strictly uses `DATA/RAW` (uppercase) as the default data path in configuration.*

2.  **Data Files**:
    Place `.dbn.zst` (Databento compressed) files in `DATA/RAW`.

### Pipeline Execution
Run the `BayesianTrainingOrchestrator`:
```bash
python training/orchestrator.py --data-dir DATA/RAW --iterations 50 --output checkpoints/
```
*   **--data-dir**: Path to raw data files.
*   **--iterations**: Number of optimization loops per day.
*   **--output**: Directory for saving models and logs.

## 4. Project Structure

*   **`core/`**: The heart of the system. Contains `QuantumFieldEngine` (vectorized math), `BayesianBrain`, and `ThreeBodyQuantumState`.
*   **`training/`**: Orchestration logic (`BayesianTrainingOrchestrator`), DOE (Design of Experiments), and parameter generation.
*   **`tests/`**: Comprehensive test suite using `pytest`.
*   **`scripts/`**: Utility scripts for building, CUDA health checks (`gpu_health_check.py`), and status reporting.
*   **`docs/`**: Documentation, primarily `TECHNICAL_MANUAL.md`.

## 5. Debugging & Logging

Use these resources to troubleshoot issues.

*   **Log Files**:
    *   `CUDA_Debug.log`: Logs related to GPU initialization and kernel execution.
    *   `training.log` / `orchestrator.log`: Runtime logs from the training process (if configured).
*   **Git Policy**: Do not commit generated log files unless explicitly requested for debugging history. They should be ignored by `.gitignore`.

## 6. CI/CD & Automation

### Unified Pipeline
The workflow is defined in `.github/workflows/unified_test_pipeline.yml`. It runs automatically on push/PR to `main`.

### Status Reporting
*   **Script**: `scripts/generate_status_report.py`
*   **Output**: Console output and potentially updated docs (`CURRENT_STATUS.md`).
*   **Purpose**: Snapshots project health and validation status.

## 8. Coding Standards (mandatory — apply to every file you touch)

### No Magic Numbers
Every numeric literal that is a threshold, limit, ratio, count, or tunable parameter
MUST be assigned to a named constant. Never write a raw number inline.

- Module-level constants use `ALL_CAPS` and live at the top of the file.
- Function/method-local constants use `_UPPER_CASE` and are defined immediately
  before the block that uses them.

```python
# WRONG
if dist < 4.5:
if n_members >= 10 and win_rate >= 0.55:
term_pid = 0.5 * z + 0.1 * cumsum + 0.2 * diff

# RIGHT — module-level
CLUSTER_MAX_DIST       = 4.5    # max Euclidean dist (scaled feature space) to accept cluster match
EXCEPTION_MIN_MEMBERS  = 10     # minimum pattern count before a template earns Gate 0 exception
EXCEPTION_MIN_WIN_RATE = 0.55   # win rate floor for data-quality override
DEFAULT_PID_KP         = 0.5    # proportional gain (dimensionless)
DEFAULT_PID_KI         = 0.1    # integral gain (dimensionless)
DEFAULT_PID_KD         = 0.2    # derivative gain (dimensionless)

# RIGHT — local block
_MIN_REGIME_BARS = 3    # consecutive PID-regime bars required before firing a signal (count)
_MAX_Z_ENTER     = 2.0  # z-score ceiling; above = nightmare field, do not enter (sigma units)
if self._regime_n >= _MIN_REGIME_BARS and abs(z) < _MAX_Z_ENTER:
```

This applies to:
- Physics thresholds (sigma multiples, z-score cutoffs, probability floors)
- PID constants (KP, KI, KD, tension thresholds)
- Gate thresholds (dist, ADX, Hurst, conviction, win rate)
- Lookahead bar counts and time windows
- Score adjustment values
- Any literal that would need a comment to explain what it means

### Constants Must Document Units and Scale
Every constant's comment must state the value's meaning AND its units or scale:
```python
PID_MAX_ADX         = 30.0   # ADX (0–100 scale); above this = structural drive, not PID regime
PID_MIN_OSC_COH     = 0.5    # fraction (0–1); oscillation_coherence floor for PID detection
PID_LOOKAHEAD_BARS  = 40     # bars at 15s resolution = 10 minutes maximum audit window
POINT_VALUE_MNQ     = 2.0    # USD per point for MNQ (Micro Nasdaq futures)
```

### No Inline String Literals for Category Values
Gate labels, pattern type names, regime names, and oracle label names must be
defined as named constants, not scattered as inline strings:
```python
# WRONG
_candidate_gate[id(p)] = 'gate0_r3_snap'
if pattern_type == 'ROCHE_SNAP':

# RIGHT
GATE_LABEL_R3_SNAP  = 'gate0_r3_snap'
PATTERN_ROCHE_SNAP  = 'ROCHE_SNAP'
_candidate_gate[id(p)] = GATE_LABEL_R3_SNAP
if pattern_type == PATTERN_ROCHE_SNAP:
```

### CUDA / GPU — Always Try GPU First, Fall Back Gracefully

This project runs on NVIDIA GPUs via PyTorch (`cu121`) and Numba CUDA.
Every compute-heavy path MUST follow the try-CUDA-first pattern:

```python
# WRONG — CPU only
result = sklearn_kmeans.fit(X)

# RIGHT — CUDA first, sklearn CPU fallback
try:
    if torch.cuda.is_available():
        torch.cuda.synchronize()          # flush pending ops before fit
        torch.cuda.empty_cache()          # prevent OOM from previous step
        model = CUDAKMeans(n_clusters=k)
        model.fit(X_tensor)
    else:
        raise RuntimeError("no CUDA")
except Exception:
    from sklearn.cluster import KMeans
    model = KMeans(n_clusters=k, n_init=3).fit(X)
```

Rules:
- Always call `torch.cuda.synchronize()` + `torch.cuda.empty_cache()` immediately
  before any large GPU allocation (KMeans fit, batch kernel launch).
- Wrap GPU paths in `try/except` with a CPU fallback — never let a CUDA error
  crash the entire training run.
- Use `CUDAKMeans` (in `training/cuda_kmeans.py`) for clustering in the main process.
  Worker subprocesses use sklearn CPU KMeans (`use_cuda=False` branch in
  `FractalClusteringEngine._get_kmeans()`).
- Numba CUDA kernels (in `core/cuda_physics.py`) are called from the physics engine.
  Always guard with `if CUDA_AVAILABLE:` before invoking them.
- All physics arrays must be `np.float64` and `C_CONTIGUOUS` before GPU transfer:
  ```python
  arr = data.values.astype(np.float64)
  if not arr.flags['C_CONTIGUOUS']:
      arr = np.ascontiguousarray(arr)
  ```

### Vectorize — No Per-Bar Python Loops in Hot Paths

`batch_compute_states()` processes an entire day (~5,300 bars) at once using NumPy
arrays. Never add a Python `for` loop over bars inside the physics engine.

```python
# WRONG — one bar at a time
for i, row in day_data.iterrows():
    z = (row['close'] - center) / sigma

# RIGHT — vectorized
z_scores = (closes - center) / sigma        # shape: (n,)
```

Use vectorized NumPy/PyTorch operations for rolling statistics, PID terms,
oscillation coherence, and all other per-bar quantities computed in the engine.

### Asset Constants — Always Use `self.asset`, Never Hardcode

All asset-specific values come from the `Asset` object passed to the orchestrator.
Never hardcode instrument parameters:

```python
# WRONG
pnl = price_diff * 2.0          # hardcoded MNQ point value
if price % 0.25 == 0:           # hardcoded tick size

# RIGHT
pnl = price_diff * self.asset.point_value
tick = self.asset.tick_size
```

Known asset values (for reference / test fixtures only):
- MNQ point value: `2.0` USD/point
- MNQ tick size: `0.25` points
- NQ point value: `20.0` USD/point

### File I/O — Always UTF-8, Never Bare Open

All file reads and writes must specify `encoding='utf-8'`. Windows CP1252 will
silently corrupt non-ASCII characters in log files and reports:

```python
# WRONG
with open(path, 'w') as f: ...

# RIGHT
with open(path, 'w', encoding='utf-8') as f: ...
with open(path, 'r', encoding='utf-8') as f: ...
```

CSV files: always pass `newline=''` to `open()` when using `csv.writer`.
Never write non-ASCII characters to any file in this project.

### Frozen Dataclasses — Don't Break `__hash__` / `__eq__`

`ThreeBodyQuantumState` is a frozen dataclass with custom `__hash__` and `__eq__`.
When adding new fields:
1. Add with a default value so existing constructors don't break.
2. Include the field in `__hash__` if it participates in state identity.
3. Include the field in `__eq__` for consistency.
4. Add it to the default constructor call near the bottom of the file.
5. Never add mutable objects (lists, dicts) as fields — frozen dataclasses
   cannot contain mutable types.

### Checkpoints — JSON for Config, Pickle for Objects

- Human-readable config/weights (depth weights, DOE params): save as `.json`
  using `json.dump` with `indent=2`.
- Python objects (scaler, binner, pattern library): save as `.pkl` using `pickle`.
- Never save trained models or scalers as JSON — use pickle.
- Always load with a `try/except FileNotFoundError` and a sensible default.

### Suppress Known Benign Warnings at Module Top

Suppress Numba and PyTorch performance warnings that spam the log without
indicating real problems. Do this at the top of the module, not inline:

```python
import warnings
warnings.filterwarnings("ignore",
    message=".*Grid size.*will likely result in GPU under-utilization.*")
```

Do NOT suppress actual errors or unknown warnings.

## 7. Technical Context

*   **Dependencies**:
    -   `numpy < 2`: Required for compatibility.
    -   `torch`: Used for tensor operations, configured for CUDA (`cu121`) where available.
    -   `numba`: Used for JIT compilation acceleration.
    -   `pandas` & `pandas_ta`: Data manipulation and technical analysis.
*   **Hardware**: The system is optimized for NVIDIA GPUs (CUDA) but will fallback to CPU if necessary. Use `scripts/gpu_health_check.py` to verify GPU status.
