# Bayesian-AI Project Map

This document serves as a comprehensive technical reference for the Bayesian-AI trading system, detailing the file structure, module responsibilities, classes, functions, and key configuration parameters.

## 1. File Structure

```text
.
├── config/                 # Configuration and Constants
│   ├── settings.py         # Global system settings (Operational Mode, Paths)
│   ├── symbols.py          # Ticker symbol definitions and mappings
│   └── workflow_manifest.json
├── core/                   # Core Logic Engines
│   ├── engine_core.py      # (Root) Main BayesianEngine class
│   ├── layer_engine.py     # 9-Layer State Computation (Static & Fluid)
│   ├── bayesian_brain.py   # Probability Learning & Persistence
│   ├── data_aggregator.py  # Real-time Tick/Bar Data Management
│   └── state_vector.py     # Immutable State Data Structure
├── cuda_modules/           # GPU-Accelerated Components (Numba)
│   ├── pattern_detector.py # L7 Pattern Recognition
│   ├── confirmation.py     # L8 Volume Confirmation
│   ├── velocity_gate.py    # L9 Cascade Detection
│   └── hardened_verification.py # GPU Audit & Verification
├── execution/              # Trade Execution & Management
│   └── wave_rider.py       # Position Entry, Exit, and Trailing Stops
├── training/               # Historical Learning Pipeline
│   ├── orchestrator.py     # Training Orchestrator (Grid Search, Walk-Forward)
│   └── databento_loader.py # Data Ingestion & Normalization
├── scripts/                # Utility & Build Scripts
│   ├── generate_status_report.py
│   ├── build_executable.py
│   ├── sentinel_bridge.py
│   └── ...
├── tests/                  # Test Suite
│   ├── topic_math.py
│   ├── topic_diagnostics.py
│   ├── test_phase1.py
│   └── ...
├── SYSTEM_LOGIC.md         # Detailed System Logic Documentation
└── engine_core.py          # (Entry Point Alias)
```

---

## 2. Module Reference

### 2.1 Core Modules (`core/`)

#### `engine_core.py`
**Class: `BayesianEngine`**
*   **Purpose:** Central coordinator that ties together Data, Logic, Learning, and Execution.
*   **Key Methods:**
    *   `initialize_session(historical_data, user_kill_zones)`: Pre-computes Static Context (L1-L4).
    *   `on_tick(tick_data)`: Main event loop. Aggregates data, computes state, manages positions, and checks entry conditions.
    *   `_close_position(price, info)`: Finalizes trades, updates PnL, and sends outcomes to the Brain.
*   **Key Variables:** `daily_pnl`, `MAX_DAILY_LOSS` (-200), `MIN_PROB` (0.80), `MIN_CONF` (0.30).

#### `layer_engine.py`
**Class: `LayerEngine`**
*   **Purpose:** Computes the 9-layer `StateVector` from market data. Separates Static (session-start) from Fluid (real-time) layers.
*   **Key Methods:**
    *   `initialize_static_context(...)`: Computes L1-L4 (Bias, Regime, Swing, Zone) once.
    *   `compute_current_state(current_data)`: Computes L5-L9 (Trend, Structure, Pattern, Confirm, Cascade) every tick.
    *   `_compute_L7_15m(...)`: Delegates to `pattern_detector`.
    *   `_compute_L9_1s(...)`: Delegates to `velocity_gate`.

#### `bayesian_brain.py`
**Class: `BayesianBrain`**
*   **Purpose:** Learns win probabilities for state vectors. Uses a HashMap (`StateVector` -> `TradeStats`).
*   **Key Methods:**
    *   `update(outcome)`: Updates `wins`/`losses` for a specific state.
    *   `get_probability(state)`: Returns `(wins + 1) / (total + 2)` (Laplace Smoothing).
    *   `get_confidence(state)`: Returns `min(total / 30.0, 1.0)`.
    *   `should_fire(state)`: Returns True if Prob and Conf meet thresholds.
    *   `save/load(filepath)`: Persists knowledge to `probability_table.pkl`.

#### `state_vector.py`
**Class: `StateVector`** (Dataclass)
*   **Purpose:** Immutable hashable object representing the market state.
*   **Fields:** `L1_bias`, `L2_regime`, `L3_swing`, `L4_zone`, `L5_trend`, `L6_structure`, `L7_pattern`, `L8_confirm`, `L9_cascade`.
*   **Logic:** `__hash__` and `__eq__` strictly exclude metadata (`timestamp`, `price`) to ensure identical market conditions map to the same key.

#### `data_aggregator.py`
**Class: `DataAggregator`**
*   **Purpose:** Manages a ring buffer of ticks and generates OHLC bars on demand.
*   **Key Methods:**
    *   `add_tick(tick)`: Inserts new tick into numpy buffers.
    *   `get_current_data()`: Returns a snapshot dict with 'ticks', 'bars_5m', 'bars_15m', etc.
    *   `_get_ordered_data()`: Reconstructs chronological arrays from ring buffer.

### 2.2 CUDA Modules (`cuda_modules/`)

#### `pattern_detector.py`
**Class: `CUDAPatternDetector`**
*   **Purpose:** L7 Pattern Recognition on 15m bars.
*   **Methods:** `detect(bars)`, `_detect_gpu`, `_detect_cpu`.
*   **Logic:** Prioritizes **Compression** (Range < 70% prev) > **Wedge** > **Breakdown**.
*   **Optimization:** Transfers only last 200 bars (`LOOKBACK`) to GPU.

#### `confirmation.py`
**Class: `CUDAConfirmationEngine`**
*   **Purpose:** L8 Volume Confirmation on 5m bars.
*   **Methods:** `confirm(bars)`, `confirm_kernel` (Numba).
*   **Logic:** Checks if current volume > 1.2x mean of last 3 volumes.

#### `velocity_gate.py`
**Class: `CUDAVelocityGate`**
*   **Purpose:** L9 Cascade Detection on ticks.
*   **Methods:** `detect_cascade(tick_data)`.
*   **Logic:** Checks if price moves ≥ 10 points in ≤ 0.5s within the last 50 ticks.
*   **Optimization:** Processes only relevant tail of tick data.

### 2.3 Execution (`execution/`)

#### `wave_rider.py`
**Class: `WaveRider`**
*   **Purpose:** Manages active positions.
*   **Key Methods:**
    *   `open_position(entry_price, side, state)`: Sets initial Stop Loss (20 ticks).
    *   `update_trail(current_price, current_state)`: Updates High Water Mark and calculates Adaptive Trailing Stop.
*   **Adaptive Trail:**
    *   Profit < $50: Trail 10 ticks.
    *   Profit < $150: Trail 20 ticks.
    *   Profit >= $150: Trail 30 ticks.
*   **Exits:** Stop Loss, Trailing Stop, or Structure Break (L7 change / L8 loss).

### 2.4 Training (`training/`)

#### `orchestrator.py`
**Class: `TrainingOrchestrator`**
*   **Purpose:** Runs backtests/training loops.
*   **Key Methods:**
    *   `run_training(iterations)`: Main loop. Replays data -> `BayesianEngine.on_tick`.
    *   `run_grid_search(param_grid)`: Optimizes `min_prob`, `min_conf`, etc.
    *   `run_walk_forward(...)`: Tests robustness over sliding time windows.
    *   `run_monte_carlo(...)`: Randomly samples data blocks.

#### `databento_loader.py`
**Class: `DatabentoLoader`**
*   **Purpose:** Loads `.dbn` files.
*   **Logic:** Normalizes columns to `['timestamp', 'price', 'volume', 'type']`. Preserves OHLC if present. Filters for trade events (`action='T'`).

---

## 3. Configuration & Constants (`config/`)

#### `settings.py`
*   `OPERATIONAL_MODE`: "LEARNING" (Default) or "EXECUTE".
*   `RAW_DATA_PATH`: "DATA/RAW".

#### `symbols.py`
*   Defines asset profiles (Tick Size, Point Value, Fee).
*   **MNQ:** Tick 0.25, Point $2, Fee $0.50.
*   **ES:** Tick 0.25, Point $50, Fee $2.00.

---

## 4. Global Variables & Thresholds

| Component | Variable | Value | Description |
|-----------|----------|-------|-------------|
| **BayesianEngine** | `MAX_DAILY_LOSS` | -200.0 | Daily stop loss limit ($) |
| **BayesianEngine** | `MIN_PROB` | 0.80 | Minimum win probability to trade |
| **BayesianEngine** | `MIN_CONF` | 0.30 | Minimum confidence score |
| **LayerEngine** | `L1_bias` | ±5% | 90-day change threshold |
| **LayerEngine** | `L2_regime` | 1.5x | Trending vs Chopping range multiplier |
| **VelocityGate** | `cascade_threshold` | 10.0 | Points moved for cascade |
| **VelocityGate** | `time_window` | 0.5 | Time (s) for cascade move |
| **WaveRider** | `stop_dist` | 20 ticks | Initial stop loss distance |
