# Bayesian-AI Technical Manual

This document provides a comprehensive technical reference for the Bayesian-AI trading system, consolidating system logic, architecture, module references, and mathematical foundations.

---

## 1. System Logic & Architecture

The system operates on a strict **LOAD -> TRANSFORM -> ANALYZE -> VISUALIZE** pipeline. It utilizes a 9-layer hierarchical state model to capture market conditions across multiple timeframes, from 90 days down to 1 second.

### Core Components
- **`core/engine_core.py` (BayesianEngine):** The central coordinator that manages the trading session, data aggregation, state computation, and trade execution.
- **`core/layer_engine.py` (LayerEngine):** Responsible for computing the 9-layer state vector. It separates computation into **Static** (L1-L4) and **Fluid** (L5-L9) contexts.
- **`core/bayesian_brain.py` (BayesianBrain):** The learning module that maintains a probability table mapping state vectors to win rates.
- **`training/orchestrator.py` (TrainingOrchestrator):** Manages historical data replay to train the Bayesian Brain.
- **`execution/wave_rider.py` (WaveRider):** Handles trade execution, position management, and exit logic.

### The 9-Layer Hierarchy

The system's "vision" is defined by a 9-layer state vector. A trade is only considered when a high-probability state is identified.

| Layer | Timeframe | Description | Computation Type | Logic / Thresholds |
|-------|-----------|-------------|------------------|-------------------|
| **L1** | 90-Day | **Bias** (Bull/Bear/Range) | Static (CPU) | **Bull:** > +5% change.<br>**Bear:** < -5% change.<br>**Range:** Within ±5%. |
| **L2** | 30-Day | **Regime** (Trending/Chopping) | Static (CPU) | **Trending:** Recent 5-day range > 1.5x of 30-day avg range.<br>**Chopping:** Otherwise. |
| **L3** | 1-Week | **Swing** (Structure) | Static (CPU) | Checks last 4 weeks for Higher Highs/Lower Lows.<br>Else Sideways. |
| **L4** | Daily | **Zone** (Support/Resistance) | Static (CPU) | **Support:** Closer to 5-day Low.<br>**Resistance:** Closer to 5-day High.<br>**Killzone:** Within tolerance of user zones.<br>**Mid:** In between. |
| **L5** | 4-Hour | **Trend** (Direction) | Fluid (CPU) | **Up:** Last 3 closes ascending.<br>**Down:** Last 3 closes descending.<br>**Flat:** Mixed. |
| **L6** | 1-Hour | **Structure** (Candle mix) | Fluid (CPU) | **Bullish:** Bull candles > 1.5x Bear candles.<br>**Bearish:** Bear candles > 1.5x Bull candles.<br>**Neutral:** Balanced. |
| **L7** | 15-Min | **Pattern** (Recognition) | Fluid (**CUDA**) | **Compression:** Recent range < 70% of prev range.<br>**Wedge:** Converging highs/lows.<br>**Breakdown:** New low below prior support.<br>*Priority: Compression > Wedge > Breakdown.* |
| **L8** | 5-Min | **Confirmation** (Volume) | Fluid (**CUDA**) | **Confirmed:** Current volume > 1.2x mean of last 3 volumes.<br>**Unconfirmed:** Otherwise. |
| **L9** | 1-Sec | **Velocity** (Cascade) | Fluid (**CUDA**) | **Cascade:** ≥ 10 points movement in ≤ 0.5 seconds.<br>(Uses sliding window of last 50 ticks). |

### Learning Function (Training)

The learning process involves replaying historical data to populate the `BayesianBrain`'s probability table. This is orchestrated by `training/orchestrator.py`.

#### Workflow
1.  **Data Loading:**
    - Loads `.dbn` (Databento) or `.parquet` files.
    - Concatenates multiple files and sorts by timestamp.
    - Resamples tick data to 1-second OHLC for static context generation if needed.

2.  **Static Context Initialization:**
    - The `LayerEngine` computes L1-L4 based on the entire historical dataset up to the start point.

3.  **Iteration Loop:**
    - The system runs for a specified number of `iterations` (default: 1000).
    - For each iteration:
        - **Tick Simulation:** Processes historical ticks sequentially.
        - **State Computation:** `LayerEngine` computes the current 9-layer state.
        - **Trade Simulation:**
            - If `L9_cascade` triggers AND `BayesianBrain` confirms high probability, a trade is simulated.
        - **Outcome Recording:** WIN/LOSS results update the `BayesianBrain`.

4.  **Bayesian Update Logic (`bayesian_brain.py`):**
    - **Probability:** Calculated using Laplace Smoothing:
      $$ P = \frac{\text{wins} + 1}{\text{total} + 2} $$
    - **Confidence:** Based on sample size (saturation at 30 samples):
      $$ C = \min\left(\frac{\text{total}}{30.0}, 1.0\right) $$

#### Optimization Modes
- **Grid Search:** Systematically tests combinations of `min_prob`, `min_conf`, etc.
- **Walk-Forward:** Trains on a window (e.g., Week 1), tests on the next (Week 2), moves forward.
- **Monte Carlo:** Randomly samples contiguous data blocks to test robustness.

### Live Trading Function

In live mode, the system uses the pre-trained `BayesianBrain` to make real-time decisions.

#### Workflow
1.  **Initialization:**
    - `BayesianEngine` loads `probability_table.pkl`.
    - `LayerEngine` initializes L1-L4 using recent history.
    - Auto-detects GPU availability; falls back to CPU if necessary.

2.  **Real-Time Tick Processing:**
    - **Aggregation:** Ticks -> 1s, 5m, 15m, 1h, 4h bars.
    - **Computation:** L5-L6 (CPU), L7-L9 (CUDA).

3.  **Decision Logic:**
    - **Entry Trigger:**
        1.  **L9 Velocity:** `True` (Cascade detected).
        2.  **Bayesian Check:** `Probability >= MIN_PROB` (0.80) AND `Confidence >= MIN_CONF` (0.30).
    - **Execution:** `WaveRider.open_position()`.

4.  **Position Management (`WaveRider`):**
    - **Initial Stop Loss:** Entry ± 20 ticks.
    - **Adaptive Trailing Stop:**
        - Profit < $50: Trail 10 ticks.
        - Profit < $150: Trail 20 ticks.
        - Profit >= $150: Trail 30 ticks.
    - **Exits:**
        - Stop Loss Hit.
        - Trailing Stop Hit.
        - **Structure Break:** L7 pattern changes or L8 volume confirmation lost.

### CUDA Acceleration Modules

The system delegates computationally intensive tasks to the GPU via Numba.

#### Pattern Detector (L7)
- **File:** `cuda_modules/pattern_detector.py`
- **Logic:** Parallelizes pattern checks over bar windows.
- **Optimization:** Transfers only the last 200 bars (`LOOKBACK`) to GPU to minimize latency.
- **Priorities:** Checks **Compression** first (highest priority), then **Wedge**, then **Breakdown**.

#### Confirmation Engine (L8)
- **File:** `cuda_modules/confirmation.py`
- **Logic:** Vectorized volume analysis.
- **Threshold:** Volume > 1.2 * Mean(Last 3 Volumes).
- **Optimization:** Transfers last 100 volume points.

#### Velocity Gate (L9)
- **File:** `cuda_modules/velocity_gate.py`
- **Logic:** High-speed cascade detection on raw ticks.
- **Kernel:** Scans last 50 ticks per thread.
- **Condition:** Absolute price move ≥ 10.0 points within ≤ 0.5 seconds.
- **Optimization:** Only processes recent tick buffer.

### Safety & Risk Management

- **Max Daily Loss:** Trading halts if `daily_pnl < MAX_DAILY_LOSS` (default -$200).
- **Kill Zones:** Specific price levels where trading can be restricted or emphasized.
- **Graceful Degradation:** All CUDA modules have explicit CPU fallbacks if `numba.cuda` is unavailable or fails to initialize.

---

## 2. Project Map & Module Reference

### File Structure

```text
.
├── AGENTS.md               # Instructions for AI agents
├── CURRENT_STATUS.md       # Live project health and code stats
├── DATA/                   # Data storage
│   └── RAW/                # Raw market data (DBN/Parquet)
├── README.md               # Project Entry Point
├── config/                 # Configuration and Constants
│   ├── settings.py
│   ├── symbols.py
│   └── workflow_manifest.json
├── core/                   # Core Logic Engines
│   ├── bayesian_brain.py
│   ├── data_aggregator.py
│   ├── engine_core.py      # (Entry Point Alias)
│   ├── layer_engine.py
│   └── state_vector.py
├── cuda_modules/           # GPU-Accelerated Components (Numba)
│   ├── confirmation.py
│   ├── hardened_verification.py
│   ├── pattern_detector.py
│   └── velocity_gate.py
├── docs/                   # Documentation
│   ├── CHANGELOG.md
│   ├── LEARNING_DASHBOARD_GUIDE.md
│   ├── TECHNICAL_MANUAL.md
│   └── archive/
├── execution/              # Trade Execution & Management
│   └── wave_rider.py
├── notebooks/              # Jupyter Notebooks
│   └── learning_dashboard.ipynb
├── requirements.txt        # Python Dependencies
├── scripts/                # Utility & Build Scripts
│   ├── build_executable.py
│   ├── generate_learning_dashboard.py
│   ├── generate_status_report.py
│   ├── inspect_results.py
│   ├── manifest_integrity_check.py
│   ├── sentinel_bridge.py
│   ├── setup_test_data.py
│   └── verify_environment.py
├── tests/                  # Test Suite
│   ├── test_*.py
│   └── topic_*.py
└── training/               # Historical Learning Pipeline
    ├── databento_loader.py
    └── orchestrator.py
```

### Module Reference

#### 2.1 Core Modules (`core/`)

**`core/engine_core.py` (BayesianEngine)**
*   **Purpose:** Central coordinator that ties together Data, Logic, Learning, and Execution.
*   **Key Methods:**
    *   `initialize_session(historical_data, user_kill_zones)`: Pre-computes Static Context (L1-L4).
    *   `on_tick(tick_data)`: Main event loop. Aggregates data, computes state, manages positions, and checks entry conditions.
    *   `_close_position(price, info)`: Finalizes trades, updates PnL, and sends outcomes to the Brain.
*   **Key Variables:** `daily_pnl`, `MAX_DAILY_LOSS` (-200), `MIN_PROB` (0.80), `MIN_CONF` (0.30).

**`layer_engine.py` (LayerEngine)**
*   **Purpose:** Computes the 9-layer `StateVector` from market data. Separates Static (session-start) from Fluid (real-time) layers.
*   **Key Methods:**
    *   `initialize_static_context(...)`: Computes L1-L4 (Bias, Regime, Swing, Zone) once.
    *   `compute_current_state(current_data)`: Computes L5-L9 (Trend, Structure, Pattern, Confirm, Cascade) every tick.
    *   `_compute_L7_15m(...)`: Delegates to `pattern_detector`.
    *   `_compute_L9_1s(...)`: Delegates to `velocity_gate`.

**`bayesian_brain.py` (BayesianBrain)**
*   **Purpose:** Learns win probabilities for state vectors. Uses a HashMap (`StateVector` -> `TradeStats`).
*   **Key Methods:**
    *   `update(outcome)`: Updates `wins`/`losses` for a specific state.
    *   `get_probability(state)`: Returns `(wins + 1) / (total + 2)` (Laplace Smoothing).
    *   `get_confidence(state)`: Returns `min(total / 30.0, 1.0)`.
    *   `should_fire(state)`: Returns True if Prob and Conf meet thresholds.
    *   `save/load(filepath)`: Persists knowledge to `probability_table.pkl`.

**`state_vector.py` (StateVector)**
*   **Purpose:** Immutable hashable object representing the market state.
*   **Fields:** `L1_bias`, `L2_regime`, `L3_swing`, `L4_zone`, `L5_trend`, `L6_structure`, `L7_pattern`, `L8_confirm`, `L9_cascade`.
*   **Logic:** `__hash__` and `__eq__` strictly exclude metadata (`timestamp`, `price`) to ensure identical market conditions map to the same key.

**`data_aggregator.py` (DataAggregator)**
*   **Purpose:** Manages a ring buffer of ticks and generates OHLC bars on demand.
*   **Key Methods:**
    *   `add_tick(tick)`: Inserts new tick into numpy buffers.
    *   `get_current_data()`: Returns a snapshot dict with 'ticks', 'bars_5m', 'bars_15m', etc.
    *   `_get_ordered_data()`: Reconstructs chronological arrays from ring buffer.

#### 2.2 CUDA Modules (`cuda_modules/`)

**`pattern_detector.py` (CUDAPatternDetector)**
*   **Purpose:** L7 Pattern Recognition on 15m bars.
*   **Methods:** `detect(bars)`, `_detect_gpu`, `_detect_cpu`.
*   **Logic:** Prioritizes **Compression** (Range < 70% prev) > **Wedge** > **Breakdown**.
*   **Optimization:** Transfers only last 200 bars (`LOOKBACK`) to GPU.

**`confirmation.py` (CUDAConfirmationEngine)**
*   **Purpose:** L8 Volume Confirmation on 5m bars.
*   **Methods:** `confirm(bars)`, `confirm_kernel` (Numba).
*   **Logic:** Checks if current volume > 1.2x mean of last 3 volumes.

**`velocity_gate.py` (CUDAVelocityGate)**
*   **Purpose:** L9 Cascade Detection on ticks.
*   **Methods:** `detect_cascade(tick_data)`.
*   **Logic:** Checks if price moves ≥ 10 points in ≤ 0.5s within the last 50 ticks.
*   **Optimization:** Processes only relevant tail of tick data.

#### 2.3 Execution (`execution/`)

**`wave_rider.py` (WaveRider)**
*   **Purpose:** Manages active positions.
*   **Key Methods:**
    *   `open_position(entry_price, side, state)`: Sets initial Stop Loss (20 ticks).
    *   `update_trail(current_price, current_state)`: Updates High Water Mark and calculates Adaptive Trailing Stop.
*   **Adaptive Trail:**
    *   Profit < $50: Trail 10 ticks.
    *   Profit < $150: Trail 20 ticks.
    *   Profit >= $150: Trail 30 ticks.
*   **Exits:** Stop Loss, Trailing Stop, or Structure Break (L7 change / L8 loss).

#### 2.4 Training (`training/`)

**`orchestrator.py` (TrainingOrchestrator)**
*   **Purpose:** Runs backtests/training loops.
*   **Key Methods:**
    *   `run_training(iterations)`: Main loop. Replays data -> `BayesianEngine.on_tick`.
    *   `run_grid_search(param_grid)`: Optimizes `min_prob`, `min_conf`, etc.
    *   `run_walk_forward(...)`: Tests robustness over sliding time windows.
    *   `run_monte_carlo(...)`: Randomly samples data blocks.

**`databento_loader.py` (DatabentoLoader)**
*   **Purpose:** Loads `.dbn` files.
*   **Logic:** Normalizes columns to `['timestamp', 'price', 'volume', 'type']`. Preserves OHLC if present. Filters for trade events (`action='T'`).

### Configuration & Constants (`config/`)

**`settings.py`**
*   `OPERATIONAL_MODE`: "LEARNING" (Default) or "EXECUTE".
*   `RAW_DATA_PATH`: "DATA/RAW".

**`symbols.py`**
*   Defines asset profiles (Tick Size, Point Value, Fee).
*   **MNQ:** Tick 0.25, Point $2, Fee $0.50.
*   **ES:** Tick 0.25, Point $50, Fee $2.00.

### Global Variables & Thresholds

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

---

## 3. Mathematical & Data Report

### Mathematical Application

The Bayesian-AI system applies mathematics primarily through its **Bayesian Probability Engine** and **High-Frequency State Analysis**.

#### A. Bayesian Probability (`core/bayesian_brain.py`)

The core decision-making logic relies on Bayesian inference to estimate the probability of a "WIN" outcome given a specific market state.

1.  **State Representation**:
    The market is discretized into a unique `StateVector` composed of 9 hierarchical layers (L1-L9).
    $$ S = \{L1, L2, L3, L4, L5, L6, L7, L8, L9\} $$
    This vector serves as the key for the probability lookup table.

2.  **Probability Estimation**:
    The system calculates the win probability $P(Win|S)$ using **Laplace Smoothing** to handle small sample sizes and avoid zero-probability issues.

    $$ P(Win|S) = \frac{Wins + 1}{Total\_Trials + 2} $$

    *   **Prior**: With 0 data, $P(Win|S) = \frac{0+1}{0+2} = 0.5$ (Neutral 50%).
    *   **Updates**: As trades execute, the counts ($Wins$, $Total$) are updated in the `probability_table.pkl`.

3.  **Confidence Metric**:
    A confidence score determines if the sample size is sufficient to trust the probability estimate.

    $$ C(S) = \min\left(\frac{Total\_Trials}{30}, 1.0\right) $$

    *   Full confidence (1.0) is achieved at **30 samples**.
    *   A trade is only executed if $P(Win|S) \ge 0.80$ and $C(S) \ge 0.30$.

#### B. High-Frequency Math (`cuda/velocity_gate.py`)

The **L9 Velocity Gate** prevents execution during adverse high-volatility events (cascades) using a sliding window algorithm.

1.  **Cascade Detection**:
    The system analyzes the last **50 ticks** to detect rapid price displacements.

    $$ \Delta P = | \max(P_{window}) - \min(P_{window}) | $$
    $$ \Delta t = t_{end} - t_{start} $$

    A cascade is flagged if:
    $$ \Delta P \ge 10.0 \text{ points} \quad \text{AND} \quad \Delta t \le 0.5 \text{ seconds} $$

2.  **Optimization**:
    To maintain $O(1)$ performance on high-frequency data streams, the system creates a localized slice (buffer) of the last **200 ticks** before passing data to the CPU/GPU, ensuring processing time does not scale with total history size.

### Databento File Usage

The system is designed to ingest and process high-fidelity market data from **Databento** (`.dbn` files).

#### A. Ingestion Pipeline (`training/databento_loader.py`)

1.  **Loading**:
    The system uses the `databento` Python library to read compressed Databento files (`.dbn.zst`).
    ```python
    data = db.DBNStore.from_file(filepath)
    df = data.to_df()
    ```

2.  **Normalization**:
    Raw Databento columns are mapped to the internal schema:
    *   `ts_event` / `ts_recv` $\rightarrow$ `timestamp` (converted to UNIX float seconds)
    *   `price` / `close` $\rightarrow$ `price`
    *   `size` / `volume` $\rightarrow$ `volume`
    *   `action` / `side` $\rightarrow$ `type`

    *Update*: The loader was enhanced to preserve OHLC columns (`open`, `high`, `low`) when processing OHLCV data, ensuring compatibility with Layer 1-4 static context generation.

#### B. Data Flow

1.  **Training Mode**:
    *   **Input**: Historical `.dbn` files or `.parquet` archives in `DATA/RAW`.
    *   **Process**: `TrainingOrchestrator` iterates through ticks, simulating the `BayesianEngine` response to build the probability table.
    *   **Output**: A trained `probability_table.pkl`.

2.  **Verification**:
    *   Scripts like `scripts/setup_test_data.py` convert raw Databento trades into standardized Parquet files (`trades.parquet`, `ohlcv-1s.parquet`) to facilitate rapid integration testing without re-parsing large `.dbn` files repeatedly.
