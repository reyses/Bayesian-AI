# Bayesian-AI System Logic Manual

This manual provides a detailed breakdown of the Bayesian-AI trading system's logic, covering both the **Learning (Training)** and **Live Trading** functions.

## 1. System Architecture Overview

The system operates on a **Load -> Transform -> Analyze -> Visualize** pipeline. It utilizes a 9-layer hierarchical state model to capture market conditions across multiple timeframes, from 90 days down to 1 second.

### Core Components
- **`engine_core.py` (BayesianEngine):** The central coordinator that manages the trading session, data aggregation, state computation, and trade execution.
- **`core/layer_engine.py` (LayerEngine):** Responsible for computing the 9-layer state vector. It separates computation into **Static** (L1-L4) and **Fluid** (L5-L9) contexts.
- **`core/bayesian_brain.py` (BayesianBrain):** The learning module that maintains a probability table mapping state vectors to win rates.
- **`training/orchestrator.py` (TrainingOrchestrator):** Manages historical data replay to train the Bayesian Brain.
- **`execution/wave_rider.py` (WaveRider):** Handles trade execution, position management, and exit logic.

---

## 2. The 9-Layer Hierarchy

The system's "vision" is defined by a 9-layer state vector. A trade is only considered when a high-probability state is identified.

| Layer | Timeframe | Description | Computation Type | Component |
|-------|-----------|-------------|------------------|-----------|
| **L1** | 90-Day | **Bias** (Bull/Bear/Range) | Static (CPU) | `LayerEngine` |
| **L2** | 30-Day | **Regime** (Trending/Chopping) | Static (CPU) | `LayerEngine` |
| **L3** | 1-Week | **Swing** (Higher Highs/Lower Lows) | Static (CPU) | `LayerEngine` |
| **L4** | Daily | **Zone** (Support/Resistance/Killzone) | Static (CPU) | `LayerEngine` |
| **L5** | 4-Hour | **Trend** (Up/Down/Flat) | Fluid (CPU) | `LayerEngine` |
| **L6** | 1-Hour | **Structure** (Bullish/Bearish/Neutral) | Fluid (CPU) | `LayerEngine` |
| **L7** | 15-Min | **Pattern** (Recognition) | Fluid (**CUDA**) | `pattern_detector.py` |
| **L8** | 5-Min | **Confirmation** (Volume/Structure) | Fluid (**CUDA**) | `confirmation.py` |
| **L9** | 1-Sec | **Velocity** (Cascade Detection) | Fluid (**CUDA**) | `velocity_gate.py` |

---

## 3. Learning Function (Training)

The learning process involves replaying historical data to populate the `BayesianBrain`'s probability table. This is orchestrated by `training/orchestrator.py`.

### Workflow
1.  **Data Loading:**
    - Historical data (Parquet or Databento `.dbn`) is loaded.
    - Data is sanitized (NaNs handled, types enforced).
    - If needed, tick data is resampled to 1-second OHLC for static context generation.

2.  **Static Context Initialization:**
    - The `LayerEngine` computes L1-L4 based on the entire historical dataset up to the start point.
    - **L1 (Bias):** Calculates % change over the last 90 days. (>5% Bull, <-5% Bear).
    - **L2 (Regime):** Compares recent 5-day range vs. 30-day average range. (>1.5x = Trending).
    - **L3 (Swing):** Analyzes the last 4 weekly candles for HH/LL structure.
    - **L4 (Zone):** Identifies if price is near support (5-day low), resistance (5-day high), or user-defined "Kill Zones".

3.  **Iteration Loop:**
    - The system runs for a specified number of `iterations` (default: 1000).
    - For each iteration:
        - The `BayesianEngine` is reset.
        - **Tick Simulation:** The engine processes historical ticks sequentially via `on_tick()`.
        - **State Computation:** For every tick, `LayerEngine` computes the current 9-layer state.
        - **Trade Simulation:**
            - If `L9_cascade` (Velocity Gate) triggers AND the state has high probability/confidence in the `BayesianBrain`, a trade is simulated.
            - The trade is managed by `WaveRider` until exit.
        - **Outcome Recording:** When a trade closes, the result (WIN/LOSS) is fed back to `BayesianBrain.update()`.
            - `wins` or `losses` count is incremented for that specific `StateVector`.

4.  **Persistence:**
    - The learned `probability_table` is saved to `probability_table.pkl`.

### Optimization Modes
- **Grid Search:** Systematically tests combinations of parameters (`min_prob`, `min_conf`, `stop_loss`).
- **Walk-Forward:** Trains on a window, tests on the next, moves forward.
- **Monte Carlo:** Randomly samples contiguous data blocks to test robustness.

---

## 4. Live Trading Function

In live mode, the system uses the pre-trained `BayesianBrain` to make real-time decisions.

### Workflow
1.  **Initialization:**
    - `BayesianEngine` loads the `probability_table.pkl`.
    - `LayerEngine` initializes the **Static Context** (L1-L4) using recent historical data.

2.  **Real-Time Tick Processing (`on_tick`):**
    - **Aggregation:** Incoming ticks are added to `DataAggregator`, which maintains real-time OHLC buffers (1s, 5m, 15m, 1h, 4h).
    - **State Computation:** `LayerEngine.compute_current_state()` derives the full 9-layer vector.
        - **L5-L6:** Computed on CPU from aggregated bars.
        - **L7-L9:** Computed on GPU (if available) for speed.

3.  **Decision Logic:**
    - **Entry Condition:**
        1.  **Velocity Trigger:** `L9_cascade` must be TRUE (detected by `velocity_gate.py`).
        2.  **Probability Check:** The current `StateVector` is looked up in `BayesianBrain`.
            - `Probability = wins / total`
            - `Confidence = min(total / 30, 1.0)`
        3.  **Thresholds:** `Probability >= MIN_PROB` (default 0.80) AND `Confidence >= MIN_CONF` (default 0.30).
    - **Execution:** If conditions are met, `WaveRider.open_position()` is called.

4.  **Position Management (`WaveRider`):**
    - **Trailing Stop:** Updates the stop loss based on price movement and market structure.
    - **Exit Logic:**
        - **Stop Loss:** Hard price level hit.
        - **Structure Break:** L6/L7 structure shifts against the trade.
        - **Time Exit:** Trade open too long without progress.

5.  **Learning (Live):**
    - When a live trade closes, `BayesianBrain.update()` is called to reinforce the learning model with real-time outcomes.

---

## 5. CUDA Acceleration

The system delegates computationally intensive tasks to the GPU via Numba:

- **L7 (Pattern Detector):** Parallelizes pattern recognition (e.g., flags, wedges) over sliding windows of 15-minute bars.
- **L8 (Confirmation):** vectorized volume analysis to confirm moves.
- **L9 (Velocity Gate):** High-speed detection of price cascades (rapid tick sequences) on the 1-second timeframe.

---

## 6. Safety & Risk Management

- **Max Daily Loss:** Trading stops if `daily_pnl` drops below `MAX_DAILY_LOSS` (default -$200).
- **Kill Zones:** Specific price levels where trading is either focused or restricted.
- **Confidence Scaling:** Position sizing can be scaled based on the Bayesian confidence score.
