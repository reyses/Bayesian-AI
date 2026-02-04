# Bayesian-AI System Logic

This document provides a comprehensive breakdown of the Bayesian-AI trading system's logic, consolidating high-level architecture, detailed module inner workings, and technical parameters.

## 1. System Architecture Overview

The system operates on a strict **LOAD -> TRANSFORM -> ANALYZE -> VISUALIZE** pipeline. It utilizes a 9-layer hierarchical state model to capture market conditions across multiple timeframes, from 90 days down to 1 second.

### Core Components
- **`engine_core.py` (BayesianEngine):** The central coordinator that manages the trading session, data aggregation, state computation, and trade execution.
- **`core/layer_engine.py` (LayerEngine):** Responsible for computing the 9-layer state vector. It separates computation into **Static** (L1-L4) and **Fluid** (L5-L9) contexts.
- **`core/bayesian_brain.py` (BayesianBrain):** The learning module that maintains a probability table mapping state vectors to win rates.
- **`training/orchestrator.py` (TrainingOrchestrator):** Manages historical data replay to train the Bayesian Brain.
- **`execution/wave_rider.py` (WaveRider):** Handles trade execution, position management, and exit logic.

---

## 2. The 9-Layer Hierarchy

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

---

## 3. Learning Function (Training)

The learning process involves replaying historical data to populate the `BayesianBrain`'s probability table. This is orchestrated by `training/orchestrator.py`.

### Workflow
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

### Optimization Modes
- **Grid Search:** Systematically tests combinations of `min_prob`, `min_conf`, etc.
- **Walk-Forward:** Trains on a window (e.g., Week 1), tests on the next (Week 2), moves forward.
- **Monte Carlo:** Randomly samples contiguous data blocks to test robustness.

---

## 4. Live Trading Function

In live mode, the system uses the pre-trained `BayesianBrain` to make real-time decisions.

### Workflow
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

---

## 5. CUDA Acceleration Modules

The system delegates computationally intensive tasks to the GPU via Numba.

### Pattern Detector (L7)
- **File:** `cuda_modules/pattern_detector.py`
- **Logic:** Parallelizes pattern checks over bar windows.
- **Optimization:** Transfers only the last 200 bars (`LOOKBACK`) to GPU to minimize latency.
- **Priorities:** Checks **Compression** first (highest priority), then **Wedge**, then **Breakdown**.

### Confirmation Engine (L8)
- **File:** `cuda_modules/confirmation.py`
- **Logic:** Vectorized volume analysis.
- **Threshold:** Volume > 1.2 * Mean(Last 3 Volumes).
- **Optimization:** Transfers last 100 volume points.

### Velocity Gate (L9)
- **File:** `cuda_modules/velocity_gate.py`
- **Logic:** High-speed cascade detection on raw ticks.
- **Kernel:** Scans last 50 ticks per thread.
- **Condition:** Absolute price move ≥ 10.0 points within ≤ 0.5 seconds.
- **Optimization:** Only processes recent tick buffer.

---

## 6. Safety & Risk Management

- **Max Daily Loss:** Trading halts if `daily_pnl < MAX_DAILY_LOSS` (default -$200).
- **Kill Zones:** Specific price levels where trading can be restricted or emphasized.
- **Graceful Degradation:** All CUDA modules have explicit CPU fallbacks if `numba.cuda` is unavailable or fails to initialize.
