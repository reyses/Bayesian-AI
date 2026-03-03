# Bayesian-AI System Description

> **Algorithmic Trading System with Fractal Three-Body Quantum Model & Bayesian Inference**

---

## 1. Purpose and Overview

Bayesian-AI is a high-frequency algorithmic trading system designed for US equity index futures (NQ, ES, MNQ, MES). It models the market as a **three-body gravitational system** where price is a particle moving between three attractors (fair value, upper resistance, lower support), then applies **Bayesian probability learning** to discover which market states lead to profitable trades.

The system operates across a **multi-timeframe fractal hierarchy** (from 1-day bars down to 1-second ticks), discovers recurring physics-based patterns in historical data, clusters them into reusable templates, optimizes parameters via Design of Experiments (DOE), and executes trades through a consensus-based belief network.

**Primary instrument:** Micro Nasdaq-100 futures (MNQ)
**Operational mode:** LEARNING (parameter optimization) / EXECUTE (live trading)

---

## 2. System Architecture

### 2.1 High-Level Pipeline

```
LOAD -> TRANSFORM -> ANALYZE -> DECIDE -> EXECUTE -> LEARN
```

The full pipeline consists of five phases:

| Phase | Name                  | Description                                                     |
|-------|-----------------------|-----------------------------------------------------------------|
| 1     | Data Preparation      | Converts raw Databento DBN files to Parquet (ATLAS format)      |
| 2     | Pattern Discovery     | Scans history for physics events (Roche Limit, Structural Drive)|
| 3     | Template Optimization | Clusters events into templates; optimizes parameters via DOE    |
| 4     | Forward Pass          | Replays history using the Playbook (WaveRider execution)        |
| 5     | Strategy Selection    | Analyzes results to rank templates (Tier 1-4)                   |

### 2.2 Data Flow

```
Raw Databento DBN files (.dbn.zst)
        |
        v
  [dbn_to_parquet.py] -- Phase 1: Data Prep
        |
        v
  Parquet ATLAS files (OHLCV 1-second bars)
        |
        v
  [QuantumFieldEngine] -- Computes 3-body state for every bar
        |
        v
  ThreeBodyQuantumState (16D+ state vector per bar)
        |
        v
  [FractalDiscoveryAgent] -- Phase 2: Scans for physics events
        |
        v
  PatternEvent list (Roche Snap, Structural Drive, etc.)
        |
        v
  [FractalClusteringEngine] -- Phase 3: K-Means clustering (GPU/CPU)
        |
        v
  Pattern Templates (centroids with optimized parameters)
        |
        v
  [TimeframeBeliefNetwork] -- 8 independent workers vote on direction
        |
        v
  [WaveRider] -- Phase 4: Executes trades with dynamic trailing stops
        |
        v
  Trade Log (oracle_trade_log.csv)
        |
        v
  [BayesianBrain] -- Learns: TemplateID -> Win Probability
        |
        v
  [TradeAnalytics] -- Phase 5: Ranks templates by tier (1=Best, 4=Toxic)
```

---

## 3. Core Components

### 3.1 Quantum Field Engine (`core/quantum_field_engine.py`)

The physics engine that transforms raw OHLCV data into a quantum state representation. It models the market as a **three-body gravitational problem**:

- **Body 1 (Center Star):** Fair value regression line -- attractive force (mean reversion)
- **Body 2 (Upper Singularity):** +2 sigma resistance -- repulsive force
- **Body 3 (Lower Singularity):** -2 sigma support -- repulsive force
- **Particle:** Current price -- exists in superposition between the three attractors

**Key computations:**
- **Z-score:** Normalized distance from fair value in sigma units
- **Force fields:** Tidal reversion force (z^2/9), upper/lower repulsion (1/r^3), momentum (kinetic energy), and net force vector
- **Wave function:** Quantum superposition amplitudes (psi = a0*psi_center + a1*psi_upper + a2*psi_lower)
- **Tunnel probability:** Monte Carlo simulation of mean-reversion likelihood (Ornstein-Uhlenbeck process)
- **Decoherence metrics:** Shannon entropy, coherence (1.0 = superposition, 0.0 = collapsed)
- **Trend indicators:** ADX/DMI, Hurst exponent, PID oscillation detection

**Acceleration:** The engine uses CUDA via Numba kernels (`core/cuda_physics.py`) for the heavy physics calculations, with automatic CPU fallback. Vectorized NumPy operations process entire days of bars at once (`batch_compute_states()`).

### 3.2 Three-Body Quantum State (`core/three_body_state.py`)

A frozen dataclass that encapsulates the complete market state at a single point in time. Each bar produces one `ThreeBodyQuantumState` instance containing:

- **Three attractors:** center position, upper/lower singularity, event horizons (+/-3 sigma)
- **Particle state:** position, velocity, z-score
- **Force fields:** reversion, upper/lower repulsion, momentum, net force
- **Wave function:** complex amplitudes (a0, a1, a2) and probabilities (P_center, P_upper, P_lower)
- **Decoherence:** entropy, coherence, pattern maturity, momentum strength
- **Measurement operators:** structure confirmation (L8), cascade trigger (L9)

State identity is determined via a **Dynamic Binner** that discretizes continuous features (z-score, momentum) into histogram bins using the Freedman-Diaconis rule, enabling hash-based lookups.

### 3.3 Bayesian Brain (`core/bayesian_brain.py`)

A HashMap-based learning engine that maps market states to win probabilities:

```
probability_table[StateKey] = {wins: X, losses: Y, total: Z}
```

- **Input:** `TradeOutcome` objects after each simulated or real trade
- **Key:** Template ID (preferred) or raw state hash
- **Output:** Win probability = wins / total (with optional Bayesian smoothing)
- **Direction-aware:** Maintains separate statistics for LONG vs SHORT per template
- **Persistence:** Serialized to/from pickle files as checkpoints

The `QuantumBayesianBrain` extension adds statistical validation via Monte Carlo risk analysis.

### 3.4 Dynamic Binner (`core/dynamic_binner.py`)

Computes optimal histogram bin edges from observed data using the **Freedman-Diaconis rule** (with Sturges fallback), then maps continuous feature values to their nearest bin center. This ensures that similar market states hash to the same key, enabling the Bayesian Brain to accumulate meaningful statistics.

### 3.5 Risk Engine (`core/risk_engine.py`)

A Monte Carlo simulation engine that estimates:
- **Tunnel Probability:** Likelihood that price reverts to fair value (center)
- **Escape Probability:** Likelihood that price hits the event horizon (+/-3 sigma)

Uses an **Ornstein-Uhlenbeck process** (mean-reverting stochastic model) with 500 simulated paths over a 600-second horizon.

### 3.6 Adaptive Confidence Manager (`core/adaptive_confidence.py`)

Implements progressive tightening of trading criteria through 4 learning phases:

| Phase | Name          | Probability Threshold | Confidence Threshold | Goal                             |
|-------|---------------|-----------------------|----------------------|----------------------------------|
| 1     | EXPLORATION   | 0.00                  | 0.00                 | Build initial probability map    |
| 2     | REFINEMENT    | 0.45                  | 0.20                 | Filter obvious losers            |
| 3     | OPTIMIZATION  | 0.55                  | 0.30                 | Focus on high-probability setups |
| 4     | MASTERY       | 0.80                  | 0.40                 | Elite-only execution             |

Transitions occur after accumulating sufficient trade count (200-400 trades per phase).

---

## 4. Training Pipeline

### 4.1 Training Orchestrator (`training/orchestrator.py`)

The central controller that coordinates the entire training workflow. It supports:

- **Walk-forward training:** Day-by-day chronological simulation with in-sample learning and out-of-sample validation
- **Pattern discovery:** Recursive fractal scanning across 8+ timeframe levels
- **Template optimization:** DOE-based parameter tuning per cluster
- **Forward pass (Phase 4):** Blind replay using frozen templates (no peeking)
- **Dashboard:** Real-time Tkinter visualization of the training manifold
- **Checkpoint management:** Resume capability after interruption

### 4.2 Data Loading (`training/databento_loader.py`, `training/dbn_to_parquet.py`)

- **Source:** Databento DBN files (compressed with zstandard)
- **Instruments:** CME Globex futures (GLBX-MDP3)
- **Resolution:** 1-second OHLCV bars
- **Format:** Converted to Parquet for fast columnar access via PyArrow

### 4.3 Fractal Discovery Agent (`training/fractal_discovery_agent.py`)

Scans each day of historical data for physics-based events:
- **Roche Limit events:** Price approaches a singularity boundary
- **Structural Drive events:** Strong directional force confirmed by ADX/Hurst
- **Compression events:** Volatility squeeze detected via geometric patterns

Events are tagged with their fractal depth level (timeframe) and a parent chain linking higher-timeframe context.

### 4.4 Fractal Clustering Engine (`training/fractal_clustering.py`)

Groups discovered pattern events into reusable templates using **K-Means clustering** (GPU-accelerated via CUDA when available, scikit-learn CPU fallback):

- **Initial clustering:** raw_patterns / 100 initial clusters
- **Feature extraction:** z-score, velocity, momentum, coherence, wave function amplitudes
- **Template parameters:** Each cluster centroid becomes a `PatternTemplate` with associated entry/exit parameters

### 4.5 Timeframe Belief Network (`training/timeframe_belief_network.py`)

A consensus engine where **8 independent timeframe workers** (1h down to 15s) each vote on trade direction:

- Each worker computes conviction based on its timeframe's quantum state
- Trades are only taken when the **"Golden Path"** of conviction is aligned (minimum conviction threshold)
- The belief network produces a `BeliefState` encoding the aggregate view

### 4.6 Wave Rider (`training/wave_rider.py`)

The trade execution engine handling:
- **Entry logic:** Based on template match + belief network consensus
- **Dynamic trailing stops:** Adjusted based on wave function maturity
- **Coherent Structure Tether (CST):** Monitors whether the entry thesis is still valid
- **Regret analysis:** Post-trade evaluation that measures exit efficiency (too early vs too late)
- **Exit reasons:** Trail stop, structure break, time exit, belief flip

### 4.7 Design of Experiments (DOE) (`training/doe_parameter_generator.py`)

Advanced parameter optimization using:
- **Latin Hypercube Sampling (LHS):** Space-filling experimental designs
- **Response Surface Optimization:** Models the parameter-to-PnL response surface
- **ANOVA Analysis** (`training/anova_analyzer.py`): Identifies statistically significant parameters
- **Thompson Sampling** (`training/thompson_refiner.py`): Bayesian bandit for adaptive refinement
- **PnL-prioritized regret analysis** (`training/batch_regret_analyzer.py`): End-of-day evaluation

### 4.8 Monte Carlo Engine (`training/monte_carlo_engine.py`)

Runs Monte Carlo simulations per template/timeframe combination to estimate expected performance distributions and confidence intervals.

### 4.9 PID Oscillation Analyzer (`training/pid_oscillation_analyzer.py`)

Detects PID-controller-like oscillatory regimes in price action. When the market is oscillating around fair value with high coherence, the analyzer identifies these conditions for specialized mean-reversion strategies.

---

## 5. Live Trading Module

### 5.1 Live Engine (`live/live_engine.py`)

Mirrors the Phase 4 forward pass but operates on real-time bars from NinjaTrader 8. Key simplifications vs training:
- No oracle markers (no lookahead)
- No score/loser tracking
- Direct state-to-centroid matching from the latest bar
- Real equity from NT8 POSITION messages

### 5.2 NT8 Client (`live/nt8_client.py`)

An asyncio TCP client that communicates with the BayesianBridge indicator running inside NinjaTrader 8. Handles:
- Connection / reconnection with exponential backoff
- Heartbeat monitoring
- Inbound message dispatching via asyncio.Queue
- Outbound message encoding

### 5.3 Bar Aggregator (`live/bar_aggregator.py`)

Aggregates tick-level data from NT8 into uniform OHLCV bars at the required resolution.

### 5.4 Order Manager (`live/order_manager.py`)

Tracks order lifecycle and position state. NT8 is the source of truth for position; the OrderManager maintains a local shadow updated on FILL/POSITION messages. Logs trades to CSV.

---

## 6. Configuration

### 6.1 System Settings (`config/settings.py`)

| Setting                        | Value       | Description                                |
|--------------------------------|-------------|--------------------------------------------|
| `OPERATIONAL_MODE`             | `LEARNING`  | Current mode (LEARNING or EXECUTE)         |
| `RAW_DATA_PATH`                | `DATA/RAW`  | Default directory for raw data files       |
| `ANCHOR_DATE`                  | `2025-07-30`| Start date for data file selection         |
| `DEFAULT_BASE_SLIPPAGE`        | `0.25`      | Base slippage per trade (points)           |
| `DEFAULT_VELOCITY_SLIPPAGE_FACTOR` | `0.1`  | Additional slippage in high-velocity bars  |

### 6.2 Asset Profiles (`config/symbols.py`)

Supports four CME equity index futures:

| Ticker | Full Name               | Tick Size | Tick Value | Point Value |
|--------|-------------------------|-----------|------------|-------------|
| NQ     | Nasdaq-100 E-mini       | 0.25      | $5.00      | $20.00      |
| ES     | S&P 500 E-mini          | 0.25      | $12.50     | $50.00      |
| MNQ    | Micro Nasdaq-100        | 0.25      | $0.50      | $2.00       |
| MES    | Micro S&P 500           | 0.25      | $1.25      | $5.00       |

---

## 7. Technology Stack

### 7.1 Core Dependencies

| Library        | Version      | Purpose                                              |
|----------------|--------------|------------------------------------------------------|
| Python         | 3.10+        | Runtime                                              |
| PyTorch        | >=2.1, <2.6  | GPU tensor operations, CUDA acceleration (cu121)     |
| NumPy          | <2           | Vectorized numerical computation                     |
| Numba          | 0.61.2       | JIT-compiled CUDA kernels for physics computations   |
| Pandas         | >=2.2.0      | DataFrame operations and time-series handling        |
| SciPy          | 1.13.1       | Statistical functions (erfi, etc.)                   |
| scikit-learn   | latest       | CPU fallback for K-Means clustering                  |
| Databento      | 0.70.0       | Market data API and DBN format loading               |
| PyArrow        | 23.0.0       | Parquet file I/O                                     |
| pandas_ta      | fork         | Technical analysis indicators (ADX, RSI, etc.)       |
| hurst          | latest       | Hurst exponent calculation                           |
| QuantLib       | latest       | Quantitative finance library                         |
| Plotly         | latest       | Interactive charting                                 |
| Matplotlib     | >=3.8.0      | Static charting and visualization                    |
| tqdm           | latest       | Progress bars                                        |
| PyInstaller    | 6.18.0       | Executable packaging                                 |
| pytest         | latest       | Test framework                                       |

### 7.2 Hardware Requirements

- **GPU (recommended):** NVIDIA GPU with CUDA 12.1 support for PyTorch and Numba kernels
- **CPU fallback:** All GPU paths have automatic CPU fallback via try/except guards
- **Memory:** Sufficient RAM for full-day bar arrays (~5,300 bars per session)

---

## 8. Directory Structure

```
Bayesian-AI/
|-- config/              # System settings, asset profiles, workflow manifest
|   |-- settings.py      # Operational mode, data paths, slippage constants
|   |-- symbols.py       # Futures contract specifications (NQ, ES, MNQ, MES)
|   |-- oracle_config.py # Oracle configuration parameters
|
|-- core/                # Core engine components (the "brain")
|   |-- quantum_field_engine.py   # 3-body physics + wave function computation
|   |-- bayesian_brain.py         # HashMap learning: State -> WinRate
|   |-- three_body_state.py       # 16D+ frozen dataclass for market state
|   |-- dynamic_binner.py         # Freedman-Diaconis histogram binning
|   |-- risk_engine.py            # Monte Carlo tunnel/escape probability
|   |-- state_vector.py           # Legacy 9-layer state representation
|   |-- adaptive_confidence.py    # Progressive threshold tightening
|   |-- exploration_mode.py       # Phase 0 unconstrained pattern discovery
|   |-- cuda_physics.py           # Numba CUDA kernels for physics
|   |-- cuda_pattern_detector.py  # GPU-accelerated pattern detection
|   |-- physics_utils.py          # CPU fallback for ADX/DMI, Hurst
|   |-- context_detector.py       # Multi-timeframe context detection
|   |-- multi_timeframe_context.py# Cross-timeframe correlation engine
|   |-- data_aggregator.py        # Bar aggregation utilities
|   |-- pattern_utils.py          # Geometric/candlestick pattern detection
|   |-- logger.py                 # Centralized logging configuration
|
|-- training/            # Training pipeline and optimization
|   |-- orchestrator.py            # Main pipeline controller
|   |-- orchestrator_worker.py     # Multiprocess worker for template optimization
|   |-- fractal_discovery_agent.py # Physics event scanner
|   |-- fractal_clustering.py      # K-Means pattern clustering (GPU/CPU)
|   |-- fractal_atlas_builder.py   # Atlas construction for OOS testing
|   |-- timeframe_belief_network.py# 8-worker consensus voting engine
|   |-- wave_rider.py              # Trade execution + regret analysis
|   |-- doe_parameter_generator.py # Latin Hypercube Sampling + DOE
|   |-- monte_carlo_engine.py      # MC simulation per template
|   |-- anova_analyzer.py          # Statistical significance analysis
|   |-- thompson_refiner.py        # Bayesian bandit parameter tuning
|   |-- pid_oscillation_analyzer.py# PID oscillation regime detection
|   |-- batch_regret_analyzer.py   # End-of-day regret evaluation
|   |-- databento_loader.py        # Databento DBN data loader
|   |-- dbn_to_parquet.py          # DBN -> Parquet conversion
|   |-- data_loading_optimizer.py  # Data loading performance optimization
|   |-- pattern_analyzer.py        # Strongest state analysis
|   |-- progress_reporter.py       # Terminal progress output
|   |-- pipeline_checkpoint.py     # Checkpoint save/resume
|   |-- integrated_statistical_system.py # Bayesian validation + MC risk
|   |-- trade_analytics.py         # Post-training performance analysis
|   |-- run_analytics.py           # Analytics runner script
|   |-- cuda_kmeans.py             # GPU-accelerated K-Means implementation
|
|-- live/                # Real-time trading via NinjaTrader 8
|   |-- live_engine.py   # Main live trading loop
|   |-- nt8_client.py    # Asyncio TCP client for NT8 bridge
|   |-- bar_aggregator.py# Live bar aggregation
|   |-- order_manager.py # Order lifecycle + position tracking
|   |-- protocol.py      # Wire protocol encoding/decoding
|   |-- config.py        # Live trading configuration
|   |-- launcher.py      # Entry point for live mode
|
|-- visualization/       # Dashboard and charting
|   |-- live_training_dashboard.py # Real-time Tkinter training dashboard
|   |-- visualization_module.py    # Plotly/Matplotlib chart generation
|
|-- tests/               # Comprehensive pytest test suite (26+ test files)
|-- scripts/             # Utility scripts (build, CUDA health, status report)
|-- docs/                # Documentation (architecture, changelogs, specs)
|-- archive/             # Deprecated legacy 9-layer engine
|-- DATA/                # Raw and processed market data
|-- models/              # Serialized models (probability tables)
|-- notebooks/           # Jupyter dashboard notebook
|-- reports/             # Generated reports
|-- run_logs/            # Runtime logs
```

---

## 9. Key Algorithms

### 9.1 Three-Body Gravitational Model

The market is modeled as a three-body problem from celestial mechanics:

1. **Center (Fair Value):** Computed via rolling regression; exerts an attractive mean-reversion force proportional to z^2/9
2. **Upper Singularity (+2 sigma):** Resistance level; exerts repulsive force proportional to 1/r^3
3. **Lower Singularity (-2 sigma):** Support level; exerts repulsive force proportional to 1/r^3
4. **Net Force:** Vector sum determines whether price is being pulled toward center or pushed toward a boundary
5. **Event Horizons (+/-3 sigma):** Points of no return where escape probability dominates

### 9.2 Quantum Wave Function

Price exists in a superposition of being near each attractor:

```
psi = a0 * psi_center + a1 * psi_upper + a2 * psi_lower
```

- Measurement (L8/L9 confirmation) causes collapse to a definite state
- Tunnel probability (Monte Carlo OU process) determines mean-reversion likelihood
- Coherence measures how "decided" the market is (1.0 = full superposition, 0.0 = collapsed)

### 9.3 Bayesian Learning Loop

```
For each trade:
  1. Observe market state (ThreeBodyQuantumState)
  2. Match to nearest cluster template (Euclidean distance in feature space)
  3. Execute trade via WaveRider (entry + dynamic trailing stop)
  4. Record outcome: TradeOutcome(state, pnl, result='WIN'|'LOSS')
  5. Update: probability_table[template_id].wins++ or .losses++
  6. Win probability = wins / total
  7. Future trades filter by minimum probability threshold
```

### 9.4 Fractal Multi-Timeframe Consensus

Eight timeframe workers independently analyze the market at different resolutions:

```
1D -> 4h -> 1h -> 30m -> 15m -> 5m -> 1m -> 15s
```

Each worker produces a directional conviction score. A trade is only taken when:
- The majority of workers agree on direction
- Aggregate conviction exceeds the minimum threshold (MIN_CONVICTION = 0.48)
- The fractal "Golden Path" is aligned from higher to lower timeframes

---

## 10. Testing

The test suite contains **26+ test files** covering:

- **Unit tests:** Bayesian brain, quantum field engine, state vector, dynamic binner, risk engine
- **Integration tests:** Full quantum pipeline, multi-timeframe DOE, clustering
- **Physics tests:** CPU and CUDA physics calculations, pattern detection
- **Training tests:** Wave rider, batch regret analyzer, exploration mode
- **Data tests:** Databento loading, real data velocity processing
- **Dashboard tests:** UX validation, fractal visualization

Run tests with: `python -m pytest`

---

## 11. Deployment

### 11.1 Training Mode

```bash
python training/orchestrator.py --data DATA/ATLAS --dashboard
```

### 11.2 Live Trading Mode

```bash
python -m live
```

Connects to NinjaTrader 8 via TCP bridge, loads frozen templates from checkpoints, and executes trades in real-time.

### 11.3 Build Executable

```bash
python scripts/build_executable.py
```

Produces a standalone executable in `dist/Bayesian_AI_Engine/` via PyInstaller.

---

## 12. Legacy Architecture (Archived)

The original **9-Layer Hierarchy** engine (`archive/layer_engine.py`) decomposed market state into:
- **Static Layers (L1-L4):** 90-day bias, 30-day regime, weekly swing, daily zone
- **Fluid Layers (L5-L9):** 4-hour trend, 1-hour structure, 15-min pattern, 5-min confirmation, 1-second cascade trigger

This has been fully superseded by the Fractal Three-Body Quantum model and resides in the `archive/` directory for historical reference only.
