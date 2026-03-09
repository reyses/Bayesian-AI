# Bayesian-AI System Description

> **Algorithmic Trading System with Statistical Regression Bands, Multi-TF Belief Network & Bayesian Inference**

---

## 1. Purpose and Overview

Bayesian-AI is a high-frequency algorithmic trading system designed for US equity index futures (NQ, ES, MNQ, MES). It models the market using **OLS regression bands** with z-score standardization, computes a **3-class softmax probability distribution** over price location, then applies **Bayesian probability learning** to discover which market states lead to profitable trades.

The system operates across a **multi-timeframe fractal hierarchy** (from 1-day bars down to 1-second ticks), discovers recurring statistical patterns in historical data, clusters them into reusable templates, optimizes parameters via Design of Experiments (DOE), and executes trades through a consensus-based belief network.

**Primary instrument:** Micro Nasdaq-100 futures (MNQ)
**Compute requirement:** NVIDIA CUDA GPU (CPU fallback removed 2026-03-08)

---

## 2. System Architecture

### 2.1 High-Level Pipeline

```
LOAD -> TRANSFORM -> ANALYZE -> DECIDE -> EXECUTE -> LEARN
```

The full pipeline consists of six phases:

| Phase | Name                  | Description                                                     |
|-------|-----------------------|-----------------------------------------------------------------|
| 1     | Data Preparation      | Loads ATLAS parquet files (14 TFs, 12 months)                   |
| 2     | Pattern Discovery     | Scans history for statistical events (band reversals, structural drives) |
| 3     | Template Optimization | Clusters events into templates; optimizes parameters via DOE    |
| 4     | IS Forward Pass       | Replays history using the template library (ExecutionEngine)    |
| 5     | OOS Validation        | Blind out-of-sample validation on unseen data                   |
| 6     | Strategy Selection    | Analyzes OOS results to rank templates (Tier 1-4)               |

### 2.2 Data Flow

```
ATLAS Parquet files (OHLCV, 14 timeframes)
        |
        v
  [StatisticalFieldEngine] -- Computes MarketState for every bar (CUDA)
        |
        v
  MarketState (50+ field frozen dataclass per bar)
        |
        v
  [FractalDiscoveryAgent] -- Phase 2: Scans for statistical events
        |
        v
  PatternEvent list (band reversals, structural drives, compressions)
        |
        v
  [FractalClusteringEngine] -- Phase 3: Recursive K-Means clustering (GPU)
        |
        v
  Pattern Templates (centroids with per-template OLS/logistic models)
        |
        v
  [ExecutionEngine] -- Phase 4: Gate cascade + direction cascade
        |
        v
  [ExitEngine] -- Unified exit cascade (SL->TP->Envelope->Giveback->...)
        |
        v
  Trade Log (oracle_trade_log.csv)
        |
        v
  [BayesianBrain] -- Learns: TemplateID -> Win Probability
        |
        v
  [TradeAnalytics] -- Phase 5-6: Analytics + strategy selection
```

---

## 3. Core Components

### 3.1 Statistical Field Engine (`core/statistical_field_engine.py`)

Transforms raw OHLCV data into a complete market state using GPU-accelerated computation:

- **Regression bands:** OLS linear regression over rolling window, +/- 2 sigma and 3 sigma bands
- **Z-score:** Normalized distance from regression center in sigma units
- **Force fields:** Mean reversion force (z^2/9), band pressure (1/distance^3), momentum (kinetic energy), net force vector
- **Probability distribution:** 3-class softmax — P(center), P(upper), P(lower) via squared amplitude weights
- **OU first-passage:** Analytical erfi-based reversion/breakout probability (replaced Monte Carlo)
- **Entropy:** Shannon entropy of probability distribution (0=collapsed to one state, 1=uniform)
- **Indicators:** ADX/DMI strength, Hurst exponent (trending vs mean-reverting), PID oscillation detection

**Acceleration:** CUDA via Numba kernels (`core/cuda_statistics.py`). CUDA required.

### 3.2 Market State (`core/market_state.py`)

A frozen dataclass (hashable) encapsulating complete market microstructure at a single bar:
- Regression band levels, z-score, velocity, momentum
- Force fields (reversion, band pressure, momentum, net)
- Probability distribution and entropy
- ADX, DMI, Hurst exponent
- Band zone (INNER/CHAOS/UPPER_EXTREME/LOWER_EXTREME)
- OU dynamics (reversion probability, breakout probability)

### 3.3 Bayesian Brain (`core/bayesian_brain.py`)

HashMap-based learning engine: `table[state_key] = {wins, losses, total}`
- **Input:** TradeOutcome objects after each trade
- **Key:** Template ID (preferred) or raw state hash
- **Output:** P(win) = wins / total
- **Direction-aware:** Separate stats for LONG vs SHORT per template
- **Persistence:** Pickle serialization to checkpoints

### 3.4 Execution Engine (`core/execution_engine.py`)

Single source of truth for entry decisions (IS, OOS, and live):
- **Direction cascade** (4 priorities): TBN momentum alignment → belief conviction → band confluence
- **Gate cascade** (sequential): pattern quality → cluster match → brain probability → conviction → direction

### 3.5 Exit Engine (`core/exit_engine.py`)

Unified exit cascade for all phases:
1. **Stop Loss** — hard boundary
2. **Take Profit** — fixed target from template
3. **Band Urgent** — tighten at 3-sigma extremes
4. **Envelope Decay** — main exit (self-tuning halflife 8-60 bars)
5. **Peak Giveback** — trim when unraveling from high water mark
6. **Breakeven Lock** — protect profit after threshold reached
7. **Belief Flip** — exit on TBN direction change
8. **Hold** — max duration timer

Self-tuning: `record_trade_outcome()` calibrates halflife and giveback every 30 trades.

### 3.6 Timeframe Belief Network (`core/timeframe_belief_network.py`)

11 independent timeframe workers (1s through 1D) each predict direction P(LONG):
- **Path conviction:** Weighted geometric mean of per-worker conviction
- **Band confluence:** Multi-TF SE band direction consensus (Priority 4 in cascade)
- Each worker maintains BandContext: z-score relative to bands at its timeframe

### 3.7 Feature Extraction (`core/feature_extraction.py`)

Canonical 16D feature vector — single source of truth used by both clustering and live:
`[|z|, log1p(|v|), log1p(|m|), entropy, tf_scale, depth, parent_ctx, adx, hurst, dmi_diff, parent_z, parent_dmi, root_band_reversal, tf_alignment, pid, osc_coherence]`

---

## 4. Training Pipeline

### Entry point: `python training/trainer.py`

- `--fresh --forward-pass` — full pipeline (wipe + IS + OOS + strategy)
- `--forward-pass` — IS + OOS using existing templates
- `--oos` — standalone OOS rerun
- `--ping-pong` — continuous wave-riding mode

### Key training components:
- **FractalDiscoveryAgent** (`training/fractal_discovery_agent.py`): Multi-TF fractal scan with oracle markers
- **Orchestrator Worker** (`training/orchestrator_worker.py`): Numba JIT simulation with spectral gates (Fourier half-cycle + Laplace kinetic damping)
- **Trade Analytics** (`training/trade_analytics.py`): t-tests, ANOVA, OLS, logistic regression, capture rate analysis

---

## 5. Live Trading Module

### 5.1 Live Engine (`live/live_engine.py`)
Mirrors Phase 4 forward pass on real-time NT8 bars. No oracle (no lookahead).
Warm-up via `HistoryReplayEngine` (5 days of ATLAS data).

### 5.2 Supporting modules:
- **NT8 Client** (`live/nt8_client.py`): Asyncio TCP bridge to NinjaTrader 8
- **Bar Aggregator** (`live/bar_aggregator.py`): 1s → 15s OHLCV aggregation
- **Order Manager** (`live/order_manager.py`): Order lifecycle, position tracking
- **Session Tracker** (`live/session_tracker.py`): PnL, drawdowns, trade log, reports
- **Ping Pong** (`live/ping_pong.py`): Direction flip + ATR sizing for wave-riding
- **History Replay** (`live/history_replay.py`): Compressed warm-up from ATLAS data
- **Atlas Loader** (`live/atlas_loader.py`): Parquet reader for date ranges

---

## 6. Configuration

### Asset Profiles (`config/symbols.py`)

| Ticker | Full Name          | Tick Size | Tick Value | Point Value |
|--------|--------------------|-----------|------------|-------------|
| NQ     | Nasdaq-100 E-mini  | 0.25      | $5.00      | $20.00      |
| ES     | S&P 500 E-mini     | 0.25      | $12.50     | $50.00      |
| MNQ    | Micro Nasdaq-100   | 0.25      | $0.50      | $2.00       |
| MES    | Micro S&P 500      | 0.25      | $1.25      | $5.00       |

---

## 7. Technology Stack

| Library    | Purpose                                         |
|------------|-------------------------------------------------|
| PyTorch    | GPU tensor operations, CUDA acceleration (cu121) |
| Numba      | JIT-compiled CUDA kernels                       |
| NumPy      | Vectorized numerical computation                |
| SciPy      | Statistical functions (erfi, t-tests, etc.)     |
| Pandas     | DataFrame operations, time-series               |
| scikit-learn | K-Means fallback, logistic regression          |
| Databento  | Market data API and DBN format loading           |
| PyArrow    | Parquet file I/O                                |
| tqdm       | Progress bars                                   |
| Tkinter    | Real-time dashboard                             |

**Hardware:** NVIDIA CUDA GPU required. ~8GB VRAM sufficient.
