# Bayesian-AI: Complete Logic & Architecture User's Guide

> **Version:** 2.0 | **Date:** 2026-02-27 | **Engine:** Fractal Three-Body Quantum (Active)

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [High-Level Architecture](#2-high-level-architecture)
3. [Directory Structure & Module Map](#3-directory-structure--module-map)
4. [Core Engine: The Physics Model](#4-core-engine-the-physics-model)
   - 4.1 [The Three-Body Problem Analogy](#41-the-three-body-problem-analogy)
   - 4.2 [QuantumFieldEngine — The Math Core](#42-quantumfieldengine--the-math-core)
   - 4.3 [ThreeBodyQuantumState — The State Object](#43-threebodyquantumstate--the-state-object)
   - 4.4 [Lagrange Zones & Trade Classification](#44-lagrange-zones--trade-classification)
   - 4.5 [Quantum Wave Function & Tunneling](#45-quantum-wave-function--tunneling)
   - 4.6 [PID Control Force](#46-pid-control-force)
   - 4.7 [Technical Indicators (ADX, Hurst, Patterns)](#47-technical-indicators-adx-hurst-patterns)
5. [Bayesian Learning System](#5-bayesian-learning-system)
   - 5.1 [BayesianBrain — Probability Table](#51-bayesianbrain--probability-table)
   - 5.2 [Dynamic Binning (Freedman-Diaconis)](#52-dynamic-binning-freedman-diaconis)
   - 5.3 [Adaptive Confidence Phases](#53-adaptive-confidence-phases)
   - 5.4 [Decision Logic: Should Fire?](#54-decision-logic-should-fire)
6. [Training Pipeline](#6-training-pipeline)
   - 6.1 [BayesianTrainingOrchestrator — Master Controller](#61-bayesiantrainingorchestrator--master-controller)
   - 6.2 [Data Loading (Databento)](#62-data-loading-databento)
   - 6.3 [Fractal Discovery Agent (Top-Down Scanner)](#63-fractal-discovery-agent-top-down-scanner)
   - 6.4 [Fractal Clustering Engine (Hypervolume Tree)](#64-fractal-clustering-engine-hypervolume-tree)
   - 6.5 [Oracle Labeling System](#65-oracle-labeling-system)
   - 6.6 [DOE Parameter Optimization](#66-doe-parameter-optimization)
   - 6.7 [Monte Carlo Engine](#67-monte-carlo-engine)
   - 6.8 [Timeframe Belief Network](#68-timeframe-belief-network)
   - 6.9 [Walk-Forward Training Loop](#69-walk-forward-training-loop)
   - 6.10 [Wave Rider Exit System & Regret Analysis](#610-wave-rider-exit-system--regret-analysis)
7. [Multi-Timeframe Architecture](#7-multi-timeframe-architecture)
8. [CUDA / GPU Acceleration](#8-cuda--gpu-acceleration)
9. [Risk Management](#9-risk-management)
10. [Live Trading System](#10-live-trading-system)
11. [Visualization & Dashboard](#11-visualization--dashboard)
12. [Configuration & Constants](#12-configuration--constants)
13. [Data Flow: End-to-End Walkthrough](#13-data-flow-end-to-end-walkthrough)
14. [Testing & Validation](#14-testing--validation)
15. [Legacy Architecture (Archived)](#15-legacy-architecture-archived)
16. [Glossary](#16-glossary)

---

## 1. System Overview

**Bayesian-AI** is a GPU-accelerated algorithmic trading system designed for Micro Nasdaq (MNQ) futures. It combines:

- **Physics-based market modeling** — prices are modeled as particles in a three-body gravitational field
- **Bayesian probability learning** — a hash-map records win/loss counts per discretized market state
- **Fractal multi-timeframe analysis** — patterns are discovered hierarchically from daily down to 1-second resolution
- **Design of Experiments (DOE)** — systematic parameter optimization via Latin Hypercube Sampling and Optuna TPE
- **Monte Carlo validation** — statistical confidence in strategy edges before deployment

The system operates in two modes:
- **LEARNING** — trains on historical data, builds probability tables, discovers patterns
- **EXECUTE** — live trading via NinjaTrader 8 TCP bridge

**Primary instrument:** MNQ (Micro E-mini Nasdaq-100 Futures)
- Tick size: 0.25 points
- Tick value: $0.50
- Point value: $2.00

---

## 2. High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        BAYESIAN-AI SYSTEM                           │
│                                                                     │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────────────┐  │
│  │  DATA LAYER  │───▶│ PHYSICS CORE │───▶│  LEARNING / DECISION │  │
│  │              │    │              │    │                      │  │
│  │ Databento    │    │ QuantumField │    │ BayesianBrain        │  │
│  │ Parquet      │    │ Engine       │    │ AdaptiveConfidence   │  │
│  │ Atlas Builder│    │ CUDA Kernels │    │ TimeframeBeliefNet   │  │
│  └──────────────┘    └──────────────┘    └──────────────────────┘  │
│         │                    │                      │               │
│         ▼                    ▼                      ▼               │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────────────┐  │
│  │  DISCOVERY   │    │  CLUSTERING  │    │   EXECUTION          │  │
│  │              │    │              │    │                      │  │
│  │ FractalDisc. │    │ Hypervolume  │    │ WaveRider (Exits)    │  │
│  │ Agent        │    │ Tree / IMR   │    │ OrderManager         │  │
│  │ PatternEvent │    │ Templates    │    │ NT8 Bridge           │  │
│  └──────────────┘    └──────────────┘    └──────────────────────┘  │
│         │                    │                      │               │
│         ▼                    ▼                      ▼               │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────────────┐  │
│  │ OPTIMIZATION │    │  VALIDATION  │    │   VISUALIZATION      │  │
│  │              │    │              │    │                      │  │
│  │ DOE / Optuna │    │ Monte Carlo  │    │ Fractal Command      │  │
│  │ Thompson     │    │ ANOVA        │    │ Center (Tkinter)     │  │
│  │ Refiner      │    │ Regret Anal. │    │ Jupyter Dashboard    │  │
│  └──────────────┘    └──────────────┘    └──────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
```

**Pipeline:** `LOAD → TRANSFORM → ANALYZE → LEARN → OPTIMIZE → VALIDATE → EXECUTE`

---

## 3. Directory Structure & Module Map

```
Bayesian-AI/
├── core/                      # The physics & decision engine
│   ├── quantum_field_engine.py   # Master physics calculator (GPU+CPU)
│   ├── bayesian_brain.py         # Probability table (HashMap learning)
│   ├── three_body_state.py       # ThreeBodyQuantumState frozen dataclass
│   ├── state_vector.py           # Legacy 9-layer state (deprecated)
│   ├── risk_engine.py            # Monte Carlo OU-process risk estimator
│   ├── cuda_physics.py           # Numba CUDA kernels (GPU acceleration)
│   ├── cuda_pattern_detector.py  # CUDA geometric/candlestick detection
│   ├── physics_utils.py          # CPU-based ADX/DMI, Hurst, FFT
│   ├── pattern_utils.py          # Geometric + candlestick pattern detectors
│   ├── dynamic_binner.py         # Freedman-Diaconis histogram binner
│   ├── adaptive_confidence.py    # 4-phase learning progression
│   ├── context_detector.py       # 200-parameter context activation
│   ├── exploration_mode.py       # Phase 0: unconstrained pattern discovery
│   ├── multi_timeframe_context.py # 8-layer timeframe context engine
│   ├── data_aggregator.py        # Tick-to-bar ring buffer
│   └── logger.py                 # Centralized logging
│
├── training/                  # Training orchestration & analysis
│   ├── orchestrator.py           # MASTER: Walk-forward training loop
│   ├── orchestrator_worker.py    # Numba-optimized trade simulation
│   ├── doe_parameter_generator.py # Optuna TPE PID optimization
│   ├── fractal_discovery_agent.py # Top-down hierarchical pattern scanner
│   ├── fractal_clustering.py     # Hypervolume tree (IMR splitting)
│   ├── fractal_dna_tree.py       # DNA tree for fractal encoding
│   ├── fractal_atlas_builder.py  # Multi-TF parquet atlas construction
│   ├── timeframe_belief_network.py # N-worker TF consensus engine
│   ├── wave_rider.py             # Exit management + regret analysis
│   ├── monte_carlo_engine.py     # Brute-force template×TF×params sweep
│   ├── anova_analyzer.py         # ANOVA statistical significance
│   ├── thompson_refiner.py       # Thompson sampling parameter refinement
│   ├── batch_regret_analyzer.py  # End-of-day opportunity cost analysis
│   ├── pid_oscillation_analyzer.py # PID regime detection
│   ├── pattern_analyzer.py       # Pattern strength analysis
│   ├── integrated_statistical_system.py # Bayesian state validation
│   ├── databento_loader.py       # .dbn.zst → DataFrame loader
│   ├── dbn_to_parquet.py         # Raw trades → OHLCV parquet
│   ├── data_loading_optimizer.py # Parallel data loading
│   ├── pipeline_checkpoint.py    # Training resume checkpoints
│   ├── progress_reporter.py      # Terminal progress display
│   ├── trade_analytics.py        # Trade statistics computation
│   ├── run_analytics.py          # Post-run analysis
│   └── cuda_kmeans.py            # GPU-accelerated K-Means clustering
│
├── config/                    # Configuration & asset profiles
│   ├── settings.py               # Global mode (LEARNING/EXECUTE), paths
│   ├── symbols.py                # Asset profiles (MNQ, NQ, ES, MES)
│   ├── oracle_config.py          # Oracle lookahead & marker thresholds
│   └── workflow_manifest.json    # CI/CD manifest
│
├── live/                      # Live trading bridge
│   ├── live_engine.py            # Main live trading loop
│   ├── nt8_client.py             # Async TCP client for NinjaTrader 8
│   ├── bar_aggregator.py         # Real-time bar construction
│   ├── order_manager.py          # Order placement & management
│   ├── protocol.py               # Wire protocol encoder/decoder
│   ├── config.py                 # Live trading configuration
│   └── launcher.py               # Live system entry point
│
├── visualization/             # Real-time dashboards
│   ├── live_training_dashboard.py # Tkinter "Fractal Command Center"
│   └── visualization_module.py   # Chart generation utilities
│
├── tests/                     # Comprehensive pytest suite (36+ tests)
├── scripts/                   # Utility scripts (build, CUDA health, etc.)
├── tools/                     # Analysis tools (benchmarks, visualizers)
├── notebooks/                 # Jupyter dashboard
├── archive/                   # Deprecated 9-layer engine
├── docs/                      # Documentation
├── DATA/                      # Training data (parquet atlas)
├── models/                    # Saved probability tables (.pkl)
└── run_logs/                  # Signal & trade logs (.csv)
```

---

## 4. Core Engine: The Physics Model

### 4.1 The Three-Body Problem Analogy

The system models the market as a **three-body gravitational system** where price is a particle influenced by three attractors:

```
Event Horizon Upper (+3σ)     ← Point of No Return
    ┌─────────────────────┐
    │ Upper Singularity   │   ← +2σ (Roche Limit) — REPULSIVE
    │      (+2σ)          │
    │                     │
    │    Center Star      │   ← Fair Value (Regression Line) — ATTRACTIVE
    │       (0σ)          │
    │                     │
    │ Lower Singularity   │   ← -2σ (Roche Limit) — REPULSIVE
    │      (-2σ)          │
    └─────────────────────┘
Event Horizon Lower (-3σ)     ← Point of No Return
```

**Key physics mappings:**
| Physics Concept | Market Meaning |
|---|---|
| Center Star (Body 1) | Fair value from linear regression — ATTRACTIVE |
| Upper Singularity (Body 2) | +2σ resistance — REPULSIVE |
| Lower Singularity (Body 3) | -2σ support — REPULSIVE |
| Particle Position | Current price |
| Particle Velocity | Price change (bar to bar) |
| z-score | Normalized distance from center (sigma units) |
| Gravity (F_reversion) | -θ × z × σ — pulls price toward center |
| Repulsion (F_upper/lower) | 1/r³ — pushes price away from ±2σ boundaries |
| Momentum (F_momentum) | velocity × volume / σ — kinetic energy |
| Net Force (F_net) | gravity + momentum + repulsion |

### 4.2 QuantumFieldEngine — The Math Core

**File:** `core/quantum_field_engine.py` — Class `QuantumFieldEngine`

This is the heart of the system. It computes all physics quantities for an entire day of bars in a single vectorized pass.

**Primary method: `batch_compute_states(day_data, use_cuda=True, params=None)`**

Input: DataFrame with columns `[open, high, low, close, volume, timestamp]`
Output: List of dicts, each containing a `ThreeBodyQuantumState` and metadata

**Computation pipeline (per day, ~5300 bars at 15s resolution):**

```
Step 1: Rolling Linear Regression (21-bar window)
  ├── Center = mean_y + slope × ((rp-1) - mean_x)
  ├── Sigma  = sqrt(RSS / (rp-2)), clamped ≥ 1e-6
  └── Slope  = (sum_xy - mean_x × sum_y) / denom

Step 2: Z-Score & Kinematics
  ├── z_score  = (price - center) / sigma
  ├── velocity = price[i] - price[i-1]
  └── momentum = (velocity × volume) / sigma

Step 3: Force Fields
  ├── F_gravity   = -0.5 × z × σ
  ├── F_repulsion = ±1/r³ (from nearest Roche limit)
  └── F_net       = gravity + momentum + repulsion

Step 4: Quantum Wave Function
  ├── E0 = -z²/2,  E1 = -(z-2)²/2,  E2 = -(z+2)²/2
  ├── Softmax normalization → P_center, P_upper, P_lower
  └── Shannon Entropy H = -Σ(p × log(p))

Step 5: PID Control Force
  ├── P = kp × z (proportional)
  ├── I = ki × clamp(cumsum(z), -10, 10) (integral)
  ├── D = kd × diff(z) (derivative)
  └── term_pid = P + I + D

Step 6: Oscillation Coherence
  └── 1 / (1 + rolling_std(z, window=5))

Step 7: OU Tunneling Probabilities
  ├── P(tunnel) = 1 - erfi(|z|/√2) / erfi(3/√2)
  ├── P(escape) = erfi(|z|/√2) / erfi(3/√2)
  └── Barrier height = 0.025 × (9 - z²)

Step 8: Technical Indicators
  ├── ADX/DMI (14-period Wilder smoothing)
  ├── Hurst Exponent (R/S method, 4 sub-windows)
  ├── Geometric Patterns (Compression, Wedge, Breakdown)
  └── Candlestick Patterns (Doji, Hammer, Engulfing)

Step 9: Archetype Detection
  ├── ROCHE_SNAP:      |z| > 2.0 AND |velocity| > 0.5
  └── STRUCTURAL_DRIVE: |momentum| > 5.0 AND coherence < 0.3
```

**GPU vs CPU execution:**
- **GPU path** (Numba CUDA): 4 parallel kernels → `compute_physics_kernel`, `detect_archetype_kernel`, `compute_dm_tr_kernel`, `compute_hurst_kernel`
- **CPU path** (NumPy vectorized): Uses `np.convolve` for O(N) regression, vectorized wave function, `sliding_window_view` for oscillation coherence

### 4.3 ThreeBodyQuantumState — The State Object

**File:** `core/three_body_state.py` — Frozen dataclass with **60+ fields**

This is the complete quantum state representation of the market at a single point in time. It serves as the **key** in the Bayesian probability table via custom `__hash__` and `__eq__`.

**Field categories:**

| Category | Fields | Purpose |
|---|---|---|
| Three Attractors | center_position, upper/lower_singularity, event_horizons | Define the gravitational field |
| Particle State | position, velocity, z_score | Current price dynamics |
| Force Fields | F_reversion, F_upper/lower_repulsion, F_momentum, F_net | Net force on price |
| Wave Function | amplitude_center/upper/lower, P_at_center/near_upper/near_lower | Probability distribution |
| Decoherence | entropy, coherence, pattern_maturity, momentum_strength | Wave function collapse state |
| Measurement | structure_confirmed, cascade_detected, spin_inverted | Trigger conditions |
| Classification | lagrange_zone, stability_index | Market regime label |
| Tunneling | tunnel_probability, escape_probability, barrier_height | Mean reversion likelihood |
| Indicators | hurst_exponent, adx_strength, dmi_plus/minus | Technical analysis |
| Patterns | pattern_type, candlestick_pattern | Shape recognition |
| Nightmare Field | sigma_fractal, term_pid, oscillation_coherence | Advanced dynamics |
| Fractal | fractal_alignment_count, fractal_confidence, fractal_edge | Multi-scale alignment |
| Multi-TF Context | daily_trend, h4_trend, h1_trend, session | Higher timeframe context |

**Hashing strategy (for Bayesian table lookups):**

```python
core = (
    z_bin,                    # Binned z-score (DynamicBinner)
    momentum_bin,             # Binned momentum (DynamicBinner)
    lagrange_zone,            # L1_STABLE | L2_ROCHE | L3_ROCHE
    structure_confirmed,      # bool
    cascade_detected,         # bool
    trend_direction_15m,      # UP | DOWN | RANGE
    pattern_type,             # NONE | COMPRESSION | WEDGE | BREAKDOWN
    market_regime             # STABLE | CHAOTIC
)
hash(core + context_tuple)   # context_tuple grows with available TF data
```

### 4.4 Lagrange Zones & Trade Classification

The z-score determines the particle's "zone" in the three-body field:

```
z < -2.0  →  L3_ROCHE  (Lower Roche limit — mean reversion SHORT opportunity)
-2 ≤ z < -1 →  CHAOS   (Chaotic zone — no clear direction)
-1 ≤ z ≤ 1  →  L1_STABLE (Near center — equilibrium, no trade)
 1 < z ≤ 2  →  CHAOS   (Chaotic zone — no clear direction)
z > 2.0   →  L2_ROCHE  (Upper Roche limit — mean reversion LONG SHORT opportunity)
```

**Trading logic:** Only trade at Roche limits (L2_ROCHE or L3_ROCHE), where the particle is at the gravitational boundary between center attraction and singularity repulsion.

### 4.5 Quantum Wave Function & Tunneling

The **wave function** models the probability of price being at one of three attractors:

```
ψ = a₀|center⟩ + a₁|upper⟩ + a₂|lower⟩

Energies:    E₀ = -z²/2       (center well)
             E₁ = -(z-2)²/2   (upper well)
             E₂ = -(z+2)²/2   (lower well)

Probabilities: Softmax(E₀, E₁, E₂) → P_center, P_upper, P_lower
```

**Tunneling probability** (OU first-passage): Probability that price reverts to center before hitting the event horizon (±3σ):

```
P(tunnel) = 1 - erfi(|z|/√2) / erfi(3/√2)
```

When `P(tunnel) ≥ 0.80`, the system considers a mean-reversion trade.

**Coherence** = Shannon entropy / ln(3). Low coherence (< 0.3) means the wave function has "collapsed" to a definite state — strong directional signal.

### 4.6 PID Control Force

Models the algorithmic market-maker force acting on price:

```
term_pid = Kp × z + Ki × clamp(cumsum(z), -10, 10) + Kd × diff(z)
```

| Term | Default | Meaning |
|---|---|---|
| Kp (Proportional) | 0.5 | Reaction strength to current displacement |
| Ki (Integral) | 0.1 | Accumulated bias (mean-reversion pressure) |
| Kd (Derivative) | 0.2 | Rate of change dampening |

**Oscillation coherence** measures how tight the z-score oscillation is: `1 / (1 + rolling_std(z, 5))`. Values near 1.0 indicate a PID-driven regime (tight periodic oscillation).

### 4.7 Technical Indicators (ADX, Hurst, Patterns)

**ADX/DMI** (14-period Wilder smoothing):
- ADX > 25 → Strong trend (structural drive, not mean-reversion)
- DMI+ vs DMI- → Directional bias

**Hurst Exponent** (Rescaled Range method):
- H < 0.5 → Anti-persistent (mean-reverting / chop) — HALT
- H = 0.5 → Random walk (Brownian)
- H > 0.5 → Persistent (trending) — proceed

**Geometric Patterns** (vectorized detection):
- COMPRESSION: Recent 5-bar range < 70% of prior 5-bar range
- WEDGE: Higher lows AND lower highs over 5 bars
- BREAKDOWN: Current low < min of previous 4 lows

**Candlestick Patterns:**
- DOJI: body/range < 10%
- HAMMER: lower shadow > 2× body, upper shadow < 10% range
- ENGULFING_BULL / ENGULFING_BEAR: Standard engulfing criteria

---

## 5. Bayesian Learning System

### 5.1 BayesianBrain — Probability Table

**File:** `core/bayesian_brain.py` — Classes `BayesianBrain`, `QuantumBayesianBrain`

The brain is a simple but powerful HashMap:

```
table[ThreeBodyQuantumState] = {'wins': X, 'losses': Y, 'total': Z}
```

**Probability estimation** uses a **pessimistic Beta prior** Beta(1, 10):

```python
P(win) = (wins + 1) / (total + 11)
```

This means:
- New state (0 trades): P = 1/11 ≈ 9% — very pessimistic
- 10 wins / 0 losses: P = 11/21 ≈ 52% — still cautious
- 80 wins / 20 losses: P = 81/111 ≈ 73% — growing confidence

**Confidence** scales linearly with sample size:

```python
confidence = min(total / 100.0, 1.0)  # 100 trades = full confidence
```

### 5.2 Dynamic Binning (Freedman-Diaconis)

**File:** `core/dynamic_binner.py` — Class `DynamicBinner`

Continuous values (z_score, momentum) are binned into histogram centers before hashing. Bin edges are computed from observed data using the **Freedman-Diaconis rule**:

```
bin_width = 2 × IQR × n^(-1/3)
```

This prevents:
- **Too few bins** (loss of discrimination between states)
- **Too many bins** (sparse data, poor learning)
- **Fixed bins** (insensitive to data distribution)

Typical result: 5-30 bins per variable, automatically adapted to data.

### 5.3 Adaptive Confidence Phases

**File:** `core/adaptive_confidence.py` — Class `AdaptiveConfidenceManager`

Learning progresses through 4 phases with progressively tighter thresholds:

```
Phase 1: EXPLORATION     (trades 0-200)
  P ≥ 0%,  Conf ≥ 0%    — Fire at everything at Roche
  Goal: Build initial probability map

Phase 2: REFINEMENT      (trades 200-600)
  P ≥ 45%, Conf ≥ 20%   — Filter obvious losers
  Goal: Begin discriminating states

Phase 3: OPTIMIZATION    (trades 600-1000)
  P ≥ 55%, Conf ≥ 30%   — Focus on promising setups
  Goal: Narrow to high-probability patterns

Phase 4: MASTERY         (trades 1000+)
  P ≥ 80%, Conf ≥ 40%   — Exploit proven edge only
  Goal: Maximum selectivity
```

**Phase advancement criteria:**
- Minimum trades in phase completed
- Recent 50-trade win rate > 55%
- At least 10 high-confidence states learned
- Recent Sharpe ratio > 0.5

### 5.4 Decision Logic: Should Fire?

The complete decision chain (from most permissive to most selective):

```
1. UnconstrainedExplorer.should_fire()    [Phase 0: fire everything]
       ↓
2. AdaptiveConfidenceManager.should_fire() [Phase 1-4: progressive gates]
       ↓
3. BayesianBrain.should_fire()            [Basic: P≥80%, Conf≥30%]
       ↓
4. QuantumBayesianBrain.should_fire_quantum() [Full quantum gates]
       │
       ├── Lagrange zone must be L2_ROCHE or L3_ROCHE
       ├── structure_confirmed AND cascade_detected
       ├── F_momentum ≤ F_reversion × 1.5
       ├── P(win) ≥ 80%
       └── Confidence ≥ 30%
       ↓
5. ThreeBodyQuantumState.get_trade_directive() [Final filters]
       │
       ├── Hurst ≥ 0.5 (no chop)
       ├── Daily regime filter (no shorting bull expansion)
       └── Tunnel probability ≥ 80%
```

---

## 6. Training Pipeline

### 6.1 BayesianTrainingOrchestrator — Master Controller

**File:** `training/orchestrator.py` — Class `BayesianTrainingOrchestrator`

The orchestrator is the single entry point for all training. It coordinates:

```
python training/orchestrator.py --data-dir DATA/RAW --iterations 50 --output checkpoints/
```

**Orchestration phases:**

```
Phase 1: DATA PREPARATION
  ├── Load Databento .dbn.zst files
  ├── Build multi-timeframe atlas (1s→1D parquet)
  └── Fit DynamicBinner on observed z/momentum distributions

Phase 2: FRACTAL DISCOVERY
  ├── FractalDiscoveryAgent scans atlas top-down
  ├── Detect ROCHE_SNAP and STRUCTURAL_DRIVE patterns
  ├── Oracle labels each pattern with lookahead MFE/MAE
  └── Build parent_chain ancestry for each pattern

Phase 3: CLUSTERING
  ├── Extract 16D feature vectors from patterns
  ├── Build HypervolumeTree via IMR geometric splitting
  ├── Each leaf = PatternTemplate with aggregated statistics
  └── Compute transition probabilities between templates

Phase 4: WALK-FORWARD TRAINING (Day-by-Day)
  ├── For each day in dataset:
  │   ├── Compute batch quantum states (GPU)
  │   ├── Match patterns to nearest template (centroid distance)
  │   ├── TimeframeBeliefNetwork updates worker beliefs
  │   ├── Gate 0: Physics gate (ADX, Hurst, regime)
  │   ├── Gate 1: Cluster distance check (< 4.5)
  │   ├── Simulate trade with Numba fast_sim_loop
  │   ├── Record outcome in BayesianBrain
  │   └── Batch regret analysis (end of day)
  └── Checkpoint save every N days

Phase 5: DOE OPTIMIZATION
  ├── Optuna TPE optimizes PID (kp, ki, kd) per cluster
  ├── Thompson Refiner for parameter exploration
  └── ANOVA validates statistical significance

Phase 6: VALIDATION & REPORTING
  ├── Monte Carlo bootstrap confidence intervals
  ├── Walk-forward out-of-sample testing
  ├── Golden Path analysis (opportunity cost)
  └── Generate reports (PnL, Sharpe, drawdown)
```

### 6.2 Data Loading (Databento)

**File:** `training/databento_loader.py` — Class `DatabentoLoader`

Loads `.dbn.zst` compressed files from Databento (CME market data):

```python
loader = DatabentoLoader()
df = loader.load_file("DATA/RAW/glbx-mdp3-20250730.trades.0000.dbn.zst")
# Returns DataFrame with: timestamp, open, high, low, close, volume
```

**File:** `training/fractal_atlas_builder.py` — Builds the multi-timeframe atlas:

```
DATA/ATLAS_1MONTH/
├── 1s/2025_01.parquet     # 1-second bars
├── 5s/2025_01.parquet     # 5-second bars
├── 15s/2025_01.parquet    # 15-second bars (primary resolution)
├── 1m/2025_01.parquet     # 1-minute bars
├── 5m/2025_01.parquet     # 5-minute bars (decision level)
├── 15m/2025_01.parquet    # 15-minute bars
├── 1h/2025_01.parquet     # 1-hour bars
├── 4h/2025_01.parquet     # 4-hour bars
├── 1D/2025_01.parquet     # Daily bars
└── 1W/2025_01.parquet     # Weekly bars
```

### 6.3 Fractal Discovery Agent (Top-Down Scanner)

**File:** `training/fractal_discovery_agent.py` — Class `FractalDiscoveryAgent`

Scans the atlas using a **fractal top-down approach**:

```
Step 1: Start at largest TF (1D) → scan for ROCHE_SNAP / STRUCTURAL_DRIVE
Step 2: For each macro pattern, define time window
Step 3: Drill down to next TF (4h) within that window
Step 4: Repeat recursively: 1h → 30m → 15m → 5m → 3m → 2m → 1m → 30s → 15s → 5s → 1s
```

**Context-only timeframes** (1D, 4h, 1h, 30m): Enrich `parent_chain` context but don't require patterns to exist. They never block drill-down.

**Primary signal level:** 15m — macro scan starts here for stable intraday signals.

Each discovered pattern becomes a `PatternEvent`:

```python
@dataclass
class PatternEvent:
    pattern_type: str        # 'ROCHE_SNAP' or 'STRUCTURAL_DRIVE'
    timestamp: float
    price: float
    z_score: float
    velocity: float
    momentum: float
    coherence: float
    state: ThreeBodyQuantumState
    timeframe: str           # e.g., '15s'
    depth: int               # 0=macro, higher=finer
    parent_type: str         # Parent's pattern type
    parent_chain: list       # Full ancestry chain [{z, dmi+, dmi-, type}, ...]
    oracle_marker: int       # +2=mega_long, +1=scalp_long, 0=noise, -1=scalp_short, -2=mega_short
    oracle_meta: dict        # {mfe, mae, lookahead_bars}
```

### 6.4 Fractal Clustering Engine (Hypervolume Tree)

**File:** `training/fractal_clustering.py` — Classes `FractalClusteringEngine`, `HypervolumeTree`, `PatternTemplate`

Discovered patterns are grouped into **templates** via geometric clustering:

**16D Feature Vector** (per pattern):

```
[|z|, log1p(|v|), log1p(|m|), coherence,           # 4: physics
 tf_scale, depth, parent_ctx,                        # 3: ancestry
 adx/100, hurst, dmi_diff/100,                       # 3: regime
 parent_z, parent_dmi_diff, root_is_roche,           # 3: parent
 tf_alignment, term_pid, oscillation_coherence]      # 3: dynamics
```

**IMR (Individual-Moving Range) Splitting:**

Instead of KMeans, the system uses SPC-style geometric splitting:

1. Order patterns along the principal axis (PCA or regression direction)
2. Compute IMR chart (Individual values + Moving Range)
3. Split where composite signal exceeds control threshold: `x + y > z`
4. Recursively split until R² > 0.90 or group size < 30

Each leaf node becomes a `PatternTemplate` with:
- Centroid (16D mean)
- Member count
- Win rate, expectancy, MFE/MAE statistics
- Direction bias (LONG/SHORT)
- Regression models (MFE prediction, direction prediction)
- Transition probabilities to other templates

### 6.5 Oracle Labeling System

**File:** `config/oracle_config.py` — Configuration for oracle markers

During training (NOT live), each pattern gets **oracle labels** via lookahead:

```
Oracle computes MFE (Max Favorable Excursion) and MAE (Max Adverse Excursion)
over the next N bars (configurable per timeframe).

Classification:
  MFE/MAE ≥ 3.0 AND MFE ≥ 5 ticks  →  MEGA    (marker ±2)
  MFE/MAE ≥ 1.2 AND MFE ≥ 5 ticks  →  SCALP   (marker ±1)
  Otherwise                           →  NOISE   (marker 0)

Direction from sign of MFE: positive = LONG, negative = SHORT
```

| Marker | Value | Meaning |
|---|---|---|
| MARKER_MEGA_LONG | +2 | Large move up, excellent risk/reward |
| MARKER_SCALP_LONG | +1 | Small move up, acceptable risk/reward |
| MARKER_NOISE | 0 | No clear direction or insufficient move |
| MARKER_SCALP_SHORT | -1 | Small move down |
| MARKER_MEGA_SHORT | -2 | Large move down |

### 6.6 DOE Parameter Optimization

**File:** `training/doe_parameter_generator.py` — Class `DOEParameterGenerator`

Uses **Optuna TPE** (Tree-structured Parzen Estimator) to optimize PID parameters:

```python
generator = DOEParameterGenerator(context_detector)
best_params = generator.optimize_pid(
    objective_fn=sharpe_function,   # Maximize Sharpe ratio
    n_trials=200,
    seed=42
)
# Returns: {'pid_kp': 0.73, 'pid_ki': 0.08, 'pid_kd': 0.35}
```

**Parameter ranges:**

| Parameter | Min | Max | Consumed By |
|---|---|---|---|
| pid_kp | 0.1 | 1.0 | QuantumFieldEngine.batch_compute_states |
| pid_ki | 0.01 | 0.2 | QuantumFieldEngine.batch_compute_states |
| pid_kd | 0.1 | 0.5 | QuantumFieldEngine.batch_compute_states |

### 6.7 Monte Carlo Engine

**File:** `training/monte_carlo_engine.py` — Class `MonteCarloEngine`

Brute-force simulation: sweeps all combinations of `template × timeframe × parameters`:

```
For each template:
  For each timeframe:
    For each parameter set:
      Simulate all matching patterns
      Record PnL, win rate, Sharpe, max drawdown
      Keep top 10 iterations by PnL
```

**File:** `core/risk_engine.py` — Class `QuantumRiskEngine`

Uses **Ornstein-Uhlenbeck process** Monte Carlo for probability estimation:

```
dX = -θ(X - μ)dt + σdW

500 paths × 600 steps (10 minutes at 1s)
Track: P(hit center) = tunnel probability
       P(hit ±3σ)    = escape probability
```

### 6.8 Timeframe Belief Network

**File:** `training/timeframe_belief_network.py` — Class `TimeframeBeliefNetwork`

N parallel workers, each monitoring a different timeframe:

```
Worker TF  │ Update Every │ Role
───────────┼──────────────┼──────────────────────
1h         │ 240 bars     │ Macro trend context
30m        │ 120 bars     │ Intermediate structure
15m        │  60 bars     │ Decision-level signal
5m         │  20 bars     │ Primary trade decision
3m         │  12 bars     │ Fine-grain confirmation
1m         │   4 bars     │ Precise timing
30s        │   2 bars     │ Execution window
15s        │   1 bar      │ Entry trigger
```

Each worker produces a `WorkerBelief`:
- `dir_prob`: P(LONG) from logistic regression [0..1]
- `pred_mfe`: Predicted MFE in price points
- `conviction`: |dir_prob - 0.5| × 2

**Path Conviction** (psychohistory principle):

```
conviction = weighted_geometric_mean(P(correct_dir) across all active TF levels)
```

Higher-TF beliefs carry more weight (they summarize more history). When all levels agree (1h→30m→15m→5m→1m→15s all LONG), path conviction is very high.

### 6.9 Walk-Forward Training Loop

The inner training loop processes one day at a time:

```python
for day_idx, (date, day_data) in enumerate(atlas_days):
    # 1. Compute quantum states for all bars (GPU batch)
    states = engine.batch_compute_states(day_data, params=current_pid)

    # 2. Scan for pattern events (Roche snaps, structural drives)
    patterns = discovery_agent.scan_day_cascade(day_data, date, states)

    # 3. Match each pattern to nearest template (16D Euclidean distance)
    for pattern in patterns:
        template, distance = clustering.match_to_template(pattern)
        if distance > 4.5:
            continue  # Too far from any known cluster

    # 4. Gate 0: Physics check
        if not passes_physics_gate(pattern.state):
            continue

    # 5. Gate 1: TimeframeBeliefNetwork consensus
        belief = tbn.get_belief()
        if belief.conviction < 0.60:
            continue

    # 6. Simulate trade (Numba-optimized)
        outcome = simulate_trade(pattern, template, day_data, params)

    # 7. Record in Bayesian brain
        brain.update(outcome)

    # 8. End-of-day regret analysis
    regret = batch_regret_analyzer.analyze(day_trades, day_data)
```

### 6.10 Wave Rider Exit System & Regret Analysis

**File:** `training/wave_rider.py` — Classes `WaveRider`, `RegretAnalyzer`

**Trade simulation** uses a Numba JIT-compiled loop with spectral exit logic:

```
For each bar after entry:
  1. Check STOP LOSS (absolute, overrides everything)
  2. FOURIER GATE: Wait for half-cycle to complete before allowing exit
  3. LAPLACE GATE: Exit if kinetic energy is critically damped
  4. Check TAKE PROFIT
  5. Check TIME EXIT (max hold)
```

**Regret Analysis** evaluates exit quality post-trade:

```
RegretMarkers:
  actual_pnl         — What we got
  potential_max_pnl   — What was possible (MFE)
  exit_efficiency     — actual / potential (0.0 to 1.0)
  regret_type         — 'closed_too_early', 'closed_too_late', 'optimal'
  pnl_left_on_table   — Potential minus actual
  gave_back_pnl       — How much was surrendered after peak
```

---

## 7. Multi-Timeframe Architecture

**File:** `core/multi_timeframe_context.py` — Class `MultiTimeframeContext`

The system operates across 8 timeframe layers with progressive context availability:

```
Layer │ Timeframe │ Available │ Role
──────┼───────────┼───────────┼──────────────────────────────
  1   │ Daily     │ Day 22+   │ Macro trend (BULL/BEAR/RANGE) + vol regime
  2   │ 4-Hour    │ Day 4+    │ Session context (ASIA/EUROPE/US/OVERLAP)
  3   │ 1-Hour    │ Day 2+    │ Intraday wave structure
  4   │ 15-Min    │ Day 1+    │ PRIMARY DECISION LAYER (always available)
  5   │ 5-Min     │ Day 1+    │ Pattern setup
  6   │ 1-Min     │ Day 1+    │ Confirmation
  7   │ 15-Sec    │ Day 1+    │ Tactical entry (primary engine resolution)
  8   │ 1-Sec     │ Day 1+    │ Execution precision
```

**Context levels:**
- `MINIMAL` (Day 1): Only 15m and below — confidence modifier 0.6
- `PARTIAL` (Day 2-21): + 1h and/or 4h — confidence modifier 0.8
- `FULL` (Day 22+): All timeframes — confidence modifier 1.0

Each higher timeframe computes:
- **Trend** (linear regression slope significance): UP/DOWN/RANGE
- **Volatility** (recent vs historical std ratio): HIGH/NORMAL/LOW
- **Fractal pattern** (geometric detection): COMPRESSION/WEDGE/BREAKDOWN/NONE
- **Session** (4h only): ASIA/EUROPE/US/OVERLAP

---

## 8. CUDA / GPU Acceleration

**File:** `core/cuda_physics.py` — Numba CUDA kernels

The system is designed for NVIDIA GPU acceleration via PyTorch (CUDA 12.1) and Numba:

**4 CUDA Kernels:**

| Kernel | Purpose | Parallelism |
|---|---|---|
| `compute_physics_kernel` | Regression, z-score, forces, wave function | One thread per bar |
| `detect_archetype_kernel` | ROCHE_SNAP and STRUCTURAL_DRIVE detection | One thread per bar |
| `compute_dm_tr_kernel` | True Range + Directional Movement (ADX Pass 1) | One thread per bar |
| `compute_hurst_kernel` | R/S Hurst exponent (4 sub-windows) | One thread per bar |

**GPU Strategy:**
```
CUDA First → CPU Fallback

try:
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        # GPU path
    else:
        raise RuntimeError("no CUDA")
except Exception:
    # CPU fallback (NumPy vectorized)
```

**CUDAKMeans** (`training/cuda_kmeans.py`): GPU-accelerated K-Means for clustering in the main process. Worker subprocesses use sklearn CPU KMeans.

**Data requirements for GPU transfer:**
- All arrays must be `np.float64` and `C_CONTIGUOUS`
- Kernel launch: 256 threads per block, `ceil(N/256)` blocks per grid

---

## 9. Risk Management

**Multiple layers of risk control:**

1. **Physics Gates** — Only trade at Roche limits with confirmed structure
2. **Hurst Filter** — H < 0.5 means chop, halt execution
3. **Regime Filter** — Don't short bull expansions, don't buy bear crashes
4. **Momentum Override** — F_momentum > 1.5 × F_reversion = breakout, don't fade
5. **Bayesian Gate** — P(win) ≥ 80%, Confidence ≥ 30%
6. **TBN Consensus** — Path conviction ≥ 60% across timeframe workers
7. **Cluster Distance** — Template match distance < 4.5 (Euclidean in 16D)
8. **Stop Loss** — Absolute stop at event horizon (±3σ)
9. **Slippage Model** — Base 0.25 points + velocity-dependent factor
10. **Monte Carlo Validation** — 10,000 simulations for drawdown estimation

**Opportunity Cost ("Marshmallow Test"):**

Before entering a trade, the system checks if waiting for a connected higher-EV template has better risk-adjusted returns.

---

## 10. Live Trading System

**Files:** `live/` directory

The live system mirrors the training Phase 4 forward pass but operates on real-time data:

```
┌──────────────┐     TCP     ┌──────────────┐     Quantum     ┌──────────────┐
│ NinjaTrader 8│────────────▶│  NT8Client   │───────Engine────▶│ LiveEngine   │
│ (C# Bridge)  │◀────────────│ (asyncio TCP)│◀───────────────── │ (Python)     │
│              │  POSITION   │              │    SIGNAL         │              │
│              │  HEARTBEAT  │              │    SUBSCRIBE      │              │
│              │  BAR DATA   │              │    ORDER          │              │
└──────────────┘             └──────────────┘                  └──────────────┘
```

**Key simplifications vs training:**
- No oracle markers (no lookahead in live)
- No regret tracking
- Single "best candidate" per bar (no multi-TF cascade scan)
- Real equity from NT8 POSITION messages

**LiveEngine loop:**
1. Receive 15s bar from NT8 bridge
2. Update LiveBarAggregator
3. Compute quantum state
4. Match to template
5. Check gates (physics, TBN, Bayesian)
6. If approved: send order via OrderManager
7. Track position, trail stops
8. Log signals and trades

---

## 11. Visualization & Dashboard

**File:** `visualization/live_training_dashboard.py` — Tkinter "Fractal Command Center"

Real-time training dashboard showing:
- Equity curve
- Pareto chart of profit gap (Missed, Wrong Dir, Too Early, Noise)
- Template performance table
- PID parameter evolution
- Phase progression
- Per-day PnL and Sharpe

**File:** `notebooks/dashboard.ipynb` — Jupyter interactive dashboard

---

## 12. Configuration & Constants

### config/settings.py
```python
OPERATIONAL_MODE = "LEARNING"      # "LEARNING" or "EXECUTE"
RAW_DATA_PATH = "DATA/RAW"
ANCHOR_DATE = "2025-07-30"
DEFAULT_BASE_SLIPPAGE = 0.25       # points
DEFAULT_VELOCITY_SLIPPAGE_FACTOR = 0.1
```

### config/symbols.py — Asset Profiles
```python
MNQ = AssetProfile(
    ticker="MNQ",
    tick_size=0.25,     # $0.50/tick
    point_value=2.0,    # $2/point
    base_price=21500.0
)
```

### config/oracle_config.py — Oracle Thresholds
```python
ORACLE_LOOKAHEAD_BARS = {'15s': 60, '5m': 24, '1h': 8, ...}
ORACLE_MIN_MOVE_TICKS = 5
ORACLE_HOME_RUN_RATIO = 3.0   # MFE/MAE for MEGA classification
```

### Physics Constants
```python
# Regression
REGRESSION_PERIOD = 21            # Rolling linear regression window

# Field Structure
SIGMA_ROCHE_MULTIPLIER = 2.0     # ±2σ = Roche limits (singularities)
SIGMA_EVENT_MULTIPLIER = 3.0     # ±3σ = Event horizons (point of no return)

# Gravity
GRAVITY_THETA = 0.5              # Mean-reversion pull strength
REPULSION_EPSILON = 0.01         # Prevent division by zero
REPULSION_FORCE_CAP = 100.0      # Maximum repulsive force

# Archetypes
VELOCITY_THRESHOLD = 0.5         # Min velocity for ROCHE_SNAP
MOMENTUM_THRESHOLD = 5.0         # Min momentum for STRUCTURAL_DRIVE
COHERENCE_THRESHOLD = 0.3        # Max coherence for STRUCTURAL_DRIVE

# PID (Defaults)
DEFAULT_PID_KP = 0.5             # Proportional gain
DEFAULT_PID_KI = 0.1             # Integral gain
DEFAULT_PID_KD = 0.2             # Derivative gain

# Indicators
ADX_PERIOD = 14                  # Wilder smoothing period
HURST_WINDOW = 100               # R/S calculation window

# Clustering
MAX_CLUSTER_DISTANCE = 4.5       # Max Euclidean dist for template match
MIN_GROUP_SIZE = 30              # Min patterns per cluster
```

---

## 13. Data Flow: End-to-End Walkthrough

Here is the complete data flow for a single training iteration:

```
RAW DATA (.dbn.zst)
       │
       ▼
  DatabentoLoader.load_file()
       │ Decode compressed Databento trade data
       ▼
  FractalAtlasBuilder.build()
       │ Resample 1s bars → 14 timeframes → parquet
       ▼
  DATA/ATLAS_1MONTH/{tf}/{year_month}.parquet
       │
       ▼
  FractalDiscoveryAgent.scan_atlas()
       │ Top-down hierarchical scan: 1D → 4h → 1h → ... → 1s
       │ At each TF:
       │   1. Load parquet file
       │   2. QuantumFieldEngine.batch_compute_states(data)
       │   3. Identify ROCHE_SNAP / STRUCTURAL_DRIVE events
       │   4. Oracle: compute MFE/MAE/markers with lookahead
       │   5. Build parent_chain ancestry
       ▼
  List[PatternEvent]  (~thousands of patterns)
       │
       ▼
  QuantumFieldEngine.build_16d_vector()
       │ Extract 16D feature vector per pattern
       ▼
  FractalClusteringEngine.cluster()
       │ 1. StandardScaler normalization
       │ 2. IMR geometric splitting (HypervolumeTree)
       │ 3. Each leaf → PatternTemplate
       │ 4. Aggregate statistics (win_rate, expectancy, etc.)
       │ 5. Fit regression models (MFE, direction)
       │ 6. Build transition probability matrix
       ▼
  Dict[int, PatternTemplate]  (pattern library)
       │
       ▼
  Walk-Forward Day Loop (orchestrator.py)
       │
       │  For each day:
       │    │
       │    ▼
       │  QuantumFieldEngine.batch_compute_states(day_data, params)
       │    │ GPU: 4 CUDA kernels (physics, archetypes, ADX, Hurst)
       │    │ CPU: Vectorized NumPy fallback
       │    ▼
       │  List[ThreeBodyQuantumState]  (one per bar, ~5300/day)
       │    │
       │    ▼
       │  Pattern Matching: state → nearest template (16D distance)
       │    │
       │    ├── Gate 0: Physics (ADX<25 or Hurst≥0.6 for trending)
       │    ├── Gate 1: Distance < 4.5
       │    ├── Gate 2: TBN conviction ≥ 0.60
       │    └── Gate 3: BayesianBrain P(win) ≥ threshold
       │    │
       │    ▼
       │  simulate_trade_standalone()  (Numba JIT)
       │    │ 1. Determine direction from template bias
       │    │ 2. Set TP/SL from oracle-calibrated parameters
       │    │ 3. Fast loop through future bars
       │    │ 4. Spectral exits (Fourier gate, Laplace gate)
       │    ▼
       │  TradeOutcome(state, entry, exit, pnl, result)
       │    │
       │    ▼
       │  BayesianBrain.update(outcome)
       │    │ table[state]['wins'] += 1  or  table[state]['losses'] += 1
       │    ▼
       │  BatchRegretAnalyzer.analyze(day_trades)
       │    │ Compute regret markers, opportunity cost
       │    ▼
       │  ProgressReporter.report(metrics)
       │
       ▼
  DOE Optimization (per cluster)
       │ Optuna TPE: optimize PID (kp, ki, kd) → maximize Sharpe
       ▼
  Checkpoint Save
       │ brain.save() → models/quantum_probability_table.pkl
       │ binner.save() → checkpoints/binner.pkl
       │ templates → checkpoints/pattern_library.pkl
       ▼
  REPORTS
       ├── reports/oos/phase4_report.txt
       ├── reports/oos/trade_analytics.txt
       └── run_logs/signal_log_{year}_{month}.csv
```

---

## 14. Testing & Validation

**Test suite:** 36+ test files in `tests/`

| Test File | What It Validates |
|---|---|
| `test_bayesian_brain.py` | Core decision logic, probability calculations |
| `test_quantum_field_engine.py` | Vectorized math engine, state computation |
| `test_integration_quantum.py` | Full system integration |
| `test_cpu_physics.py` | CPU fallback correctness |
| `test_cuda_physics.py` | GPU kernel correctness |
| `test_dynamic_binner.py` | Histogram binning accuracy |
| `test_state_vector.py` | Hash/equality for state lookups |
| `test_three_body_exits.py` | Exit logic validation |
| `test_wave_rider.py` | Wave Rider exit system |
| `test_timeframe_belief_network.py` | TBN consensus logic |
| `test_clustering_integration.py` | Clustering pipeline |
| `test_doe_features.py` | DOE parameter generation |
| `test_fractal_atlas.py` | Atlas building |
| `test_pattern_recognition.py` | Pattern detection |
| `test_training_validation.py` | Training loop correctness |
| `test_performance_optimizations.py` | Speed benchmarks |

**Running tests:**
```bash
python -m pytest                          # All tests
python -m pytest tests/test_bayesian_brain.py  # Specific module
python tests/topic_math.py                 # Math verification suite
```

**Validation topics:**
- `topic_build.py`: Executable build, module imports, operational mode
- `topic_math.py`: Physics equations, probability calculations
- `topic_diagnostics.py`: CUDA availability, data paths, system health

---

## 15. Legacy Architecture (Archived)

The original **9-Layer Hierarchy** engine is preserved in `archive/`:

```
archive/
├── layer_engine.py              # 9-layer state engine
├── cuda_modules/                # Legacy CUDA acceleration
│   ├── confirmation.py
│   ├── hardened_verification.py
│   ├── pattern_detector.py
│   └── velocity_gate.py
├── old_core/                    # Original core engines
│   ├── engine_core.py
│   ├── fractal_three_body.py
│   └── resonance_cascade.py
└── old_training/                # Original training scripts
```

The legacy `StateVector` (9-layer) is still in `core/state_vector.py` for backward compatibility but is not used in the active pipeline.

**9-Layer hierarchy (deprecated):**
```
STATIC LAYERS (Session-level):
  L1: 90d bias (bull/bear/range)
  L2: 30d regime (trending/chopping)
  L3: 1wk swing (higher_highs/lower_lows/sideways)
  L4: Daily zone (at_support/at_resistance/at_killzone)

FLUID LAYERS (Intraday):
  L5: 4hr trend (up/down/flat)
  L6: 1hr structure (bullish/bearish/neutral)
  L7: 15m pattern (flag/wedge/compression/breakdown)
  L8: 5m confirmation (True/False)
  L9: 1s velocity trigger (True/False)
```

---

## 16. Glossary

| Term | Definition |
|---|---|
| **Atlas** | Multi-timeframe parquet dataset (1s to 1W bars) |
| **Barrier Height** | OU potential energy: V(B) - V(z), determines tunneling difficulty |
| **BayesianBrain** | HashMap-based learning system: state → win/loss counts |
| **Coherence** | Shannon entropy / ln(3). 0=collapsed (decisive), 1=superposition (uncertain) |
| **CST** | Coherent Structure Tether — validates pattern consistency via 16D vector distance |
| **DMAIC** | Define-Measure-Analyze-Improve-Control (Six Sigma framework) |
| **DOE** | Design of Experiments — systematic parameter optimization |
| **DynamicBinner** | Freedman-Diaconis histogram binner for state discretization |
| **Event Horizon** | ±3σ boundary — point of no return (extreme breakout) |
| **Fission** | Splitting a cluster into sub-clusters for better specificity |
| **FN (False Negative)** | Real move that was gate-blocked (opportunity cost) |
| **Golden Path** | Maximum achievable PnL via non-overlapping trades |
| **Hurst Exponent** | Fractal dimension: H<0.5=chop, H=0.5=random, H>0.5=trending |
| **Hypervolume Tree** | Hierarchical clustering structure where each node spans a 16D cell |
| **IMR** | Individual-Moving Range control chart for cluster splitting |
| **Lagrange Zone** | Market regime classification: L1_STABLE, CHAOS, L2/L3_ROCHE |
| **MFE/MAE** | Max Favorable/Adverse Excursion — best and worst price after entry |
| **Nightmare Protocol** | The physics model: gravity wells, O-U process, quantum mechanics |
| **Oracle** | Training-only lookahead labeler (no lookahead in live) |
| **OU Process** | Ornstein-Uhlenbeck: dX = -θ(X-μ)dt + σdW (mean-reverting SDE) |
| **PID** | Proportional-Integral-Derivative control force model |
| **Roche Limit** | ±2σ boundary — gravitational tidal force boundary |
| **ROCHE_SNAP** | Pattern: |z|>2 AND |velocity|>0.5 — price at Roche limit with momentum |
| **Sigma (σ)** | Standard deviation of regression residuals (volatility measure) |
| **STRUCTURAL_DRIVE** | Pattern: |momentum|>5 AND coherence<0.3 — strong directional force |
| **TBN** | Timeframe Belief Network — multi-TF consensus engine |
| **Template** | Cluster centroid + statistics — represents a class of similar patterns |
| **Tunnel Probability** | P(revert to center before hitting event horizon) |
| **Wave Function** | Probability distribution across 3 attractors: ψ = a₀|0⟩ + a₁|+2σ⟩ + a₂|-2σ⟩ |
| **z-score** | Normalized distance: (price - center) / σ |

---

*This guide was auto-generated from source code analysis of the Bayesian-AI repository.*
