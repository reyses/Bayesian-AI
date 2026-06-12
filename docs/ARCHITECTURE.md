# Architecture Reference — Bayesian-AI
> Last updated: 2026-06-11. RL Pivot & λ-Completion (V2 Stack). CUDA-only (CPU fallback removed).

## 1. Core Pipeline (V2 Stack)

The foundational computation layer that guarantees zero-lookahead feature extraction and manages streaming state.

| File | Main Class/Function | Purpose |
|------|---------------------|---------|
| `core_v2/statistical_field_engine.py` | `StatisticalFieldEngine` | GPU-accelerated OLS regression bands, exact z-scores, rolling variance ratio proxies, and Nightmare Protocol ($\hat{\lambda}$) states. Zero lookahead via trailing Numba kernels. |
| `core_v2/features.py` | `extract_features()` | Canonical V2 feature definitions. |
| `core_v2/live_features.py` | `LiveFeatureEngine` | Single-bar sequential feature extraction matching the exact output of the bulk dataset builder. |
| `core_v2/ledger.py` | `PositionLedger` | Immutable position and PnL ledger ensuring strict state boundaries between execution and feature spaces. |
| `core_v2/exits.py` | `ExitEngine` | Nightmare Protocol (NMP) exit models and λ-based exit completion logic. |
| `core_v2/engine_signals.py` | `SignalGenerator` | Maps structured features to entry signals and gate cascades. |
| `core_v2/sim_executor.py` | `ForwardPassSystem` | Causal simulator (FPS) executing historical environments perfectly aligned with live streaming logic. |
| `core_v2/build_dataset.py` | `main()` | Parallel multi-timeframe Parquet dataset generation (e.g., 200D/297D) using SFE. |

## 2. Live Stack

The live trading environment bridging the V2 core logic to the broker API.

| File | Main Class | Purpose |
|------|------------|---------|
| `live/engine_v2.py` | `LiveEngineV2` | Main live trading loop processing tick-data through the SFE and dispatching orders. |
| `live/nt8_client.py` | `NT8Client` | Asyncio TCP bridge to NinjaTrader 8. |
| `live/order_manager.py` | `OrderManager` | Order lifecycle tracking and position shadowing. |
| `live/l5_decider.py` | `L5Decider` | Implementation of the zigzag and active execution logic algorithms. |
| `live/dashboard_v2.py` | `LiveDashboard` | Real-time CLI/curses-based telemetry and visual system monitoring. |
| `live/session_tracker.py` | `SessionTracker` | Diagnostic telemetry processing and session state management. |

## 3. Training & Science

The advanced ML and diagnostic tools layered on top of the V2 Core.

| File | Purpose |
|------|---------|
| `training/rl_engine/` | Parallel Worlds Curriculum Reinforcement Learning (PW-CRL) implementation: CNN+LSTM DQN engine featuring V-trace and hindsight-regret shadow queues. |
| `tools/convert_nt8_atlas.py` | Data ingestion utility that converts raw broker exports to the internal Parquet ATLAS format. |
| `tools/trade_visualizer.py` | Trade inspection tool displaying chart segments overlaid with entry/exit points and regime bounds. |
| `tools/parity_validate.py` | Verifies the output parity between historical `build_dataset.py` forward-passes and `engine_v2.py` live logs. |

## Dependency Flow

```text
build_dataset.py
├── statistical_field_engine.py
├── features.py
└── sim_executor.py
    ├── exits.py
    └── engine_signals.py

live/engine_v2.py
├── core_v2/live_features.py → core_v2/statistical_field_engine.py
├── core_v2/ledger.py
├── live/l5_decider.py
├── live/order_manager.py
├── live/nt8_client.py
└── live/dashboard_v2.py

training/rl_engine/train_historical.py
└── core_v2/sim_executor.py
```

## Data Flow Pipeline

**1. Historical & Training (FPS)**
```text
Raw NT8 Export → tools/convert_nt8_atlas.py → ATLAS Parquet
  → core_v2/build_dataset.py (SFE) → Multi-TF Feature Parquets
  → training/rl_engine/train_historical.py (DQN) 
  → core_v2/sim_executor.py (Ledger & Exits) → Evaluator Logs
```

**2. Live Execution**
```text
NT8 Ticks → live/nt8_client.py → live/engine_v2.py
  → core_v2/live_features.py (SFE)
  → core_v2/engine_signals.py
  → live/l5_decider.py
  → core_v2/ledger.py (Position Update)
  → live/order_manager.py → live/nt8_client.py (Fill)
```
