# Bayesian-AI

> **High-Frequency Algorithmic Trading System for MNQ Futures**

Bayesian-AI is a high-frequency algorithmic trading system for US equity index futures (MNQ). It models the market using OLS regression bands with exact z-score standardization via a zero-lookahead Statistical Field Engine (SFE).

The architecture has evolved from discrete Bayesian probability learning into a **Parallel Worlds Curriculum Reinforcement Learning (PW-CRL)** engine, combined with a **V2 Live Execution Engine** handling streaming state.

**Primary instrument:** Micro Nasdaq-100 futures (MNQ)
**Compute:** NVIDIA CUDA GPU required (CPU fallback removed)

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run V2 Feature Extraction (Builds 200D/297D Parquets)
python core_v2/build_dataset.py --atlas DATA/ATLAS_NT8 --fresh

# RL Training Pipeline (Parallel Worlds Curriculum)
python training/rl_engine/train_historical.py --agent-type ENTRY_NMP

# Live trading V2 (dry run via L5 zigzag engine)
python -m live.engine_v2 --engine-mode l5 --mock

# Run visualization engine (e.g., trade visualizer)
python -m tools.viz.run --plugin trade_visualizer
```

## Documentation

- [System Description](docs/SYSTEM_DESCRIPTION.md) — architecture overview, components, algorithms
- [Architecture Reference](docs/ARCHITECTURE.md) — file/class/line-count mapping + dependency graph
- [RL Whitepaper](rl_whitepaper.md) — mechanics of the PW-CRL implementation
- [Agent Instructions](AGENTS.ini) — guidelines and entry points for AI agents
- [Changelog](docs/CHANGELOG.md) — version history

## System Architecture

The project is currently split into two main streams: the robust **V2 Streaming Engine** (live trading) and the **RL Engine** (mid-training).

### Core Components:
- **Statistical Field Engine (SFE)** — GPU-accelerated OLS regression bands, exact z-scores, rolling variance ratio proxies, and Nightmare Protocol ($\hat{\lambda}$) states. Zero lookahead guaranteed via trailing Numba kernels.
- **Forward Pass System (FPS)** — Causal VRAM-aware forward pass simulator that executes historical environments perfectly aligned with the live streaming logic.
- **Ledger** — Immutable position and PnL ledger ensuring strict state boundaries between features and decisions.
- **Parallel Worlds Curriculum RL (PW-CRL)** — CNN+LSTM DQN engine featuring V-trace, hindsight-regret shadow queues, and an `EXIT_NMP -> ENTRY_NMP -> YOLO` curriculum over an 8-agent DOE.
- **Live V2 Engine** — Live trading stack connecting the SFE directly to NinjaTrader 8 or Databento, currently running the `L5_Decider` / `zigzag` state logic.

## Key Directories

```text
core_v2/        Core V2 engine modules (SFE, FPS, ledger, features, strategy engine)
training/       RL engine (rl_engine), regret algorithms, and legacy isolated tools
live/           Live trading stack (V2 engine, L5 sidecar, NT8/Databento shims)
tools/          Production tools, data pipelines, VizEngine (tools/viz)
DATA/           ATLAS parquet datasets (gitignored)
reports/        Run outputs, charts, daily journal logs
docs/           Documentation, memory banks, daily changelogs
```
