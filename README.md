# Bayesian-AI

> **Algorithmic Trading System with Statistical Regression Bands, Multi-TF Belief Network & Bayesian Inference**

Bayesian-AI is a high-frequency algorithmic trading system for US equity index futures (MNQ, NQ, ES, MES). It models the market using OLS regression bands with z-score standardization, computes a 3-class softmax probability distribution over price location, then applies Bayesian probability learning to discover which market states lead to profitable trades.

**Primary instrument:** Micro Nasdaq-100 futures (MNQ)
**Compute:** NVIDIA CUDA GPU required (CPU fallback removed)

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Training pipeline (full: discovery -> clustering -> IS -> OOS -> strategy)
python training/run.py

# Live trading (dry run)
python -m live.launcher --dry-run

# Run visualization engine (e.g., trade visualizer)
python -m tools.viz.run --plugin trade_visualizer
```

## Documentation

- [System Description](docs/SYSTEM_DESCRIPTION.md) — architecture overview, components, algorithms
- [Architecture Reference](docs/ARCHITECTURE.md) — file/class/line-count mapping + dependency graph
- [Changelog](docs/CHANGELOG.md) — version history
- [Roadmap](docs/ROADMAP.md) — future work + dependency graph
- [Agent Instructions](AGENTS.md) — guidelines for AI agents working on this codebase

## System Architecture

**Pipeline:** `LOAD -> TRANSFORM -> ANALYZE -> DECIDE -> EXECUTE -> LEARN`

| Phase | Name | Description |
|-------|------|-------------|
| 1 | Data Preparation | Loads ATLAS parquet files (14 TFs, 12 months) |
| 2 | Pattern Discovery | Scans history for statistical events |
| 3 | Template Optimization | Clusters events into templates via GPU K-Means + DOE |
| 4 | IS Forward Pass | Replays history using ExecutionEngine |
| 5 | OOS Validation | Blind out-of-sample validation |
| 6 | Strategy Selection | Ranks templates into tiers |

**Core components:**
- **Statistical Field Engine** — GPU-accelerated OLS regression bands, z-scores, OU probabilities, forces, entropy
- **Execution Engine** — gate cascade + direction cascade + sizing (single source for IS/OOS/live)
- **Exit Engine** — unified cascade: SL -> TP -> BandUrgent -> EnvelopeDecay -> PeakGiveback -> BreakevenLock -> BeliefFlip -> Hold
- **Bayesian Brain** — hash table: state_key -> {wins, losses, P(win)}
- **Timeframe Belief Network** — 11 TF workers (1s through 1D), path conviction, band confluence
- **CUDA acceleration** — Numba kernels for regression, forces, probabilities

## Key Directories

```
core_v2/        Core engine modules (statistical field, brain, execution, exits, TBN)
training/       Training pipeline (consolidated pipelines, discovery, clustering, analytics)
live/           Live trading stack (NT8 bridge, bar aggregation, order management)
tools/          Production tools, research harness, and VizEngine (tools/viz)
DATA/           ATLAS parquet datasets (gitignored)
checkpoints/    Model state (gitignored)
reports/        Run outputs (text tracked, CSVs gitignored)
docs/           Documentation, specs, journals
```
