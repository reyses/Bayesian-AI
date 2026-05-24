# Architecture Reference — Bayesian-AI
> Last updated: 2026-03-08. CUDA-only (CPU fallback removed).

## Core Pipeline

| File | Lines | Main Class/Function | Purpose |
|------|------:|---------------------|---------|
| `core_v2/statistical_field_engine.py` | 489 | `StatisticalFieldEngine` | GPU-accelerated OLS regression bands, z-scores, OU probabilities, forces, entropy per bar |
| `core_v2/market_state.py` | 255 | `MarketState` (frozen) | 50+ field snapshot: bands, z-score, velocity, forces, probabilities, ADX/DMI, Hurst |
| `core_v2/fractal_clustering.py` | 651 | `FractalClusteringEngine` | Recursive K-Means → 100-1000 templates with per-template OLS/logistic models |
| `core_v2/execution_engine.py` | 976 | `ExecutionEngine` | Gate cascade (0-4), direction cascade, sizing. Single source for IS/OOS/live |
| `core_v2/exit_engine.py` | 710 | `ExitEngine`, `PositionState`, `make_position()` | Unified exit cascade: SL→TP→BandUrgent→EnvelopeDecay→PeakGiveback→BreakevenLock→BeliefFlip→Hold |
| `core_v2/feature_extraction.py` | 53 | `extract_feature_vector()` | Canonical 16D feature vector (single source of truth) |
| `core_v2/timeframe_belief_network.py` | 1018 | `TimeframeBeliefNetwork` | 11 TF workers (1s→1D), path conviction, band confluence, direction models |
| `core_v2/bayesian_brain.py` | 298 | `BayesianBrain` | Hash table: state_key → {wins, losses}. Direction learning per template |

## Core Support

| File | Lines | Purpose |
|------|------:|---------|
| `core_v2/cuda_statistics.py` | 320 | Numba CUDA kernels for physics (regression, forces, probabilities) |
| `core_v2/cuda_pattern_detector.py` | 189 | GPU-accelerated geometric + candlestick pattern detection |
| `core_v2/pattern_utils.py` | 139 | Vectorized pattern detection (CPU path) |
| `core_v2/physics_utils.py` | 93 | ADX/DMI/Hurst constants |
| `core_v2/risk_engine.py` | 146 | Monte Carlo OU solver (legacy — analytical erfi in SFE supersedes) |
| `core_v2/state_vector.py` | 80 | Phase 1 prototype state (deprecated, kept for test compat) |

## Live Stack

| File | Lines | Main Class | Purpose |
|------|------:|------------|---------|
| `live/live_engine.py` | 1785 | `LiveEngine` | Main live trading loop — mirrors Phase 4 on real-time NT8 feed |
| `live/history_replay.py` | 524 | `HistoryReplayEngine` | Compressed forward pass over ATLAS to warm up brain/TBN/exits |
| `live/atlas_loader.py` | 143 | `load_atlas_range()` | Parquet I/O for ATLAS multi-TF dataset |
| `live/bar_aggregator.py` | 295 | `LiveBarAggregator` | 1s→15s OHLCV aggregation with state recomputation |
| `live/nt8_client.py` | 217 | `NT8Client` | Asyncio TCP bridge to NinjaTrader 8 BayesianBridge indicator |
| `live/order_manager.py` | 302 | `OrderManager` | Order lifecycle, position shadow, trade CSV logging |
| `live/session_tracker.py` | 289 | `SessionTracker` | Session PnL, drawdowns, trade log, session reports |
| `live/ping_pong.py` | 117 | `PingPongManager` | Flip direction + ATR sizing for continuous wave-riding |
| `live/exit_watcher.py` | 86 | `ExitWatcher` | Post-exit counterfactual tracking (regret analysis) |
| `live/gui_bridge.py` | 79 | `GUIBridge` | Non-blocking queue wrapper for Tk dashboard |
| `live/trade_logger.py` | 64 | `TradeLogger` | Per-trade diagnostic CSV |
| `live/protocol.py` | 187 | Message encoding/decoding | Wire protocol for NT8 TCP communication |
| `live/launcher.py` | 246 | `main()` | Entry point for live mode |

## Training

| File | Lines | Purpose |
|------|------:|---------|
| `training/run.py` | - | Main entry point: orchestrated forward pass and evaluation pipeline |
| `training/orchestrator_worker.py` | 617 | Numba JIT simulation loops, spectral gates (Fourier half-cycle + Laplace damping) |
| `training/fractal_discovery_agent.py` | 830 | Multi-TF fractal scan → PatternEvent manifest with oracle markers |
| `training/trade_analytics.py` | 562 | Post-run analytics: t-tests, ANOVA, OLS, logistic, capture rate regression |
| `training/pattern_analyzer.py` | 548 | State table analysis, strongest patterns, contextual breakdowns |
| `training/cuda_kmeans.py` | — | GPU-accelerated K-Means with sklearn fallback |

## Visualization

| File | Lines | Purpose |
|------|------:|---------|
| `tools/viz/core/engine.py` | - | VizEngine: Unified visualization framework supporting plugins, dynamic panning, and ribbon UI |
| `tools/viz/plugins/*.py` | - | Pluggable visualization tools: classifier_inspector, feature_marker, etc. |

## Dependency Flow

```
trainer.py
├── statistical_field_engine.py → cuda_statistics.py, market_state.py
├── fractal_discovery_agent.py → bayesian_brain.py
├── fractal_clustering.py → feature_extraction.py, cuda_kmeans.py
├── execution_engine.py → timeframe_belief_network.py, exit_engine.py
├── orchestrator_worker.py (Numba JIT)
├── trade_analytics.py
└── dashboard.py

live_engine.py
├── statistical_field_engine.py
├── execution_engine.py, exit_engine.py
├── timeframe_belief_network.py, bayesian_brain.py
├── history_replay.py → atlas_loader.py
├── nt8_client.py → protocol.py
├── bar_aggregator.py, order_manager.py
├── session_tracker.py, ping_pong.py
├── exit_watcher.py, gui_bridge.py, trade_logger.py
└── dashboard.py
```

## Data Flow

```
ATLAS Parquet (14 TF x 12 months)
  → StatisticalFieldEngine.batch_compute_states()
  → MarketState per bar
  → FractalDiscoveryAgent (pattern events)
  → FractalClusteringEngine (templates)
  → ExecutionEngine (gate cascade + direction)
  → ExitEngine (exit cascade)
  → BayesianBrain (learn)
  → Trade Analytics (report)

Live: NT8 1s ticks → BarAggregator → SFE → EE → ExitEngine → OrderManager → NT8
```

## Key Algorithms

| Algorithm | Location | Description |
|-----------|----------|-------------|
| OLS regression bands | `statistical_field_engine.py` | Rolling linear regression +/- 2s/3s |
| OU first-passage | `statistical_field_engine.py` | Analytical erfi-based reversion probability |
| 3-class softmax | `statistical_field_engine.py` | P(center), P(upper), P(lower) via squared amplitudes |
| Shannon entropy | `statistical_field_engine.py` | Superposition measure (0=collapsed, 1=mixed) |
| Path conviction | `timeframe_belief_network.py` | Weighted geometric mean of per-TF P(direction) |
| Envelope decay | `exit_engine.py` | Self-tuning halflife (8-60 bars) + giveback tracking |
| Spectral gates | `orchestrator_worker.py` | Fourier half-cycle + Laplace kinetic damping |

## Total: ~18,900 lines across core/, live/, training/, visualization/
