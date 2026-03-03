# Bayesian-AI Project Memory
> Future topics backlog: see `memory/ROADMAP.md`
> Daily interaction journals: `docs/daily/YYYY-MM-DD.md`

## Workflow Preference
- **Always discuss before changing**: propose a plan, get approval, then execute
- **Progress bars are mandatory**: any long-running loop MUST have tqdm with live stats
- **NEVER run training via Bash** — user runs it themselves
- **NT8 bridge deploy**: When updating `docs/NT8_BayesianBridge.cs`, also copy it to
  `C:\Users\reyse\OneDrive\Documents\NinjaTrader 8\bin\Custom\Indicators\NT8_BayesianBridge.cs`

## Architecture
- **Core engine**: `core/quantum_field_engine.py` — ThreeBodyQuantumState per bar
- **Brain**: `core/bayesian_brain.py` — Bayesian table with hash-based state lookups
- **Orchestrator**: `training/orchestrator.py` — main loop, DOE/Optuna, forward pass
- **Clustering**: `training/fractal_clustering.py` — DMI pre-split → I-MR(DMI diff) → DBSCAN(vol+ADX)
- **Belief Network**: `training/timeframe_belief_network.py` — 10 TF workers, price-aware
- **Wave Rider**: `training/wave_rider.py` — position management, exits, CST
- **DNA Tree**: `training/fractal_dna_tree.py` — hierarchical TF context tree
- **Feature vector**: 16D — abs(z), log1p(v), log1p(m), coherence, tf_scale, depth,
  parent_ctx, self_adx, self_hurst, self_dmi_diff, parent_z, parent_dmi_diff,
  root_is_roche, tf_alignment, self_pid, osc_coh

## Report & Output Locations
> Source of truth: `_get_reports_dir(mode)` in `training/orchestrator.py` line ~43
> Pattern: `reports/{mode}/` where mode = is | oos | phase5 | training

| Path | Contents | Notes |
|------|----------|-------|
| `reports/is/` | IS forward pass: oracle_trade_log.csv, fn_oracle_log.csv, signal_log.csv, pid_oracle_log.csv, phase4_report.txt, trade_analytics.txt + `shards/` | Gitignored (CSVs) |
| `reports/oos/` | OOS forward pass: same structure as IS | Gitignored (CSVs) |
| `reports/phase5/` | phase5_report.txt (strategy selection) | .txt tracked |
| `reports/training/` | training_log.txt | Gitignored |
| `reports/benchmarks/` | Timestamped snapshot dirs + history.csv | Gitignored |
| `run_logs/` | Monthly sharded CSVs: oracle_trade_log_YYYY_MM.csv, signal_log_*, pid_signal_log_*, fn_signal_log_* | Gitignored |
| `output/diagnostics/` | Diagnostic CSVs (gate0_rule4, gate1_nomatch, etc.) | Future: SIGNAL_CAPTURE_AUDIT |
| `docs/checkpoint_reference/` | SCHEMAS.md, run_snapshot.json, depth_weights.json, samples | Tracked |

**Gitignore rules**: `reports/**/*.csv`, `reports/training/training_log.txt`
**To read latest IS results**: `reports/is/oracle_trade_log.csv` (or shards)
**To read latest OOS results**: `reports/oos/oracle_trade_log.csv`

## CLI Flags (current)
- `--fresh` — wipe ALL checkpoints + full pipeline
- `--train-only` — Phases 2-3 only
- `--forward-pass` — IS → Strategy → OOS auto-chain (existing library)
- `--forward-pass --skip-oos` — IS → Strategy only
- `--oos` — standalone OOS rerun (uses DATA/ATLAS_OOS)
- `--forward-data PATH` — custom data for forward pass (skips auto-OOS)
- `--data DATA/ATLAS_1DAY` — single-day fast validation (~3s)
- `--strategy-report` — Phase 5 only

## Data Pipeline
- ATLAS: `DATA/ATLAS/{tf}/YYYY_MM.parquet` — 14 TFs, 10 months (Jan-Oct 2025)
- ATLAS_1DAY: `DATA/ATLAS_1DAY/` — single day (Jan 2) for fast validation
- ATLAS_1WEEK: `DATA/ATLAS_1WEEK/` — 7 trading days (Jan 2-10) for screening
- ATLAS_OOS: `DATA/ATLAS_OOS/` — 2 months (Jan-Feb 2026)
- Raw 1s: `DATA/glbx-mdp3-20250101-20260209.ohlcv-1s.parquet` (13.2M bars)

## Analysis & Benchmark Tools
- `tools/run_benchmark.py` — chains IS+OOS, snapshots to reports/benchmarks/, --history
- `tools/compare_oos_runs.py` — side-by-side comparison of two runs
- `tools/analyze_exits.py` — deep exit analysis for a single run
- `tools/pattern_map.py` — signal funnel visualization
- `tools/trade_visualizer.py` — trade overlay on price waveform
- `training/run_analytics.py` — re-run analytics without forward pass
- `scripts/monthly_pnl_chart.py` — monthly PnL bar chart
- `tools/make_atlas_1day.py` — create 1-day ATLAS subset
- `tools/make_atlas_1week.py` — create 7-day ATLAS subset
- `tools/waveform_standalone.py` — price-first I-MR screening (see journal)

## Waveform Screening (active research)
- **Journal**: `docs/WAVEFORM_ANALYSIS_JOURNAL.txt` — full methodology + insights
- **Key insight**: segment on price/time FIRST, layer 16D physics on top
- Price I-MR: I=close, MR=signed bar-to-bar change, regimes from UCL breaks
- Default mode = price I-MR only; `--full` = adds 16D fractal pipeline
- Signed MR: every change is a potential pattern, sign flips = direction changes
- **Analyses completed**: A-H (screening), I (seed classification, 20 primitives),
  J (adaptive R² sub-types, IQR quality gate, R² ceiling 0.88),
  K (direction prediction: 70.6% accuracy, +16.5% lift, 4h/1h dominant),
  L-P (various), Q (signed magnitude histogram + paired 192D profiles)
- **Key K findings**: 1h_hurst #1, 4h_osc_coh #2, 15m base TF barely top-10
- **Q status**: Sign-first split working. Adaptive sigma + IQR fallback added.
  Need `--analysis-days 120` to get enough samples (default 7d only gives ~460 bars).
  CLI: `--start X` skips to analysis X, `--cache file.npz` saves/loads feature matrix
- **Integration spec**: `docs/JULES_WAVEFORM_INTEGRATION.md` (5 parts)

## Seed Library & Live Worker Architecture (KEY DECISION)
- **The waveform analysis is OFFLINE research** — too slow for live trading
- **Output = a pre-built SEED LIBRARY**: shape templates (mathematical functions
  with fitted parameters) + price models, serialized as a lookup table
- **Workers receive the library at startup**, NOT the raw analysis code
- **Live workflow**: observe N bars → delta from entry → match to seed library
  (closest function fit) → get shape type + predicted direction + magnitude
- **This replaces templates.pkl**: instead of DBSCAN clusters, workers get
  named mathematical shapes (V-reversal, ramp, sigmoid, etc.) with parameters
- **Price model (92% R²) enriches** the shape match for entry precision
- The seed library is the bridge between offline research and live execution

## Validation Ladder (4 gates, sequential — each must pass)
1. **IS (In-Sample)**: Workers + seed library profit on trained data?
   - `--forward-pass` on ATLAS. Failure = library broken, back to research.
2. **OOS (Out-of-Sample)**: Same library, unseen data — profit holds?
   - Auto-OOS on ATLAS_OOS. Failure = overfit, need more robust shapes.
3. **Live Simulated**: Real-time feed, paper trading — survives real conditions?
   - live/ module. Tests latency, gaps, microstructure. Failure = engineering.
4. **Live Real**: Real money via NT8_BayesianBridge — actual profit?
   - Tests slippage, commissions, psychology. Failure = risk management.

## Jules Specs
- **Active**: `docs/JULES_PERFORMANCE_TARGETS.md` (Phase A/B/C fixes),
  `docs/JULES_DOE_PHASE3.md` (Part 4: I-MR + dp/dt + DBSCAN framework),
  `docs/JULES_AUTO_DOCS.md`,
  `docs/JULES_WAVEFORM_INTEGRATION.md` (5 parts: TF weights, Hurst modulation,
  osc_coh blend, shape gate, feature pruning)
- **Archived**: `docs/old_jules/` — SPECTRAL_GATES, SNOWFLAKE_BASELINE,
  TASK_1_2, TEMPLATE_TIMESCALE, PLAN_PRICE_AWARE_WORKERS, SIGNAL_CAPTURE_AUDIT

## Branches
- `main` — unified clustering, three-body exits, pipeline gauntlet, all fixes
- `pre-snowflake` — **active live trading branch** (best candidate as of 2026-03-02)
  - OOS: 357 trades, 52.4% WR, $1,724.50 — better than main
  - Has live module ported from main + physics quality gate
  - Session reports: `reports/live/session_*.txt`
- Killed: `unified-cluster`, `jules/fractal-trend-*` (deleted 2026-02-27)

## Implemented Features (confirmed in codebase)
- Spectral gates: Fourier half-cycle + Laplace kinetic damping (orchestrator_worker.py)
- Template timescale: avg_mfe_bar/p75_mfe_bar time-exhaustion exits
- Price-aware workers: trade_side + profit_ticks modulate conviction
- Continuous pressure model: net_pressure drives trail widen/tighten/urgent
- Decay cascade: z-score drift from expected trajectory (belief_network)
- OU tunnel probabilities: analytical erfi-based (quantum_field_engine)
- Semantic cluster names: generate_semantic_name() (fractal_clustering)
- Live trading module: live/ (7 files) + docs/NT8_BayesianBridge.cs

## Current Baseline (IS+OOS, 2026-03-01, main branch, pre-integration PRs)
- IS: 392 trades, 33.7% WR, $8,117 total, $20.71/trade
- OOS: 162 trades, 33.3% WR, $3,353 total, $20.69/trade (2 months Jan-Feb 2026)
- Direction: 43.1% correct, all SHORT, zero LONG trades
- Template matching: BROKEN (0 matches, all WORKER_BYPASS, playbook empty)
- Conviction: NON-PREDICTIVE (flat 0.68, p=0.41)
- Best filter: depth<=3 + z<0 → 69.2% WR, $188.83/trade OOS (26 trades)
- Key insight: physics + exit system carry all profit; trail stop = mean reversion catcher
- Prior baseline (2026-02-25): 3,754 trades, 37.5% WR, $1.55/trade

## Clustering Rework (2026-02-27, in progress)
- **Pipeline**: DMI pre-split (LONG/SHORT) → I-MR on signed DMI diff → DBSCAN(vol+ADX)
- **I-MR**: Individual = DMI_diff (signed), sorted by ADX, MR = signed diff
  Boundaries: |MR| > UCL OR sign-flip with significant magnitude
- **Phase 3**: Stepwise MFE refinement (trim 10% outliers per pass), no simulation
  adj-R²(16D→MFE) computed per template
- **IMR_D4**: 3.267 (standard SPC n=2), was wrongly 2.0
- **CLUSTER_DIMS**: [1, 7] = volume + ADX (2D clustering, 16D as identity)

## Design Direction (confirmed via waveform analysis)
- **Shape-first architecture**: Identify shape type (seed function) FIRST,
  then layer 16D physics for magnitude/timing refinement
- **Separation proven**: Shape clustering (Analysis H) produces WR=0% or 98%+
  clusters. Direction comes from shape, precision from physics.
- **16F×12TF as regression constants**: The 192 values explain waveform shape
  variation WITHIN a shape type (why is this V-reversal deeper than that one?)
- **Laplacian = shape identifier**: d²p/dt² tells you ramp (d²=0) vs V-shape
  (d² sign flip) vs exponential (d² same sign as d¹) — already computed
- **Seed functions replace DBSCAN clusters**: Named mathematical functions
  (ramp, V, sigmoid, etc.) instead of opaque cluster IDs

## Requirements
- PyTorch CUDA (cu121), numba, scipy, optuna>=3.5.0
- databento + databento-dbn for data loading
