# Bayesian-AI Project Memory
> Future topics backlog: see `docs/ROADMAP.md`
> Daily interaction journals: `docs/daily/YYYY-MM-DD.md`

## **HARD RULES — DO NOT SKIP**
- **KEEP JOURNALS UPDATED**: At the start and end of EVERY session, update:
  1. `docs/daily/YYYY-MM-DD.md` — daily interaction journal (findings, changes, decisions)
  2. `docs/reference/RESEARCH_JOURNAL.txt` — research journal (analysis results, pipeline, new modules)
  3. `reports/findings/` — standalone finding reports (YYYY-MM-DD_topic.md)
  NEVER let journals go stale. Lost journals = lost days of work.
- **KEEP MEMORY UPDATED**: Update MEMORY.md when discovering new patterns, preferences, or architecture changes

## Workflow Preference
- **All test/research analyses must mirror live trading conditions** — no lookahead, use only
  data available at decision time. Slow TF bars incomplete mid-bar = use last completed bar.
- **Always discuss before changing**: propose a plan, get approval, then execute
- **Progress bars are mandatory**: any long-running loop MUST have tqdm with live stats
- **Training via Bash**: show exact command, ask "Confirm to run?" — only execute after user confirms
- **NT8 bridge deploy**: When updating `docs/NT8_BayesianBridge.cs`, also copy it to
  `C:\Users\reyse\OneDrive\Documents\NinjaTrader 8\bin\Custom\Indicators\NT8_BayesianBridge.cs`
- **NT8 bridge versioning**: Always bump version + update date + time in both header comment
  and `BRIDGE_VERSION` const (e.g. `6.1.2 — 2026-03-04 06:15`)

## Architecture
- **Core engine**: `core/statistical_field_engine.py` — MarketState per bar (CUDA-only since 2026-03-08)
- **Brain**: `core/bayesian_brain.py` — Bayesian table with hash-based state lookups
- **Trainer**: `training/trainer.py` — main entry point, CLI, 6-phase pipeline
  - Run: `python training/trainer.py --fresh --forward-pass`
- **Exit Engine**: `core/exit_engine.py` — unified exit cascade (SL→TP→BandUrgent→
  EnvelopeDecay→BreakevenLock→BeliefFlip→Hold). Trail/MaxHold/Watchdog DISABLED.
- **Execution Engine**: `core/execution_engine.py` — gate/direction/sizing, oracle-driven thresholds
- **Feature Extraction**: `core/feature_extraction.py` — canonical 16D feature vector (single source of truth)
- **Belief Network**: `core/timeframe_belief_network.py` — 11 TF workers (incl 5s, 1s),
  BandContext per worker, `get_band_confluence()` (Priority 4 in direction cascade)
- **Position factory**: `ExitEngine.open_position()` — creates PositionState (make_position deleted 2026-03-09).
  wave_rider.py DELETED (2026-03-07). All position/exit logic in exit_engine.py.
- **Trade Logger**: `live/trade_logger.py` — per-trade diagnostic CSV
- **History Replay**: `live/history_replay.py` — compressed forward pass for live warmup
- **Atlas Loader**: `live/atlas_loader.py` — ATLAS parquet reader for date ranges
- **Exit Watcher**: `live/exit_watcher.py` — post-exit counterfactual tracking (regret analysis)
- **GUI Bridge**: `live/gui_bridge.py` — non-blocking queue wrapper for Tk dashboard
- **Session Tracker**: `live/session_tracker.py` — session PnL, drawdowns, trade log, reports
- **Ping Pong**: `live/ping_pong.py` — flip direction, ATR sizing, deferred flip management
- **Worker**: `training/orchestrator_worker.py` — per-TF fractal worker
- **Dashboard**: `visualization/dashboard.py` — Tkinter "Fractal Command Center" (1600x950)
- **Clustering**: `core/fractal_clustering.py` — Main: recursive K-Means | Experimental: DMI→I-MR→DBSCAN
- **Feature vector**: 16D — abs(z), log1p(v), log1p(m), coherence, tf_scale, depth,
  parent_ctx, self_adx, self_hurst, self_dmi_diff, parent_z, parent_dmi_diff,
  root_is_roche, tf_alignment, self_pid, osc_coh

## Report & Output Locations
> Source of truth: `_get_reports_dir(mode)` in `training/trainer.py`
> Pattern: `reports/{mode}/` where mode = is | oos | phase5 | training

| Path | Contents | Notes |
|------|----------|-------|
| `reports/is/` | IS forward pass: oracle_trade_log.csv, signal_log.csv + `shards/` | CSVs gitignored |
| `reports/oos/` | OOS forward pass: same structure | CSVs gitignored |
| `reports/phase5/` | phase5_report.txt (strategy selection) | .txt tracked |
| `reports/training/` | training_log.txt | Gitignored |
| `run_logs/` | Monthly sharded CSVs | Gitignored |
| `docs/checkpoint_reference/` | SCHEMAS.md, run_snapshot.json, samples | Tracked |

**To read latest results**: `reports/is_report.txt` / `reports/oos_report.txt` (summaries)

### Post-Run Reports to Review
After every forward pass, always read these reports:
1. `reports/is_report.txt` — IS summary (WR, PnL, depth breakdown, exit quality, direction, gates)
2. `reports/oos_report.txt` — OOS summary (same structure, validates IS findings)
3. `checkpoints/oos_analytics.txt` — OOS analytics: t-tests, ANOVA, OLS, logistic, capture rate
4. `checkpoints/trade_analytics.txt` — IS version of analytics suite

## CLI Flags — entry point: `python training/trainer.py`
- `--fresh` — wipe ALL checkpoints + full pipeline
- `--train-only` — Phases 2-3 only
- `--forward-pass` — IS → OOS → Strategy auto-chain (existing library)
- `--forward-pass --skip-oos` — IS only
- `--oos` — standalone OOS rerun (uses DATA/ATLAS_OOS)
- `--forward-data PATH` — custom data for forward pass
- `--data DATA/ATLAS_1DAY` — single-day fast validation (~3s)
- `--strategy-report` — Phase 6 (strategy selection) only
- `--ping-pong` — continuous wave-riding (flip after exit, belief conviction gate)
- `--pp-conviction 0.55` / `--pp-sl` / `--pp-tp` / `--pp-trail` — PP overrides

## Data Pipeline
- ATLAS: `DATA/ATLAS/{tf}/YYYY_MM.parquet` — 14 TFs, 12 months (Jan-Dec 2025)
- ATLAS_1DAY: `DATA/ATLAS_1DAY/` — single day (Jan 2) for fast validation
- ATLAS_1WEEK: `DATA/ATLAS_1WEEK/` — 7 trading days (Jan 2-10) for screening
- ATLAS_OOS: `DATA/ATLAS_OOS/` — 2 months (Jan-Feb 2026)

## Analysis & Benchmark Tools
- `tools/analyze_gates.py` — oracle-driven gate threshold analysis, `--apply` writes JSON
- `tools/gate_interaction_matrix.py` — C&E matrix empirical validation (Spearman/Kruskal)
- `tools/golden_path.py` — Y10/Y11/Y12 chord length metrics from 1s data
- `tools/pattern_map.py` — signal funnel visualization
- `tools/trade_visualizer.py` — trade overlay on price waveform
- `tools/run_analytics.py` — re-run analytics without forward pass
- `tools/nt8_to_parquet.py` — convert NT8 exports → ATLAS parquet
- `tools/checkpoint_viewer.py` — inspect pattern_library + brain
- `tools/setup_oos_atlas.py` — train/test split of ATLAS data
- `tools/standalone_research.py` — research harness (A-R modules), `--start X` to skip.
  See `tools/STANDALONE_RESEARCH_GUIDE.md` for how to add new modules.
- `tools/research/` — subpackage: data.py, imr.py, screening.py, seeds.py, plots.py, tbn_trade_aware.py
- `tools/research_belief_flip.py` — side-by-side TBN comparison
- `tools/archive/` — one-off analysis scripts (analyze_scalps, analyze_wrong_dir, etc.)

## Waveform Screening & Seed Library
> Full details: `memory/waveform_research.md`
- Offline research producing pre-built SEED LIBRARY for live matching
- 20 shape primitives (V-reversal, ramp, sigmoid, etc.) replace DBSCAN clusters
- Spec: `docs/JULES_WAVEFORM_SEED_INTEGRATION.md` (5 parts, not yet started)

## Validation Ladder (4 gates, sequential)
1. **IS**: `--forward-pass` on ATLAS. Failure = library broken.
2. **OOS**: Auto-OOS on ATLAS_OOS. Failure = overfit.
3. **Live Simulated**: Paper trading via live/ module. Failure = engineering.
4. **Live Real**: Real money via NT8_BayesianBridge. Failure = risk management.

## Current Baseline (IS+OOS, 2026-03-07, main branch)
- IS: 7,262 trades, 85.7% WR, $86,351 total, $11.89/trade, PF 3.54
- OOS: 536 trades, 88.4% WR, $10,804 total, $20.16/trade (~$5.4K/month)
- Direction: 60.4% correct OOS (was 43.1%), taking both LONG and SHORT
- Envelope_decay is primary exit: 91% of exits, $33-52/trade avg
- Key fixes (2026-03-07): Gate 4 momentum alignment, trail/maxhold/watchdog disabled
- Prior baseline (2026-02-25): 3,754 trades, 37.5% WR, $1.55/trade

## Updated Baseline (2026-03-08, oracle removed, pipeline restructured)
- IS: $86,685 PnL, 7704 trades, 78.8% WR, 52.8% correct direction
- OOS: $22,383 PnL, 1750 trades, 77.5% WR, 57.0% correct direction
- Max DD: $395, 0 consecutive losing days
- Key change: IS oracle direction override removed (was contaminating)

## Implemented Features (confirmed in codebase)
- **Direction fix** (2026-03-06): momentum-aware physics (velocity+acceleration) in TBN worker
- **Gate 4 momentum alignment** (2026-03-06): skip sign(F_momentum) vs direction mismatches
- **Trail/MaxHold/Watchdog disabled** (2026-03-06): envelope_decay handles all exits better
- **Unified exit engine** (2026-03-06): SL→TP→BandUrgent→EnvelopeDecay→PeakGiveback→BreakevenLock→BeliefFlip→Hold
- **Self-tuning exits** (2026-03-06): too_early→grow halflife, too_late→tighten giveback
- **Dynamic halflife** (2026-03-06): giveback_ratio modulates effective_hl (base=20, range 8-60)
- **Band confluence** (2026-03-06): BandContext per TF worker + direction cascade Priority 4
- **Auto-TP re-entry** (2026-03-06): live_engine.py — bank profit, re-enter if belief agrees
- **Trade logger** (2026-03-06): live/trade_logger.py — per-trade diagnostic CSV
- **Oracle-driven gates** (2026-03-06): execution_engine.py loads from gate_thresholds.json
- Spectral gates: Fourier half-cycle + Laplace kinetic damping (orchestrator_worker.py)
- Template timescale: avg_mfe_bar/p75_mfe_bar time-exhaustion exits
- Price-aware workers: trade_side + profit_ticks modulate conviction
- Live trading module: live/ (11 files) + docs/NT8_BayesianBridge.cs
- **(2026-03-08) ExecutionEngine integration**: live delegates to EE thin wrapper
- **(2026-03-08) Compressed history replay**: `live/history_replay.py` + `live/atlas_loader.py`
- **(2026-03-08) Feature extraction unified**: `core/feature_extraction.py`
- **(2026-03-08) CPU path removed** (CUDA-only)
- **(2026-03-08) Pipeline restructured** to 6 phases (Strategy moved after OOS)
- **(2026-03-08) IS oracle direction override removed**
- **(2026-03-08) Expected profit tracking**: `get_expected_pnl()` in brain
- **(2026-03-08) Run history CSV**: `reports/run_history.csv`
- **(2026-03-08) ExitEngine simplified**: `open_position()` accepts pre-computed sizing
- **(2026-03-08) PositionState trimmed**: 7 dead fields deleted
- **(2026-03-08) LiveEngine decomposition**: exit_watcher, gui_bridge, session_tracker, ping_pong

## C&E Matrix Methodology (KEY WORKFLOW)
> Full methodology: `memory/ce_methodology.md`
- Identify problem → C&E t-test → Simulate → Fix → Add analytics bucket → Verify

## Design Docs & Specs
- `docs/active/` — currently being worked on (REMAINING_WORK, CONSOLIDATION)
- `docs/specs/` — future features (LEVEL_DETECTOR, EXPECTED_PROFIT, WAVEFORM_SEED)
- `docs/archive/` — completed specs (EXIT_ENGINE, BAND_CONTEXT, METAPHOR_PURGE, etc.)
- `docs/reference/` — journals (C&E Matrix, Pipeline Analysis, Waveform Analysis)

## Branches
- `main` — sole branch, all fixes consolidated
- Killed: `pre-snowflake` (deleted 2026-03-07), `unified-cluster`, `jules/fractal-trend-*`

## Requirements
- PyTorch CUDA (cu121), numba, scipy, optuna>=3.5.0
- databento + databento-dbn for data loading
- CUDA required — CPU fallback removed (2026-03-08)
