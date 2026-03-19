# Bayesian-AI Project Memory
> Future topics backlog: see `docs/ROADMAP.md`
> Daily interaction journals: `docs/daily/YYYY-MM-DD.md`

## **HARD RULES — DO NOT SKIP**
- **KEEP JOURNALS UPDATED**: At the start and end of EVERY session, update:
  1. `docs/daily/YYYY-MM-DD.md` — daily interaction journal (findings, changes, decisions)
  2. `docs/daily/INDEX.md` — add one-line summary at top of table (newest first)
  3. `docs/reference/RESEARCH_JOURNAL.txt` — research journal (analysis results, pipeline, new modules)
  4. `reports/findings/` — standalone finding reports (YYYY-MM-DD_topic.md)
  NEVER let journals go stale. Lost journals = lost days of work.
  **Start of session**: read `docs/daily/INDEX.md` for context (not full journals).
- **KEEP MEMORY UPDATED**: Update MEMORY.md when discovering new patterns, preferences, or architecture changes

## Workflow Preference
- **All test/research analyses must mirror live trading conditions** — no lookahead, use only
  data available at decision time. Slow TF bars incomplete mid-bar = use last completed bar.
- **Always discuss before changing**: propose a plan, get approval, then execute
- **Challenge ideas HARD**: Push back aggressively, don't be complacent. See `memory/feedback_challenge_ideas.md` + `memory/feedback_challenge_harder.md`
- **Live launcher defaults**: Default = send orders to NT8. NT8 account (sim/real) controls risk.
  `--dry-run` = opt-in observation only. See `memory/feedback_live_defaults.md`
- **Progress bars are mandatory**: any long-running loop MUST have tqdm with live stats
- **Training via Bash**: show exact command, ask "Confirm to run?" — only execute after user confirms
- **NT8 bridge deploy**: When updating `docs/NT8_BayesianBridge.cs`, also copy it to
  `C:\Users\reyse\OneDrive\Documents\NinjaTrader 8\bin\Custom\Indicators\NT8_BayesianBridge.cs`
- **NT8 bridge versioning**: Always bump version + update date + time in both header comment
  and `BRIDGE_VERSION` const (e.g. `6.1.2 — 2026-03-04 06:15`)

## **AUTO SEEDS + QUANTUM — NEXT PRIORITIES (2026-03-19)**
- **READ FIRST**: `memory/project_auto_seeds_next.md` — 31K seeds as template library
- **THEN**: Quantum reconnect as feature enrichment on seed context
- ADX chop filter: `memory/project_adx_chop_filter.md` — biggest OOS improvement
- Peak override FAILED: `memory/feedback_peak_override_failed.md` — do NOT re-enable
- **Commit flow**: code first (commit+push), then reports/CSVs separately (commit+push)

## **QUANTUM RECONNECTION (deprioritized, was #1 on 2026-03-15)**
- **READ FIRST**: `memory/project_quantum_reconnect.md` — full context + action items
- System is probabilistic by design (quantum model). Gates broke it. Reconnect wave function to scoring.
- Wave function probs (prob0/1/2), tunnel_prob, coherence computed EVERY BAR but unused in scoring
- Reference docs: `docs/reference/quantum_field_engine_original.py`, `QUANTUM_STATE_REFERENCE.md`, `NIGHTMARE_PROTOCOL.md`, `THREE_THEORIES.md`
- Decision funnel validated: 85% accuracy predicting regime starts (funnel tree research)
- PFMEA: RPN 648 (score competition), 392 (conviction position), 336 (giveback flip)
- Fresh run needed: `python training/trainer.py --fresh --lookback`

## Architecture
- **Core engine**: `core/statistical_field_engine.py` — MarketState per bar (CUDA-only since 2026-03-08)
- **Brain**: `core/bayesian_brain.py` — Bayesian table with hash-based state lookups
- **Trainer**: `training/trainer.py` — main entry point, CLI, 7-phase pipeline
  - Run: `python training/trainer.py --fresh`
- **Exit Engine**: `core/exit_engine.py` — unified exit cascade (SL→TP→BandUrgent→
  EnvelopeDecay→BreakevenLock→BeliefFlip→Hold). Trail/MaxHold/Watchdog DISABLED.
- **Execution Engine**: `core/execution_engine.py` — gate/direction/sizing, oracle-driven thresholds
- **Feature Extraction**: `core/feature_extraction.py` — canonical 16D feature vector (single source of truth)
- **Belief Network**: `core/timeframe_belief_network.py` — 11 TF workers (incl 5s, 1s),
  BandContext per worker, `get_band_confluence()` (Priority 4 in direction cascade)
- **Position factory**: `ExitEngine.open_position()` — creates PositionState (make_position deleted 2026-03-09).
  wave_rider.py DELETED (2026-03-07). All position/exit logic in exit_engine.py.
- **Trade recording**: `record_trade()` in `core/bayesian_brain.py` — shared by trainer + live.
  Constructs TradeOutcome + brain.update() + brain.direction_learn(). Single code path.
- **Trade Logger**: `live/trade_logger.py` — per-trade diagnostic CSV
- **History Replay**: `live/history_replay.py` — compressed forward pass + parity report (Phase 7)
- **Atlas Loader**: `live/atlas_loader.py` — ATLAS parquet reader for date ranges
- **Exit Watcher**: `live/exit_watcher.py` — post-exit counterfactual tracking (regret analysis)
- **GUI Bridge**: `live/gui_bridge.py` — non-blocking queue wrapper for Tk dashboard
- **Session Tracker**: `live/session_tracker.py` — session PnL, drawdowns, trade log, reports
- **Ping Pong**: `live/ping_pong.py` — flip direction, ATR sizing, deferred flip management
- **Worker**: `training/orchestrator_worker.py` — per-TF fractal worker
- **Dashboard**: `visualization/dashboard.py` — Tkinter "Fractal Command Center" (1600x950)
- **Clustering**: `core/fractal_clustering.py` — TF-bucketed recursive K-Means (patterns binned by TF before clustering)
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
- `--fresh` — wipe ALL checkpoints + full pipeline (Phases 1-7 including live replay)
- `--train-only` — Phases 2-3 only
- `--forward-pass` — IS → OOS → Strategy → Phase 7 auto-chain (existing library)
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

## Timeframe Rules
- **ATLAS has 14 TFs**: 1s, 5s, 15s, 30s, 1m, 3m, 5m, 15m, 30m, 1h — use the right one for the job
- **Oracle/template stats**: computed from 1m (discovery TF) — NOT 15s
- **Forward pass iteration**: 15s (execution TF) — only this is 15s
- **Analysis/tools**: match TF to the question (session overlay → 1h, trade anatomy → 1m or 5m)
- **4x mismatch bug (2026-03-11)**: oracle bar counts are 1m bars, consumed as 15s → 4x error everywhere

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
- `tools/dmi_crossover_validation.py` — DMI crossover accuracy on IS/OOS data
- `tools/equity_risk_simulator.py` — equity growth simulation (flat vs dynamic sizing)
- `tools/l2_risk_budget.py` — L2 risk budget (MAE/MFE of $30+ segments from 1s data)
- `tools/imr_golden_path.py` — I-MR control chart + golden path overlay
- `tools/archive/` — one-off analysis scripts (analyze_scalps, analyze_wrong_dir, etc.)

## Two-Stage Shape Primitives (2026-03-14)
- **Entry primitives**: 10-bar lookback geometry (6D) + 192D context -> UMAP+HDBSCAN per-TF
- **Exit primitives**: 32-point segment shape (34D) -> UMAP+HDBSCAN per-TF + exit calibration
- Builder: `tools/shape_primitive_builder.py --entry/--exit/--all`
- Data structures: `core/shape_primitives.py`
- Exit integration: giveback gets shape-aware thresholds, envelope gets halflife modulation
- Research line: telescoping TF (1m entry, 15s confirm, 5s/1s ticker) — see `memory/research_telescoping_tf.md`

## Validation Ladder (5 gates, sequential)
1. **IS**: Full discovery on ATLAS. Failure = library broken.
2. **OOS**: Compressed per-bar on ATLAS_OOS. Failure = overfit.
3. **Phase 7 Replay**: `--replay-only` on ATLAS (parity report). Failure = live stack broken.
4. **Live Simulated**: Paper trading via live/ module. Failure = engineering.
5. **Live Real**: Real money via NT8_BayesianBridge. Failure = risk management.

## Current Baseline (2026-03-12, V7.0.0 TF binning — main branch)
- IS: $39,736 PnL, 3,298 trades, 96.6% WR, $12.05/trade
- OOS (compressed): $8,200 PnL, 643 trades, 100% WR, $12.75/trade (all depth 8)
- Phase 7 Replay: $88.50 PnL, 120 trades, 91.7% WR, $0.74/trade — **PARITY FAILED**
- Replay issues: 82 SL exits (churn), peak_giveback 11.1% WR, 97% SHORT direction
- OOS oracle metadata blank (compressed path, no discovery = no oracle classification)

## Prior Baseline (2026-03-09, recursive hierarchy — main branch)
- IS: $83,821 PnL, 7,773 trades, 97.9% WR, $10.78/trade, 53.1% correct direction
- OOS: $21,378 PnL, 1,881 trades, 98.5% WR, $11.37/trade, 56.2% correct direction

## R2 Baseline (2026-03-11, 1m-anchor + exit tuning — exp/1m-anchor)
- IS: $43,272 PnL, 5,227 trades, 95.6% WR, $8.28/trade, 43.6% correct direction
- OOS: $7,901 PnL, 855 trades, 95.2% WR, $9.24/trade, 44.4% correct direction
- IS Max DD: $99.50, OOS Max DD: $76.00
- Giveback primary exit: 2,131 trades IS, avg 1.7 min hold, 91% gave back, $13.70/trade
- Trade health WIN vs LOSS: 0.711 vs 0.421 (IS), 0.738 vs 0.418 (OOS) — clear separation
- Pace WIN vs LOSS: 0.867 vs -1.351 (IS), 0.883 vs -1.857 (OOS) — strong signal
- ~50% of main branch PnL (expected: single-TF patterns vs recursive hierarchy)

## Prior Baselines
- 2026-03-08: $86,685 IS / $22,383 OOS, 78.8%/77.5% WR, oracle removed
- 2026-03-07: $86,351 IS / $10,804 OOS, 85.7%/88.4% WR, gate 4 momentum
- 2026-02-25: $5,818 IS, 3,754 trades, 37.5% WR, $1.55/trade

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
- **(2026-03-11) Exit tuning**: giveback 10% primary exit, envelope halflife 40 (lazy safety net)
- **(2026-03-11) Trade-aware workers**: TBN orchestrator tracks pace (tick/time progress), blends into direction
- **(2026-03-11) Trade health fusion**: `0.6*pace + 0.4*(1-decay)` — single score replacing dead decay_score
- **(2026-03-11) Report improvements**: hold time in minutes, GIVEBACK & DECAY CONTEXT table, health/pace WIN vs LOSS
- **(2026-03-11) Magic number audit**: 203 numbers identified, spec at `docs/JULES_MAGIC_NUMBER_REFACTOR.md`
- **(2026-03-12) TF-bucketed clustering**: patterns binned by TF before K-Means (no scale mixing)
- **(2026-03-12) OOS compressed per-bar**: OOS uses same path as live (no discovery, no ancestry)
- **(2026-03-12) Phase 7 live replay**: training calls actual `live.launcher --replay-only` for integrity test
- **(2026-03-12) Parity report**: `reports/live/parity_report_*.txt` — OOS vs Replay comparison
- **(2026-03-12) Live startup simplified**: no replay warmup in normal live, connects straight to NT8

## C&E Matrix Methodology (KEY WORKFLOW)
> Full methodology: `memory/ce_methodology.md`
- Identify problem → C&E t-test → Simulate → Fix → Add analytics bucket → Verify

## Research Lines
- **R1**: TF-Bucketed Clustering (main) — DONE (2026-03-12), 403 templates from 13 TF buckets
- **R2**: 1-Minute Anchor Discovery (exp/1m-anchor) — see `docs/ROADMAP.md`
- **R3**: Pre-Entry Pace Filter — see `memory/research_pre_entry_pace.md`
- **R4**: Regime Trading Framework (THE ANCHOR) — see `docs/REGIME_TRADING_SPEC.md`
  - DMI crossover entry + 192D noise filter + equity-scaled sizing
  - $10 risk → 94.9% WR, $22 avg reward, 1:2.2 R:R, $20.35 EV/trade
  - Tools: `dmi_crossover_validation.py`, `equity_risk_simulator.py`, `l2_risk_budget.py`
  - Replaces scalping ($1.67/trade) with regime riding ($20+/trade)
- **Level Detector**: see `docs/specs/LEVEL_DETECTOR` + `memory/user_level_trading.md`
- **Headroom / Nesting Gate**: see `memory/user_headroom_framework.md`

## Design Docs & Specs
- `docs/active/` — currently being worked on (REMAINING_WORK, CONSOLIDATION)
- `docs/specs/` — future features (LEVEL_DETECTOR, EXPECTED_PROFIT, WAVEFORM_SEED)
- `docs/archive/` — completed specs (EXIT_ENGINE, BAND_CONTEXT, METAPHOR_PURGE, etc.)
- `docs/reference/` — journals (C&E Matrix, Pipeline Analysis, Waveform Analysis)

## Branches
- `main` — recursive hierarchy baseline (commit `33e4890`)
- `exp/1m-anchor` — 1m anchor experiment (created at `33e4890`, changes pending commit)
- Killed: `pre-snowflake` (deleted 2026-03-07), `unified-cluster`, `jules/fractal-trend-*`

## Requirements
- PyTorch CUDA (cu121), numba, scipy, optuna>=3.5.0
- databento + databento-dbn for data loading
- CUDA required — CPU fallback removed (2026-03-08)
