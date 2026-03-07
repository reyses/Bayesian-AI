# Bayesian-AI Project Memory
> Future topics backlog: see `docs/ROADMAP.md`
> Daily interaction journals: `docs/daily/YYYY-MM-DD.md`

## **HARD RULES — DO NOT SKIP**
- **KEEP JOURNALS UPDATED**: At the start and end of EVERY session, update:
  1. `docs/daily/YYYY-MM-DD.md` — daily interaction journal (findings, changes, decisions)
  2. `docs/WAVEFORM_ANALYSIS_JOURNAL.txt` — research journal (analysis results, new modules)
  3. `docs/PIPELINE_ANALYSIS_JOURNAL.txt` — pipeline changes journal
  4. `reports/findings/` — standalone finding reports (YYYY-MM-DD_topic.md)
  NEVER let journals go stale. Lost journals = lost days of work.
- **KEEP MEMORY UPDATED**: Update MEMORY.md when discovering new patterns, preferences, or architecture changes

## Workflow Preference
- **Always discuss before changing**: propose a plan, get approval, then execute
- **Progress bars are mandatory**: any long-running loop MUST have tqdm with live stats
- **Training via Bash**: show exact command, ask "Confirm to run?" — only execute after user confirms
- **NT8 bridge deploy**: When updating `docs/NT8_BayesianBridge.cs`, also copy it to
  `C:\Users\reyse\OneDrive\Documents\NinjaTrader 8\bin\Custom\Indicators\NT8_BayesianBridge.cs`
- **NT8 bridge versioning**: Always bump version + update date + time in both header comment
  and `BRIDGE_VERSION` const (e.g. `6.1.2 — 2026-03-04 06:15`)

## Architecture
- **Core engine**: `core/quantum_field_engine.py` — ThreeBodyQuantumState per bar
- **Brain**: `core/bayesian_brain.py` — Bayesian table with hash-based state lookups
- **Trainer**: `training/trainer.py` — main entry point, CLI, forward pass
  - Run: `python training/trainer.py --fresh --forward-pass`
  - NOT `python -m training.orchestrator` (old name, no longer exists)
- **Exit Engine**: `core/exit_engine.py` — unified exit cascade (SL→TP→BandUrgent→
  EnvelopeDecay→BreakevenLock→BeliefFlip→Hold). Trail/MaxHold/Watchdog DISABLED.
- **Execution Engine**: `core/execution_engine.py` — gate/direction/sizing, oracle-driven thresholds
- **Belief Network**: `core/timeframe_belief_network.py` — 10 TF workers, BandContext per worker,
  `get_band_confluence()` for multi-TF SE band direction (Priority 4 in direction cascade)
- **Position factory**: `make_position()` in `core/exit_engine.py` — creates PositionState directly.
  wave_rider.py DELETED (2026-03-07). All position/exit logic in exit_engine.py.
- **Trade Logger**: `live/trade_logger.py` — per-trade diagnostic CSV
- **Dashboard**: `visualization/dashboard.py` — Tkinter popup "Fractal Command Center".
  Single-instance, 1600x950 window. Receives data via queue from trainer/live engine.
  Live mode: price ticker (green/red flash per tick), rolling price chart with trade
  markers, PnL tracking, net liquidity display, exit belief bar.
  Training mode: Pareto chart of loss categories (Missed/Wrong Dir/Too Early/Noise),
  template manifold (z vs momentum scatter), template leaderboard, phase progress,
  fission events.
- **Clustering**: `core/fractal_clustering.py` — DMI pre-split → I-MR(DMI diff) → DBSCAN(vol+ADX)
- **DNA Tree**: `training/fractal_dna_tree.py` — hierarchical TF context tree
- **Feature vector**: 16D — abs(z), log1p(v), log1p(m), coherence, tf_scale, depth,
  parent_ctx, self_adx, self_hurst, self_dmi_diff, parent_z, parent_dmi_diff,
  root_is_roche, tf_alignment, self_pid, osc_coh

## Report & Output Locations
> Source of truth: `_get_reports_dir(mode)` in `training/trainer.py` line ~43
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

### Post-Run Reports to Review
After every forward pass, always read these reports to understand the run:
1. `reports/is_report.txt` — IS summary (WR, PnL, depth breakdown, exit quality, direction, gates)
2. `reports/oos_report.txt` — OOS summary (same structure, validates IS findings)
3. `checkpoints/oos_analytics.txt` — OOS trade analytics suite:
   - Part 2: t-tests (winners vs losers feature comparison)
   - Part 3: ANOVA (PnL by session, day, depth, direction, exit reason)
   - Part 4: OLS regression (what predicts PnL magnitude)
   - Part 5: Logistic regression (what predicts win/loss)
   - Part 6: Capture rate regression (what predicts MFE extraction)
   - Part 7: Session × Direction cross-tab + hourly breakdown
4. `checkpoints/trade_analytics.txt` — IS version of the analytics suite

## CLI Flags (current) — entry point: `python training/trainer.py`
- `--fresh` — wipe ALL checkpoints + full pipeline
- `--train-only` — Phases 2-3 only
- `--forward-pass` — IS → Strategy → OOS auto-chain (existing library)
- `--forward-pass --skip-oos` — IS → Strategy only
- `--oos` — standalone OOS rerun (uses DATA/ATLAS_OOS)
- `--forward-data PATH` — custom data for forward pass (skips auto-OOS)
- `--data DATA/ATLAS_1DAY` — single-day fast validation (~3s)
- `--strategy-report` — Phase 5 only
- `--ping-pong` — continuous wave-riding (flip after exit, belief conviction gate)
- `--pp-conviction 0.55` / `--pp-sl` / `--pp-tp` / `--pp-trail` — PP overrides
- When `--ping-pong` active, outputs go to `checkpoints/snowflake/` + `reports/snowflake/`

## Data Pipeline
- ATLAS: `DATA/ATLAS/{tf}/YYYY_MM.parquet` — 14 TFs, 12 months (Jan-Dec 2025)
- ATLAS 1s: `DATA/ATLAS/1s/YYYY_MM.parquet` — 12 files, used by golden_path.py
- ATLAS_1DAY: `DATA/ATLAS_1DAY/` — single day (Jan 2) for fast validation
- ATLAS_1WEEK: `DATA/ATLAS_1WEEK/` — 7 trading days (Jan 2-10) for screening
- ATLAS_OOS: `DATA/ATLAS_OOS/` — 2 months (Jan-Feb 2026)
- Raw 1s: `DATA/glbx-mdp3-20250101-20260209.ohlcv-1s.parquet` (13.2M bars)

## Analysis & Benchmark Tools
- `tools/run_benchmark.py` — chains IS+OOS, snapshots to reports/benchmarks/, --history
- `tools/compare_oos_runs.py` — side-by-side comparison of two runs
- `tools/analyze_exits.py` — deep exit analysis for a single run
- `tools/analyze_gates.py` — oracle-driven gate threshold analysis, `--apply` writes JSON
- `tools/gate_interaction_matrix.py` — C&E matrix empirical validation (Spearman/Kruskal)
- `tools/golden_path.py` — Y10/Y11/Y12 chord length metrics from 1s data
- `tools/pattern_map.py` — signal funnel visualization
- `tools/trade_visualizer.py` — trade overlay on price waveform
- `tools/run_analytics.py` — re-run analytics without forward pass
- `tools/analyze_scalps.py` — worker signal analysis on counter-trend scalps
- `tools/analyze_scalp_timing.py` — temporal clustering, MFE, physics on scalps
- `tools/analyze_scalp_vs_early_exit.py` — too-early/scalp hourly overlap (r=0.716)
- `tools/analyze_wrong_dir.py` — wrong-direction filter simulations
- `scripts/monthly_pnl_chart.py` — monthly PnL bar chart
- `tools/make_atlas_1day.py` — create 1-day ATLAS subset
- `tools/make_atlas_1week.py` — create 7-day ATLAS subset
- `tools/standalone_research.py` — research harness: lettered analyses A-R, `--start X` to skip.
  See `tools/STANDALONE_RESEARCH_GUIDE.md` for how to add new modules.

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
- **Integration spec**: `docs/JULES_WAVEFORM_SEED_INTEGRATION.md` (5 parts)
  Part 1: seed_library.py (20 shapes), Part 2: 4h worker, Part 3: shape direction P0.5,
  Part 4: GradientBoosting 176D direction model, Part 5: live engine wiring

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
  `docs/JULES_WAVEFORM_SEED_INTEGRATION.md` (5 parts: seed library extraction,
  4h worker, shape classification, direction model training, live integration)
- **Archived**: `docs/old_jules/` — SPECTRAL_GATES, SNOWFLAKE_BASELINE,
  TASK_1_2, TEMPLATE_TIMESCALE, PLAN_PRICE_AWARE_WORKERS, SIGNAL_CAPTURE_AUDIT

## Branches
- `main` — unified clustering, three-body exits, pipeline gauntlet, all fixes
- `pre-snowflake` — **active live trading branch** (best candidate as of 2026-03-02)
  - OOS: 357 trades, 52.4% WR, $1,724.50 — better than main
  - Has live module ported from main + physics quality gate
  - Ping-pong mode: implemented (config overrides, snowflake output dirs)
  - PP OOS: 535 trades, 52.5% WR, $1,644.50, 56 flips at 51.8% WR (-$195 PP drag)
  - Direction improved: 50.3% correct (was 43.1%), now taking LONG trades (98)
  - Session reports: `reports/live/session_*.txt`
- Killed: `unified-cluster`, `jules/fractal-trend-*` (deleted 2026-02-27)

## Implemented Features (confirmed in codebase, 2026-03-06)
- **Direction fix**: momentum-aware physics (velocity+acceleration, not mean-reverting z)
  in TBN worker (core/timeframe_belief_network.py:251-263)
- **Gate 4 momentum alignment**: execution_engine.py — skip trades where sign(F_momentum)
  disagrees with trade direction. Eliminated 55% misaligned trades (+$7,815 recovery).
- **Trail/MaxHold/Watchdog disabled**: exit_engine.py — envelope_decay handles all exits
  better. Trail was $3-4/trade (84% too early), watchdog 0% WR, max_hold redundant.
- **Unified exit engine**: core/exit_engine.py — cascade: SL→TP→BandUrgent→EnvelopeDecay→
  PeakGiveback→BreakevenLock→BeliefFlip→Hold. Self-tuning halflife + giveback.
- **Self-tuning exits**: `record_trade_outcome()` in ExitEngine — two independent signals:
  too_early→grow halflife, too_late→tighten giveback. 30-trade calibration window.
- **Dynamic halflife**: envelope halflife shrinks when trade is giving back from peak
  (giveback_ratio modulates effective_hl). Base=20 bars, range 8-60.
- **Band confluence**: BandContext per TF worker + get_band_confluence() aggregator.
  Wired into direction cascade (Priority 4) AND exit trail adjustment
- **Auto-TP re-entry**: live_engine.py:832 — bank profit, re-enter if belief agrees
- **Trade logger**: live/trade_logger.py — per-trade diagnostic CSV
- **Oracle-driven gates**: execution_engine.py loads thresholds from gate_thresholds.json
- **Physics fields in oracle records**: trainer.py _physics_fields() — F_momentum,
  F_reversion, mom_rev_ratio, hurst, tunnel_prob, velocity, sigma, band_speed
- Spectral gates: Fourier half-cycle + Laplace kinetic damping (orchestrator_worker.py)
- Template timescale: avg_mfe_bar/p75_mfe_bar time-exhaustion exits
- Price-aware workers: trade_side + profit_ticks modulate conviction
- Decay cascade: z-score drift from expected trajectory (belief_network)
- OU tunnel probabilities: analytical erfi-based (quantum_field_engine)
- Semantic cluster names: generate_semantic_name() (fractal_clustering)
- Live trading module: live/ (7 files) + docs/NT8_BayesianBridge.cs

## C&E Matrix Methodology (KEY WORKFLOW)
> Full methodology: `memory/ce_methodology.md`

When optimizing exits (or entries), follow the **Cause & Effect matrix** approach:
1. **Identify** problem from report buckets (too-early, too-late, reversed)
2. **C&E t-test**: split trade log into problem vs good, Welch t-test all features
3. **Simulate** the exit mechanic at each bar to find the smoking gun
4. **Fix** with targeted parameter change + self-tuning feedback loop
5. **Add analytics bucket** to report so future runs track the split
6. **Verify** with `--forward-pass` — compare before/after metrics

## Design Docs
- `docs/CAUSE_AND_EFFECT_MATRIX.md` — 48 X params × 12 Y responses, entry/exit domain
  split, sample size analysis, golden path hierarchy, parameter role assignment
- `docs/CLAUDE_CODE_BAND_CONTEXT.md` — SE band confluence spec (IMPLEMENTED)
- `docs/CLAUDE_CODE_UNIFIED_EXIT_ENGINE.md` — exit engine spec (IMPLEMENTED)
- `docs/LEVEL_DETECTOR_SPEC.md` — fib + swing detection + DBSCAN levels (FUTURE STATE)
- `reports/findings/` — pure research findings (no specs/instructions), date-prefixed
  - `2026-03-07_scalp_timescale.md` — scalp/too-early overlap (r=0.716), timescale mismatch

## Current Baseline (IS+OOS, 2026-03-07, main branch)
- IS: 7,262 trades, 85.7% WR, $86,351 total, $11.89/trade, PF 3.54
- OOS: 536 trades, 88.4% WR, $10,804 total, $20.16/trade (~$5.4K/month)
- Direction: 60.4% correct OOS (was 43.1%), taking both LONG and SHORT
- Envelope_decay is primary exit: 91% of exits, $33-52/trade avg
- Depth 5-7 sweet spot: 94-100% WR, $35-44/trade OOS
- Key fixes (2026-03-07): Gate 4 momentum alignment, trail/maxhold/watchdog disabled
- Best filter: depth<=3 + z<0 → 69.2% WR, $188.83/trade OOS (26 trades)
- Key insight: physics + exit system carry all profit; trail stop = mean reversion catcher
- Prior baseline (2026-02-25): 3,754 trades, 37.5% WR, $1.55/trade

## Clustering & Design Direction
- **Pipeline**: DMI pre-split → I-MR(DMI diff) → DBSCAN(vol+ADX), 2D clustering, 16D identity
- **Shape-first**: seed functions (ramp, V, sigmoid) replace DBSCAN clusters
- **Laplacian = shape identifier**: d²p/dt² discriminates shape types

## Requirements
- PyTorch CUDA (cu121), numba, scipy, optuna>=3.5.0
- databento + databento-dbn for data loading
