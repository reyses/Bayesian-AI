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
- **Data validation FIRST**: Run `tools/validate_data.py` before ANY training or analysis. See `memory/feedback_data_validation_first.md`
- **No lookahead**: all analyses must mirror live conditions. Use only data available at decision time.
- **Always discuss before changing**: propose a plan, get approval, then execute
- **Challenge ideas HARD**: See `memory/feedback_challenge_ideas.md` + `memory/feedback_challenge_harder.md`
- **Don't be sloppy**: See `memory/feedback_sloppy_work.md` + `memory/feedback_cnn_fragility.md`
- **Live launcher defaults**: Default = send orders to NT8. `--dry-run` = opt-in. See `memory/feedback_live_defaults.md`
- **Progress bars mandatory**: tqdm with live stats for any loop > 100 iterations
- **Training via Bash**: show exact command, ask "Confirm to run?" — only execute after user confirms
- **NT8 bridge deploy**: copy to both `docs/` and NT8 indicators dir. Always bump version + timestamp.
- **Track time**: Periodically check the clock during long sessions. Flag when it's late.
- **Commit flow**: code first (commit+push), then reports/CSVs separately (commit+push)
- **Base measurements grounded**: See `memory/feedback_base_measurements.md`
- **Checkpoint every step**: All multi-step pipelines must save to disk after each step. See `memory/feedback_checkpoint_every_step.md`

## **CURRENT PRIORITIES (2026-03-27)**
- **MTF Two-Layer Counter-Proposal APPROVED**: See `memory/project_mtf_counter_proposal.md`
  - 29D features (13D base + 16D MTF from 1s/5m/15m/1h)
  - Two-layer CNN: Direction (Layer 1) + Duration (Layer 2)
  - Spec: `docs/Active/COUNTER_PROPOSAL_MTF_TWO_LAYER.md`
  - Build phases: A (features) → B (L1 train) → C (L2 train) → D (simulation)
- **TradeCNN baseline: $1,609/day OOS** — See `memory/project_tradecnn_baseline.md`
  - 13D features, StatePredictor, breakeven+trail, confidence 3.0, 2s slippage
  - Backup: `checkpoints/trade_cnn_backup_fast_v1/`
- **CNN fragility**: seed=42 required, symmetric loss only. See `memory/feedback_cnn_fragility.md`
- **NQ goal**: 3 months to NQ ($400 noise budget). See `memory/project_nq_goal.md`

## **DEPRIORITIZED (historical context only)**
- Auto seeds: `memory/project_auto_seeds_next.md` — superseded by grounded templates
- Quantum reconnect: `memory/project_quantum_reconnect.md` — physics metaphors purged
- ADX chop filter: `memory/project_adx_chop_filter.md` — ADX replaced by variance_ratio
- Peak override: `memory/feedback_peak_override_failed.md` — do NOT re-enable
- AdvanceEngine V2: paused, CNN/TradeCNN took priority

## Architecture — Core
- **SFE**: `core/statistical_field_engine.py` — MarketState per bar (CUDA-only)
- **Brain**: `core/bayesian_brain.py` — Bayesian table with hash-based state lookups
- **Trainer**: `training/trainer.py` — 7-phase pipeline. Run: `python training/trainer.py --fresh`
- **Exit Engine**: `core/exit_engine.py` — SL→TP→BandUrgent→EnvelopeDecay→BE→BeliefFlip→Hold
- **Execution Engine**: `core/execution_engine.py` — gate/direction/sizing
- **Feature Extraction**: `core/feature_extraction.py` — canonical 16D feature vector
- **Belief Network**: `core/timeframe_belief_network.py` — 11 TF workers, band confluence
- **Clustering**: `core/fractal_clustering.py` — TF-bucketed recursive K-Means

## Architecture — CNN/Trade
- **DMI Flipper**: `core/dmi_flipper.py` — smoothed cross, trail stop, breakeven, early exits
- **StatePredictor**: `core/trade_cnn.py` — CNNBackbone (13→32→64) + Head (64→7), ~16K params
- **TradeCNN Pipeline**: `training/train_trade_cnn.py` — 13D features, state labels, walk-forward
- **Direction CNN**: `training/direction_cnn.py` — 7D, seed=42, $736/day OOS
- **Direction TCN**: `training/direction_tcn.py` — dilated causal conv, $352/day OOS

## Architecture — Live
- **Live Engine**: `live/live_engine.py` — NT8 bridge orchestrator, DMI + TradeCNN modes
- **Launcher**: `live/launcher.py` — `--dmi`, `--trade-cnn`, `--dry-run` flags
- **Session Tracker**: `live/session_tracker.py` — PnL, drawdowns, trade log
- **Trade Logger**: `live/trade_logger.py` — per-trade diagnostic CSV
- **Atlas Loader**: `live/atlas_loader.py` — ATLAS parquet reader

## Report & Output Locations
| Path | Contents |
|------|----------|
| `reports/is/` | IS forward pass: oracle_trade_log.csv, signal_log.csv |
| `reports/oos/` | OOS forward pass: same structure |
| `reports/findings/` | Standalone finding reports |
| `reports/is_report.txt` / `oos_report.txt` | Summaries |

## CLI Flags — `python training/trainer.py`
- `--fresh` — full pipeline | `--train-only` — Phases 2-3
- `--forward-pass` — IS→OOS→Strategy→Phase 7 | `--oos` — standalone OOS
- `--data DATA/ATLAS_1DAY` — single-day fast validation

## Data Pipeline
- ATLAS: `DATA/ATLAS/{tf}/YYYY_MM.parquet` — 14 TFs, 12 months (Jan-Dec 2025)
- ATLAS_OOS: `DATA/ATLAS_OOS/` — 2 months (Jan-Feb 2026)
- ATLAS_1DAY / ATLAS_1WEEK — fast validation subsets

## Timeframe Rules
- **14 TFs**: 1s, 5s, 15s, 30s, 1m, 3m, 5m, 15m, 30m, 1h
- Oracle/template stats: 1m (discovery TF). Forward pass: 15s (execution TF).

## Analysis & Benchmark Tools
- `tools/trade_cnn_imr_overlay.py` — I-MR overlay + trade chart (4 panels, --date/--dpi)
- `tools/analyze_gates.py` — oracle-driven gate thresholds, `--apply` writes JSON
- `tools/standalone_research.py` — research harness (A-R modules)
- `tools/dmi_crossover_validation.py` — DMI crossover accuracy
- `tools/equity_risk_simulator.py` — equity growth simulation
- `tools/research/` — subpackage: data.py, imr.py, screening.py, seeds.py, plots.py
- `tools/archive/` — one-off scripts

## Validation Ladder (5 gates)
1. IS (ATLAS) → 2. OOS (ATLAS_OOS) → 3. Phase 7 Replay → 4. Live Sim → 5. Live Real

## Current Baselines
- **TradeCNN (2026-03-27)**: $1,609/day OOS, 24% WR, 10,571 trades. See `memory/project_tradecnn_baseline.md`
- **CNN 7D (2026-03-25)**: $736/day OOS, 31.7% WR, 4,021 trades
- **DMI cross (2026-03-25)**: $208/day, 30% WR, 25K trades. $400/day with early exits.
- **V7.0.0 TF binning (2026-03-12)**: $39,736 IS, $8,200 OOS, 96.6%/100% WR

## C&E Matrix Methodology
> Full methodology: `memory/ce_methodology.md`

## Research Lines
- **R4**: Regime Trading Framework — see `docs/REGIME_TRADING_SPEC.md`
- **Shape Primitives**: `tools/shape_primitive_builder.py`, `core/shape_primitives.py`
- **Telescoping TF**: see `memory/research_telescoping_tf.md`

## Design Docs
- `docs/Active/` — COUNTER_PROPOSAL_MTF_TWO_LAYER, PROPOSAL_MTF_SENSOR_ARRAY, RATIONALE_TCN_GROUNDED
- `docs/specs/` — LEVEL_DETECTOR, EXPECTED_PROFIT, WAVEFORM_SEED
- `docs/archive/` — completed specs

## User Profile
- See `memory/user_cognitive_style.md`, `memory/user_schedule.md`
- See `memory/user_vp_trading_system.md`, `memory/user_level_trading.md`

## Requirements
- PyTorch CUDA (cu121), numba, scipy, optuna>=3.5.0, databento
- CUDA required — CPU fallback removed
