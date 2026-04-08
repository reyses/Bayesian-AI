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
- **RCA PROCESS MANDATORY**: See `memory/feedback_rca_process.md` — follow the 9-step RCA for ALL system improvements. No shortcuts. No theoretical improvements without data.
- **1s TICKER IS THE ONLY HONEST TEST**: Batch SFE showed +$777, honest ticker showed +$48. Use `nightmare_ticker.py` for all testing. Always zero lookahead.
- **ANALYZE BY DAY, NOT MONTH**: See `memory/feedback_daily_hourly_review.md` — each day must stand on its own. Mode > mean.
- **SESSION PROTOCOL**: See `memory/feedback_session_protocol.md` — session end notes time, session start reads todo list.
- **Data validation FIRST**: Run `tools/validate_data.py` before ANY training or analysis. See `memory/feedback_data_validation_first.md`
- **No lookahead**: all analyses must mirror live conditions. Use only data available at decision time.
- **Always discuss before changing**: propose a plan, get approval, then execute
- **Challenge ideas HARD**: See `memory/feedback_challenge_ideas.md` + `memory/feedback_challenge_harder.md`
- **Don't be sloppy**: See `memory/feedback_sloppy_work.md` + `memory/feedback_cnn_fragility.md`
- **Live launcher defaults**: Default = send orders to NT8. `--dry-run` = opt-in. See `memory/feedback_live_defaults.md`
- **Progress bars mandatory**: tqdm with live stats for any loop > 100 iterations
- **Training via Bash**: show exact command, ask "Confirm to run?" — only execute after user confirms
- **NT8 bridge deploy**: copy to both `docs/` and NT8 indicators dir. Always bump version + timestamp.
- **Commit flow**: code first (commit+push), then reports/CSVs separately (commit+push)
- **Base measurements grounded**: See `memory/feedback_base_measurements.md`
- **Checkpoint every step**: All multi-step pipelines must save to disk after each step. See `memory/feedback_checkpoint_every_step.md`

## **CURRENT PRIORITIES (2026-04-08)**
- **nn_v2 3-CNN SYSTEM**: $620/day IS, $613/day OOS, 91% win days
  - Architecture: NMP → regret → blended (cascade/killshot/base) → 3 CNNs
  - **CNN Flip** (70.6%): entry direction from 6×13 TF grid. See `nn_v2/cnn_flip.py`
  - **CNN Hold** (94.8%): hold/exit decision during trade. See `nn_v2/cnn_hold.py`
  - **CNN Risk**: cuts losing trades early. See `nn_v2/cnn_risk.py`
  - Kill shot: |z|>2 + vr<1 + wick rejection (96% WR, $42/day standalone)
  - Blended engine: `nn_v2/nightmare_blended.py` (cascade + killshot + base_nmp tiers)
  - Next: Stage 2 — regret on CNN trades → discover new entry physics → expand roster
- **79D Dataset**: 345 days at 1m resolution (DATA/FEATURES_79D_1m/)
- **NQ goal**: 3 months to NQ ($400 noise budget). See `memory/project_nq_goal.md`
- **Key insight**: Regret is the teacher, CNN is the student. Trees exhausted at 55% direction.
- **Lookahead audit**: bars_norm mismatch (training vs inference) — needs fix

## **DEPRIORITIZED (2026-04-03)**
- Probabilistic system: `memory/project_probabilistic_system.md` — superseded by 79D NN
- CNN-Augmented Templates (23D): superseded by 79D
- TradeCNN: baseline was $1,609/day but on NT8 data (phantom spikes)
- MTF Two-Layer Counter-Proposal: superseded by nn_v2 pipeline

## **DEPRIORITIZED (historical context only)**
- Auto seeds: `memory/project_auto_seeds_next.md` — superseded by grounded templates
- Quantum reconnect: `memory/project_quantum_reconnect.md` — physics metaphors purged
- ADX chop filter: `memory/project_adx_chop_filter.md` — ADX replaced by variance_ratio
- Peak override: `memory/feedback_peak_override_failed.md` — do NOT re-enable
- AdvanceEngine V2: paused, CNN/TradeCNN took priority

## Architecture — nn_v2 (ACTIVE)
- **Pipeline**: ticker → aggregator → SFE → 79D → NMP → blended → 3 CNNs
- **Ticker**: `nn_v2/ticker.py` — dumb 1s bar pipe
- **Aggregator**: `nn_v2/aggregator.py` — 1s to all TFs + events
- **79D Builder**: `nn_v2/build_dataset.py` / `build_dataset_v2.py` — bulk GPU feature computation
- **NMP**: `nn_v2/nightmare.py` — z_se>2 + vr<1, inverse exit, 10-bar approach buffer
- **Blended**: `nn_v2/nightmare_blended.py` — cascade + killshot + base_nmp tiers
- **Kill Shot**: `nn_v2/nightmare_killshot.py` — wick rejection entry (96% WR)
- **CNN Flip**: `nn_v2/cnn_flip.py` — entry direction from 6×13 TF grid (70.6%)
- **CNN Hold**: `nn_v2/cnn_hold.py` — hold/exit during trade (94.8%, 98.9% HOLD acc)
- **CNN Risk**: `nn_v2/cnn_risk.py` — cut losing trades early
- **Regret**: `nn_v2/regret.py` — counterfactual (5 curves + entry lookback)
- **Tree**: `nn_v2/tree.py` — 79D → strategy (from corrected trade labels)
- **Book**: `nn_v2/book.py` — Bayesian leaf + versioned book + evolution CSV
- **AI**: `nn_v2/ai.py` — continuous positioning (LONG/SHORT/FLAT every bar)
- **Run**: `nn_v2/run.py` — single entry point for all commands

## Architecture — Core (legacy, still used by live)
- **SFE**: `core/statistical_field_engine.py` — MarketState per bar (CUDA-only)
- **DMI Flipper**: `core/dmi_flipper.py` — smoothed cross, trail stop, breakeven
- **Feature Extraction**: `core/feature_extraction.py` — 16D + 13D + 29D vectors

## Architecture — Live
- **Live Engine**: `live/live_engine.py` — NT8 bridge orchestrator, DMI + TradeCNN modes
- **Launcher**: `live/launcher.py` — `--dmi`, `--trade-cnn`, `--dry-run` flags
- **Session Tracker**: `live/session_tracker.py` — PnL, drawdowns, trade log

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

## Current Baselines (2026-04-07)
- **3-CNN Blended (nn_v2)**: $620/day IS, $613/day OOS, 91% win days, $22/trade
- **2-CNN Blended**: $605/day IS, $563/day OOS, 85% win days
- **Kill shot standalone**: $42/day OOS, 96% WR, 2.8 trades/day
- **Base NMP**: $10/day IS, $65/day OOS (raw, no CNN)

## Key Discoveries (from journals)
- **Phantom spikes were fake edge**: clean Databento data turned $4,350 into -$2,427
- **Wick rejection is universal quality signal**: predicts winners across all strategies
- **Tree exhausted at 55% direction**: CNN convolves full 6×13 grid, sees cross-TF patterns tree can't
- **Counter-flipping destroyed edge**: 54% WR on flipped trades (coin flip). Fix: corrected trades from regret
- **NMP exits throw away 97% of profit**: avg peak $98, captured $22 (overshoot), $0.40 (base)
- **Zero crossing pattern**: odd crossings = 100% winners, even = 100% losers
- **Breakeven lifespan**: 90% of winners clear BE permanently by bar 287 (24 min)

## Data Pipeline
- ATLAS: `DATA/ATLAS/{tf}/YYYY_MM.parquet` — 14 TFs, 12 months (Jan-Dec 2025)
- ATLAS_OOS: `DATA/ATLAS_OOS/` — 2 months (Jan-Feb 2026)
- FEATURES_79D_1m: `DATA/FEATURES_79D_1m/` — 345 days at 1m resolution

## Analysis Tools
- `tools/nightmare_eda.py` — deep exit analysis on clean data
- `tools/strategy_miner.py` — data-driven strategy discovery
- `tools/killshot_test.py` — kill shot validation
- `tools/derive_physics.py` — extract entry rules from corrected trades
- `tools/trade_cnn_imr_overlay.py` — I-MR overlay + trade chart

## Design Docs
- `docs/Active/` — MONTECARLO_VALIDATION_SPEC, SESSION_CONTEXT_FEATURES
- `docs/specs/` — FEATURE_VECTOR_79D_SPEC
- `docs/archive/` — completed specs

## User Profile
- See `memory/user_cognitive_style.md`, `memory/user_schedule.md`
- See `memory/user_system_specs.md` — Ryzen 5 5600X, 16GB RAM, RTX 3060 12GB VRAM

## Requirements
- PyTorch CUDA (cu121), numba, scipy, optuna>=3.5.0, databento
- CUDA required — CPU fallback removed
