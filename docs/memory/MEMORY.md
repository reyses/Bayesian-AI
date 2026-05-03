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
- **TOOLS INVENTORY**: Before building a new tool, check `research/TOOLS_INDEX.md` (master index of all 106 `tools/*.py`). Also browse `tools/_<category>/README.md` for categorized lists. If a matching tool exists, use or extend it — don't rebuild. Every new tool: add to both `research/TOOLS_INDEX.md` AND appropriate `tools/_<category>/README.md`.
- **DMAIC for project frame**: Each research project lives in `research/<topic>/project.md` (Define / Measure / Analyze / Improve / Control). Improve index points to cycle files.
- **PDCA for iteration**: Each research change = one cycle file under `research/<topic>/cycle_NN.md`. Plan written BEFORE code. Check compares actual to predicted. Artifacts in `research/<topic>/findings/YYYY-MM-DD_<topic>.md`.
- **Metric definitions (2026-04-22)**: Trade WR = (∑profit / \|∑loss\|) − 1 (0 break-even). $/trade reported as mode AND mean. Day WR = count-based (winning days / total).
- **OOS-only for NN-filtered validation (2026-04-23)**: Model trained on IS has seen that data. Running the engine on IS after training inflates results. Only OOS $/day is honest.

## Workflow Preference
- **TIER-BUILDING PLAYBOOK**: See `memory/tier_building_playbook.md` — consolidated 11-section methodology. Covers: data integrity checklist, three-question method (Q1 peak-bucket / Q2 hold cliff / Q3 peak signature), phantom entry + relaxation principle, direction-flip test, chain positions, 5 advanced EDA questions (peak-reacher, higher-TF state, resonance cascade, chop, gravity well), exit physics (trail, MAE, 5m alignment, thesis-validity), and anti-patterns. Supersedes `feedback_tier_three_questions.md`.
- **TIER-BUILDING METHOD SUMMARY** (read playbook for detail): three simple questions replace CART/ML brute force. Q1: are entries right? (peak-bucket). Q2: what persistent signal says we're wrong? (bar-N path → natural timescale → timeout). Q3: what do all peaks have in common? (entry→peak feature delta normalized by σ → universal 3-feature rule: p_center > 0.35 AND reversion_prob > 0.80 AND vr < 1.0, plus $10 amplitude gate).
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

## **CURRENT PRIORITIES (2026-05-03)**
- **V2 FEATURES × PRICE EDA STACK BUILT**: See [project_v2_features_eda_stack.md](project_v2_features_eda_stack.md). 7 descriptive tools (single/pair/triplet × current/lookback + volume×variation + visual overlay) characterize how every v2 feature behaves across regimes. **No fitting**, all on IS-only (208 days, 47k 5m bars). Headlines: 5m_velocity_1b × 5m_body pair gives WR 42-75% by quantile binning; 6-bar mono velocity → 70% WR / +38 ticks; chord (4h_body + 4h_velocity_1b + 4h_z_low_w) has cells **100% FLAT_SMOOTH** vs **100% FLAT_CHOPPY** (n=96 each — state fingerprints, no model); LOW_VOL × HIGH_VAR cell at 15m → +28.7 tick fwd (fakeout bounce); HIGH_VOL × LOW_VAR at 4h → 70% FLAT_CHOPPY purity (compression). 4h-TF features dominate state fingerprinting; 5m-TF dominates momentum patterns. Substrate for regime-conditional strategy work. Tools: `tools/v2_features_*_eda.py` (6 tools) + `tools/v2_features_overlay_viz.py`.
- **NEXT**: (1) OOS validation of all EDA findings on the 71 OOS days; (2) state-fingerprint deployment as NT8 rule; (3) regime-stratified rerun of all layers.

## **CURRENT PRIORITIES (2026-05-01)**
- **MA ALIGNMENT WINS**: See [project_v2_ma_alignment_directional.md](project_v2_ma_alignment_directional.md). 7-of-8 TF vwap_w alignment → 70.5% direction acc on 20% of 5m bars (+17.6% lift). Deterministic rule, no fit, walk-forward stable. Beats every fitted composite. **15m and 1h vwap windows carry the signal**; 5s-15s too noisy, 4h-1D too coarse for 5m decisions. Tool: `tools/v2_composite_ma_alignment.py`. Outputs: `reports/findings/v2_composite_ma_alignment/`. Commit `7dae2585`.
- **REGIME CONNECTION**: MA alignment IS a trend-regime classifier. Prior chop-edge work (zigzag wins on chop, loses on trend, d_OOS=0.77 — see [feedback_chop_edge_regime_filter.md](feedback_chop_edge_regime_filter.md)) and atlas_regime_labeler are the same problem from another angle. **Right joint system = regime-conditional strategy selection**: HIGH alignment → trend-follow MA-align direction; LOW alignment + chop → zigzag counter-trend; TRANSITIONAL → skip.
- **2D REGIME LABELS BUILT**: `tools/atlas_regime_labeler_2d.py` writes `DATA/ATLAS/regime_labels_2d.csv` (348 days × {UP/DOWN/FLAT} × {SMOOTH/CHOPPY} + IS/VAL/OOS split 60/20/20). Distribution: 25% UP, 18% DOWN, 57% FLAT; 64% SMOOTH, 36% CHOPPY. UP_CHOPPY (4 OOS) and DOWN_CHOPPY (5 OOS) are thin — flag for stat-sig. **Substrate for all future regime-conditional analysis.** Use `from tools.atlas_regime_labeler_2d import load_regime_labels`.
- **Next session — parallel tracks**: (1) per-regime MA alignment perf — hypothesis: near-perfect on UP_SMOOTH/DOWN_SMOOTH, near-zero on FLAT; (2) per-regime zigzag bleed_score perf stratified by 2D; (3) per-regime L-model perf. **Joint**: regime-router that selects strategy based on today's 2D label.
- **L is second best**: Standalone Analysis L on 5m base, |pred|>20 gate → 70.4% acc on 45% coverage (+10.6% lift over 59.8% baseline). Higher coverage but lower lift than MA. Useful as the magnitude estimator in the joint system. CSVs at `tools/plots/standalone/1y_<TF>/analysis_l_predictions.csv` (1m, 5m, 15m, 1h, 4h, 1D).
- **Composites tested and ruled out**: 5-voter L-aggregator, 2-voter (1m+5m), single-horizon refit, quantile (Q_0.25/Q_0.75 strict). All lose to standalone L AND to MA alignment. Each composite voter is a handicapped version of the full model — voting handicapped models can't recover what one full model exploits. See [project_v2_ma_alignment_directional.md](project_v2_ma_alignment_directional.md) "Anti-patterns ruled out".

## **CURRENT PRIORITIES (2026-04-17)**
- **HONEST BASELINE = -$164/day IS**. See [project_honest_baseline_2026_04_17.md](project_honest_baseline_2026_04_17.md)
  - Lookahead bias in `build_dataset.py` fixed (searchsorted shifted by period). See [feedback_lookahead_audit.md](feedback_lookahead_audit.md)
  - Previous $740/day baseline was pure lookahead inflation
  - Feature dir moved: `DATA/FEATURES_79D_1m/` → `DATA/ATLAS/FEATURES_5s/`
  - All 8 tiers at noise floor, all ~49% counter-flip (coin flip on direction)
  - KILL_SHOT peak physics DISPROVED — no feature changes at peak. See [feedback_peak_physics_dead_end.md](feedback_peak_physics_dead_end.md)
  - Oracle flip-at-exit upper bound: +$2,183/day pooled across 8 tiers
- **Frozen SFE cache bug** (live): fixed 2026-04-16. See [project_frozen_sfe_cache.md](project_frozen_sfe_cache.md)
  - All live sessions mid-Feb → 2026-04-16 traded on frozen features — PnL is noise
- **Next decision point**:
  1. Fix `training/regret.py` LOOKAHEAD (6-hour window distorts labels)
  2. Test CNN separability on FADE_CALM (24k trades) — if <58% OOS, tiers dead
  3. If (2) fails, rebuild tiers from corrected-trade clustering
- **NQ goal**: 3 months to NQ ($400 noise budget). See `memory/project_nq_goal.md`
- **Key insight from 2026-04-17**: nn_v2 playbook may not transfer. NMP was 30-35% counter (learnable); current tiers are 50% (near-random boundary).

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

## Current Baselines
- **2026-04-17 (HONEST, post-lookahead-fix)**: -$164/day IS on 348 days of 2025.
  Chains alone cost $157/day. Every tier at noise floor (-$16 to +$9/day, ~50% WR).
  See [project_honest_baseline_2026_04_17.md](project_honest_baseline_2026_04_17.md).
- **PRE-2026-04-17 numbers are CONTAMINATED by lookahead** — do not use as
  reference. $740/day, $620/day, $613/day all had higher-TF aggregation with
  future data baked in. Feature folder renamed to break accidental reuse.

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
- FEATURES (5s, honest): `DATA/ATLAS/FEATURES_5s/` (IS) + `DATA/ATLAS_OOS/FEATURES_5s/` (OOS)
  - Regenerated 2026-04-17 with lookahead fix. DATA/FEATURES_79D_1m/ is DELETED/STALE.

## Analysis Tools
- `tools/run_tier_isolated.py` — isolate each tier (no chains/catch-all), writes `training/output/isolated/{TIER}.pkl` (2026-04-17)
- `tools/killshot_peak_physics.py` — 5s path + peak physics measurement (2026-04-17)
- `tools/regret_on_isolated.py` — per-tier regret verdict table (2026-04-17)
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
