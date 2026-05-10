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
- **Quantile-cell selection overfits massively (2026-05-03)**: See `feedback_quantile_selection_overfit.md`. Layer C1 IS → OOS: 75% of top |lift| triplet cells flipped sign or collapsed in magnitude. Survival rate just 25.8%. ALWAYS OOS-validate quantile-cell metrics before quoting them. Trust large-n cells with structural composition over high-lift small-n cells.
- **Blender first, then drill down (2026-05-03)**: See `feedback_blender_first_then_drill.md`. User's research methodology — run unrestricted broad-strokes first, see what surfaces, drill into surprises. Do NOT pre-narrow scope based on what you think is interesting. When user opens a new analysis dimension, default to the full unrestricted run; pruning/optimization comes after, as a second pass for comparison.

## **CURRENT PRIORITIES (2026-05-09)**
- **METAPHORS MUST TRANSLATE TO MATH** (2026-05-09 evening, refined): See [feedback_no_human_regime_terms.md](feedback_no_human_regime_terms.md). Borrow methods and language from any discipline (physics, biology, trading, signal processing) — useful as shorthand. BUT every borrowed term needs an explicit math/statistical definition and the translation must be recorded in the file's translation table. Code labels and chart titles stay statistical-only (they stand alone, no glossary nearby). Descriptive prose can use metaphors WITH translation on first use. Translation table includes envelope = M_close ± k·SE_close, chop = high variation, compression = low sigma rank, pivot = sign(slope) flip with high curvature, etc. — extend on introduction.
- **CONTEXT FILTER vs TIER ARCHITECTURAL LOCK** (2026-05-09): See [project_context_filter_vs_tier.md](project_context_filter_vs_tier.md). Tiers fire entries with direction; context filters condition when tiers fire. The 4 FadeAtBand "robustness filters" and the CRM macro detector are context filters, NOT tiers. Compose multiplicatively as `entry AND f1 AND f2 AND ...`. New `filters/` directory convention; every new check must be classified as entry-component or context-filter.
- **FADE_AT_BAND ENTRY RULE REJECTED** (iso run 2026-05-09): See [project_fade_at_band_rejected.md](project_fade_at_band_rejected.md). Entry rule "5s touches 15m ±2σ → fade to 5m mean": IS net **−$17.27/day** on 261 days, OOS +$28.88 was a 6-FLAT_SMOOTH-day fluke (+$1,650 of +$1,964 OOS total). 3 of 6 2D regimes flip sign IS↔OOS (overfit fingerprint). Filters survive (see lock above), entry rule dies.
- **CRM detector v2** built (`tools/crm_pivot_detector_v2.py`) — context filter, 5-min monitor + triple-confirm. OVER-SUPPRESSES slow-buildup macro pivots (14:30 on 2026_02_12 rejected because σ-rank rises slowly past the window). Two-path confirm needed (fast 5min / slow 15-30min) OR vol-only fast confirm path.
- **Feature marker visualizer** (`tools/feature_marker.py`): click-drop-pin + toggle V2 features as overlays. Pins capture per-bar feature snapshot. For exploratory feature-pattern study around marked events.

## **CURRENT PRIORITIES (2026-05-08)**
- **USEFUL V2 SIGNALS** validated visually on 2026_02_12: see [project_useful_v2_signals.md](project_useful_v2_signals.md). Three-role composite: (1) 15m mean slope/curvature = strategic gate; (2) 1m-5m mean divergence = tactical entry; (3) divergence snap-back to zero = exit. Volume regime + swing_noise as additional context. VWAP redundant with regression mean; 4h mean useless intraday; 1m velocity too noisy; hurst/rprob jittery/saturated.

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

## **CURRENT PRIORITIES (2026-05-05)**
- **REGIME × DIRECTION FLIP RULE FOUND** (V2 NMP discovery): See [project_v2_flip_rule_discovery.md](project_v2_flip_rule_discovery.md). Per-cell EDA on 19,106 NMP IS trades found 3 (regime, direction) cells where flipping converts losers to winners: `(UP_SMOOTH, short)`, `(UP_CHOPPY, short)`, `(DOWN_SMOOTH, long)`. IS walk-forward CI [+$21, +$165] significant; **OOS engine impact is modest +$1.66/day** (CI [-$48, +$55]) because trade-time displacement and re-simulation overestimated by ~40×. Production strategy: `RegimeAwareReversion` (NMP_REGIME).
- **CRITICAL BUG FIXED**: `ZSeReversal` exit was firing bar 1 on every flipped trade (fade-thesis exit applied to ride-direction position). Fixed: skips when `entry_tier in {NMP_FLIP, MA_ALIGN, NMP_RIDE}` or `extras['flipped_from']` set. Without this fix, flip rule produces $0.00/trade.
- **PER-CELL CONTINUOUS FILTERS OVERFIT** (rejected): 9/12 cells had walk-forward-surviving Cohen's d 0.11-0.34 INSIDE IS. Built `FilteredRegimeAwareReversion`; **OOS result: -$19.85/day** (filter HURT). Bootstrap CI [-$59, +$17]. **70/30 walk-forward inside IS is NOT enough validation**; continuous-feature quantile thresholds break on date-disjoint OOS. Same lesson as 2026-05-03 quantile-cell overfitting.
- **V2-NATIVE ISO PIPELINE BUILT** (2026-05-05): See [project_v2_iso_pipeline.md](project_v2_iso_pipeline.md). `training_iso_v2/` (no space) replaces misnamed `training_iso V2/`. 9 legacy ExNMP tiers ported V2-native (FADE_CALM/MOMENTUM, RIDE_CALM/MOMENTUM, FADE_AGAINST, RIDE_AGAINST, KILL_SHOT, CASCADE, FREIGHT_TRAIN). Pure OHLCV wick math; multi-TF OHLCV in ticker; iso orchestrator runs N parallel engines. New `OUReversionDecay` exit fires when current rprob falls to `entry_rprob × 0.6` (OU thesis decay). Smoke: 13 trades / +$325 / 1 day. KILL_SHOT/CASCADE/FREIGHT_TRAIN need wick+velocity threshold recalibration on full IS (V2 units differ from V1 defaults).
- **V2-NATIVE TIER PLAYBOOK** (2026-05-05): See [tier_building_playbook_v2.md](tier_building_playbook_v2.md). 6-step regret-driven discovery cycle: (1) run seed, (2) `regret_full.py` multi-axis labels, (3) categorical → continuous → time splitter axes, (4) build strategy variant, (5) IS-WF + true-OOS validation, (6) ship/reject. Categorical splitters survive >> continuous quantiles. All 5 anti-patterns confirmed this session.
- **HIGH-VOL "HARNESS" ANGLE FAILED**: See [feedback_high_vol_harness_failed.md](feedback_high_vol_harness_failed.md). Loser autopsy showed entry-volatility correlates with bleed (NY hours / FLAT_CHOPPY / round-trips all share high `L2_1m_vol_mean_15`). Tested 2 levers: (1) flip direction in high-vol — peaks are symmetric, rejected; (2) vol-adaptive exit thresholds — fat-tailed peak distribution makes mean-based formulas overshoot, OOS -$112/day, rejected. Discovery: re-sim with prod thresholds gets +$4.68/t in Q5 vs engine actual +$0.33/t — state-driven exits (ZSeReversal etc.) eating high-vol profit. Real lever: surgical state-exit modification, not exit-threshold rescaling.
- **AUDIT: `training_iso V2/` (folder with space) is misnamed** — 9 of 11 `.py` files still import from `core.features`/`core.statistical_field_engine`/`training.sfe_ticker` (V1). `training_RM_physics_v2/` is correctly V2-pure (4/6 files). Flagged for cleanup.

## **CURRENT PRIORITIES (2026-05-04)**
- **V2-NATIVE TRAINING PIPELINE BUILT**: See [project_v2_native_training.md](project_v2_native_training.md). Clean rebuild of `training_v2/` reading `core_v2.features` directly (185D V2 layered, no V1 conversion). Components: ticker, engine, ledger, strategies (MA_ALIGN + REVERSION; VEL_BODY_CHORD killed), regret, Bayesian threshold deriver, V2DirectionCNN scaffolded. Production thresholds at `training_v2/output/thresholds_prod.json`.
- **THRESHOLD-TUNING CEILING ≈ +$28/day OOS**: See [feedback_threshold_tuning_ceiling.md](feedback_threshold_tuning_ceiling.md). Adaptive exits double baseline ($27→$55/day) but CI [-$5, +$63] stays just below significance at n=68. Cell granularity (regime-only / tier-only / tier×regime) all give within $2/day. **Bottleneck is entries, not exits.** Lever next: CNN as filter+entry.
- **OUTLIER-DAY TRAP**: See [feedback_outlier_day_optimizer.md](feedback_outlier_day_optimizer.md). Total-PnL grid optimizer hid 2026-03-20's $49k VEL_BODY_CHORD lottery as "+$713/day OOS uplift". Bootstrap CI revealed it: top-1 day = 97% concentration. **Default threshold-optimizer objective: median_day, not total.** VEL_BODY_CHORD permanently killed.
- **9-TIER DISCOVERY EXERCISE FAILED AT V2 SINGLE-COLUMN**: See [project_9tier_discovery_v2.md](project_9tier_discovery_v2.md). NMP-only IS produced 19,106 trades; FADE_BETTER vs FLIP_BETTER split is ~50/50 with $9-10/trade gap. But max Cohen's d across 185 V2 columns at entry = 0.040 (negligible); 0/25 features survive walk-forward. The legacy 70.6% CNN flip relied on cross-feature patterns + features (wick_ratio, dmi_diff) NOT in our V2 entry vector. Three paths: (A) add directional wicks to entry, (B) chord-style joint-quantile pairs, (C) train V2DirectionCNN.

## **CURRENT PRIORITIES (2026-05-03)**
- **V2 FEATURES × PRICE EDA STACK BUILT (9 layers)**: See [project_v2_features_eda_stack.md](project_v2_features_eda_stack.md). 9 descriptive tools (TF sweep + contextualization + single/pair/triplet × current/lookback + volume×variation + visual overlay). **No fitting**, all on IS-only (208 days, 47k 5m bars).
- **STRUCTURAL FINDING**: The composite signal CAN'T be additive. `body` at 1h FLIPS sign in correlation with forward return depending on `vol_sigma_w`'s quantile (corr ranges −0.108 to +0.039). Right composite framework is **conditional** on modifier quantile, not summing/averaging. Operates at TWO levels: day-level regime-router (route between strategies based on regime_2d) + bar-level contextualizer-router (target sign depends on modifier quantile).
- **TF INVERSION findings**: same concept's regime relationship CHANGES character with timescale. bar_range −0.18 (5s) to +0.18 (1D); vol_velocity_w only signals at 1D (-0.21 capitulation); price_accel_w 0 at short TFs to +0.55 at 1D. Universal directional carriers: price_velocity_w (+1.25 at 1h), body, vol_sigma_w (-0.41 at 1h).
- **CHORD findings**: 5m_velocity_1b × 5m_body pair WR 42-75% by quantile binning; 6-bar mono velocity → 70% WR / +38 ticks; chord (4h_body + 4h_velocity_1b + 4h_z_low_w) has cells **100% FLAT_SMOOTH** vs **100% FLAT_CHOPPY** (n=96 each — state fingerprints).
- **VOLUME × VARIATION findings**: LOW_VOL × HIGH_VAR cell at 15m → +28.7 tick fwd (fakeout bounce); HIGH_VOL × LOW_VAR at 4h → 70% FLAT_CHOPPY purity (compression). 4h-TF features dominate state fingerprinting; 1h-TF dominates contextualization.
- **NEXT (5 priorities)**: (1) Compute exact conditional rules from top contextualizers (when modifier in Q3, flip target sign); (2) OOS validation of all findings on 71 OOS days; (3) state-fingerprint NT8 deployment; (4) TF-axis × contextualizer cross (do contextualizer effects also invert across TFs?); (5) regime-stratified rerun. Tools: `tools/v2_features_*.py` (8 tools).

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
