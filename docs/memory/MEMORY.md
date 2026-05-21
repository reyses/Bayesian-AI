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
- **$/day lift framing (2026-05-17)**: See [feedback_dollar_lift_framing.md](feedback_dollar_lift_framing.md). Frame proposed $/day improvements against the HONEST FLOOR (post-caveat number), not the inflated headline. Translate to $/year. Weight tail-risk reduction separately. Compounding sizing layers (per-leg × per-hour × per-day) multiply, don't add.
- **OOS-only for NN-filtered validation (2026-04-23)**: Model trained on IS has seen that data. Running the engine on IS after training inflates results. Only OOS $/day is honest.
- **Quantile-cell selection overfits massively (2026-05-03)**: See `feedback_quantile_selection_overfit.md`. Layer C1 IS → OOS: 75% of top |lift| triplet cells flipped sign or collapsed in magnitude. Survival rate just 25.8%. ALWAYS OOS-validate quantile-cell metrics before quoting them. Trust large-n cells with structural composition over high-lift small-n cells.
- **Blender first, then drill down (2026-05-03)**: See `feedback_blender_first_then_drill.md`. User's research methodology — run unrestricted broad-strokes first, see what surfaces, drill into surprises. Do NOT pre-narrow scope based on what you think is interesting. When user opens a new analysis dimension, default to the full unrestricted run; pruning/optimization comes after, as a second pass for comparison.

## **CURRENT PRIORITIES (2026-05-20 EVENING -- FIRST SIM RUN: B9-HORIZON BUG FOUND+FIXED, ENGINE REFACTOR)**
- **First L5 SIM run (−$379) was an ENGINE BUG, not the strategy** (see [project_parity_b9_horizon_2026_05_20.md](project_parity_b9_horizon_2026_05_20.md)). Parity check vs the forward pass: validated strategy on 2026-05-20 ≈ −$59 (flat day). **B9-horizon bug**: l5_decider fired B9 at `pos.bars_held==5`; `bars_held` (core/ledger `//60`) is MINUTES → B9 fired 5min after entry, but the b9_K5 model is trained for 25s (K in 5s-bar units). 12× too late, every trade. **Fixed**: l5_decider fires B9 off a 5s-bar counter (`_bar_count − entry_bar_count >= B9_K`); core/ledger UNCHANGED (blended still needs minutes). Also fixed: mock_bridge fills stamped wall-clock not bar-time (had disabled B9 in the mock). **DURABLE TRAP**: `bars_held` is MINUTES; K horizons are 5s-bar units — never time a during-trade action off `pos.bars_held`. **DURABLE LESSON**: always parity-check a live SIM run vs the offline forward pass before interpreting its P&L. **engine_v2 refactored to zigzag-only** (blended path removed from the LIVE engine, Phase 1 of `docs/JULES_ZIGZAG_ONLY_REFACTOR.md`, verified byte-identical; Phase 2 = zigzag-state priming for the cold-start residual, PENDING). Tools: `tools/parity_live_vs_forward.py`, `tools/drawdown_stop_analysis.py`. Drawdown-stop ("cut at −$100") REJECTED: 76% of −$100 legs recover, B9 AUC 0.465 there.

## **CURRENT PRIORITIES (2026-05-20 -- MID-LEG ENTRY RESEARCH: REJECTED)**
- **MID-LEG / MISSED-SIGNAL LATE-JOIN = DO NOT BUILD** (see [project_midleg_entry_2026_05_20.md](project_midleg_entry_2026_05_20.md), report `reports/findings/regret_oracle/2026-05-20_midleg_entry_research.md`). Autonomous sleep-run, 5 experiments, 51-day sealed OOS. **Structural root cause**: hardened zigzag legs are a SEQUENTIAL PARTITION (leg N exits where leg N+1 is born; 99.8% of consecutive gaps = 0). A 1-contract greedy engine runs at 99.9% utilisation and catches 2,922/2,926 OOS legs -- there is NO "busy, missed a parallel leg" population. E1 (Fork 1, B9-gated, unconstrained) = +$303/day @ K=5 but over a population that doesn't exist live (it is the value of B9-*gating* late entries, not new money). E2 (Fork 2, B1-B6 augmentation) = -$76/day @ K=5 SIG NEGATIVE -- B9 GBM already extracts pivot-structure from raw V2 features, stacked B1-B6 preds add noise (lead-in-PCA pattern, [[feedback_leadin_pca_rejected]]). E3 (position-constrained sim) = incremental -$1/day, 4 late-joins/51d. Structural -- won't change with contract scaling. **The "lost signals" on the NT8 chart = cold-start (~7 first-of-session legs/day lost to the ~20-min warmup) or B7 skips -- NOT busy-missed legs. Real fix = zigzag-state priming (replay detect_swings + pre-warm feature engine at startup), recommended to user, AWAITING APPROVAL.** SIM deployment of the L5 stack is unaffected.

## **CURRENT PRIORITIES (2026-05-19 LATE-MORNING -- ENGINE = FORWARD-PASS VALIDATED)**
- **MOCK VALIDATED AS DECISION-IDENTICAL TO forward_pass_full_stack.** 5-day mock May 11-15 vs OOS Phase-1 forecast: $+3,582 vs $+3,716 (gap $-27/day = 3.6%). 20/20 EXACT entry+exit price match on May 11 spot check. Pipeline is deployable for SIM tomorrow.
- **--pivot-source=replay** flag added: feeds R-trigger events from `oos_hardened_legs_full.csv` instead of streaming detector. Enables regression-testing engine changes against known-good OOS forecasts before each live deploy. Production-live still uses streaming pivot detector (`--pivot-source=stream`, default).
- **Streaming pivot detector now matches production** (`tools/_viz/auto_swing_marker.detect_swings`): min_bars=36, ATR median of 42 TRs, 5-day pre-warmup. Per-day capture went from 28% (broken) to 98% (week-aggregate).
- **Same-bar flip mechanics**: L5Decider emits exit + new-direction entry in single batch; engine_v2 bypasses `can_enter` via `allow_flip=True` kwarg on `OrderManager.build_entry_order`. Required because OOS legs have exit_ts == next leg's entry_ts.
- **Zero-slip mock fills**: engine stamps `requested_price` on every PLACE_ORDER; `live/mock_bridge.py` honors it. Eliminated 1-bar-late phantom slippage from replay-loop race.
- **IS-calibrated thresholds locked in L5Decider**: B7_SKIP_THRESHOLD_NORMAL=1.90, _CAUTIOUS=2.10; B9_CUT_THRESHOLD_NORMAL=+5, _CAUTIOUS=+15. Must match `tools/forward_pass_1contract.py`.

## **CURRENT PRIORITIES (2026-05-19 sleep-run -- L5 SIM-DEPLOY BUILD, SUPERSEDED)**
- **L5 STACK BUILT + WIRED INTO engine_v2 BEHIND --engine-mode l5 FLAG. Phase-1 1c OOS NOT SIGNIFICANT.** See [project_l5_sim_deploy_2026_05_19.md](project_l5_sim_deploy_2026_05_19.md). Full autonomous sleep-run built: `training/live_feature_engine_v2.py` (185D V2 vector on demand, 300/300 parity vs batch at 1e-6); `live/l5_decider.py` (zigzag + B7 binary skip + B9 binary CUT at K=5 + B10 risk regime; same evaluate(state)->DecisionBatch interface as BlendedEngine); `tools/forward_pass_1contract.py` (IS-calibrated thresholds; B7 skip>=1.90, B9 cut<+5); `live/engine_v2.py` PATCHED ~30 LOC with engine_mode='l5' branch (default unchanged); `tools/preflight_check.py` (7 checks); `docs/Active/LIVE_L5_ARCHITECTURE.md` (thin-wrapper doc). **OOS gating (51d sealed)**: FLAT 1c $+454/day; Phase-1 stack $+496/day; **delta $+42/day CI [-$39,+$118] NOT SIG**. Deploy to SIM is appropriate (zero financial risk, real-time validation of pipeline). Production-money deployment GATED until SIM data refines thresholds.
- **THIN WRAPPER LIVE ARCHITECTURE** (see [feedback_thin_wrapper_live_engine.md](feedback_thin_wrapper_live_engine.md)): live engine is glue, not strategy. New live capabilities = new `Engine.evaluate(state)->DecisionBatch` implementation, NOT sidecars/parallel infrastructure. 2026-05-18 mistake (L5_sidecar.py + ZigzagRunnerHybrid duplicating engine_v2) is the anti-pattern. 2026-05-19 fix (L5Decider implementing the engine interface + engine_mode flag) is the right pattern.

## **CURRENT PRIORITIES (2026-05-18 LATE-LATE — C14/C15 results + B10 promotion)**
- **B10 PROMOTED: vol-regime sizer (was C14 v2 inverted action)**. Day-level layer COMPOSES with B9 multiplicatively. IS WF AUC 0.834, OOS AUC 0.949 for high-vol prediction. **Sealed OOS test PASS: +$69/day CI [+$7, +$144], single-shot with IS-selected thresholds (no grid search on OOS).** Slippage robust to $10/action ($64/day). Action surface INVERTED from intuition: zigzag wants vol (high-vol days mean $+868/day, low-vol days mean $+69/day with 3/6 negative rate). Boost 1.3x on P(high)>=0.5, cap 0.7x on P(low)>=0.7. Production model: `b10_vol_regime_{high,low}.pkl`. Trainer: `tools/train_b10_vol_regime_sizer.py`.
- **C15 FAILED**: pyramid-confidence stacker on B9 K=5. AUC 0.883 (strong signal!) but ALL recall budgets produce NEGATIVE delta -$29 to -$155/day. Root cause: 88.4% of B9's pyramids pay off; attenuating ANY costs 0.5×(exit-pnl_at_K) per leg. The 11.6% that would have hurt are smaller than the 88.4% that would have helped. **6th L5-paradigm-boundary confirmation**: any action that REFINES/CAPS B9 loses money. B9 stands alone at trade-level. Lesson: action surface MATTERS more than prediction quality.
- **C14 v1 (defensive gate) FAILED → C14 v2 INVERTED action PASSED**: vol-regime prediction is signal-rich (OOS AUC 0.949). v1 gated HIGH-vol days down → backwards because high-vol days are best for zigzag. v2 boost-high + cap-low → +$69/day OOS sig. Key methodological lesson: when AUC is high but operational delta is negative, INVERT the action before killing the model.
- **DRS REVERSED back to DO-NOT-DEPLOY on clean 51-day sample** (see `reports/findings/regret_oracle/b9_OOS_sealed_v2_51days.txt` + DRS extended eval): 23-day → 43-day "upgrade" (+$29/day, CI [+$54,+$472] sig) was sample-noise from mixed-quality partial-data days. Fresh NT8 dump + consolidated 51 clean days: Pearson DROPPED to +0.163 CI [-0.089, +0.447] not sig, rank_0.7_1.3 sizing collapsed to +$3/day CI [-$27, +$38]. Anti-predictive bias on negative days RETURNED. Lesson: borderline models need multi-pass revalidation across different sample compositions; "cleaner data" can REDUCE apparent signal by removing noise to overfit to.
- **B9 CROSS-SAMPLE CONFIRMED at +$66/day K=5, CI [+$41, +$94] on 51-day fresh dump**: virtually identical to original 31-day sealed result (+$67 CI [+$32, +$106]). CI tightened with more data, point estimate stable. K=10 (+$31) and K=30 (+$15) newly significant. Anti-doom stress: +$38/day at $10/action slippage. **B9 is the validated path forward.**
- **NT8 RAW CSV CONSOLIDATION**: layout = `DATA/RAW_NT8/{contract}/{tf}/{day}.csv` + `OLD/` archive (159 superseded). BayesianHistoryDumper v2.1.0 deployed (contract-first hierarchy). Fresh dump (52 days, 2026-03-19 to 2026-05-18) replaces partial older exports; +3 new days (May 16-18, where May 16 = Sat no trades). Each TF now byte-identical to 1s rebin (5s/15s/30s/5m/15m/30m/4h) or NT8-authoritative (1m has bad-tick filtering vs raw 1s spikes).

## **CURRENT PRIORITIES (2026-05-18 evening, SUPERSEDED by late revalidation)**
- **DRS VERDICT UPDATED: DEPLOY-WITH-MONITORING CANDIDATE** (was DO NOT DEPLOY in morning, see [reports/findings/drs/2026-05-18_drs_extended_verdict.md](reports/findings/drs/2026-05-18_drs_extended_verdict.md)). User flagged "we have data up to May 15" — found 13 NT8 CSV exports in `DATA/ATLAS_NT8/{1m,1s}/MNQ_06-26/` covering 2026-05-01 to 2026-05-15. Built `tools/sourcing/convert_nt8_csv_to_parquet.py` (1m: ts+59 shift, 1s: no shift, 5s: rebin 1s with `ts%5==4` alignment), verified rebin BYTE-IDENTICAL to existing parquet on overlap day (max diff 0.0000 OHLCV). 17 new days converted (Apr 28-30 + May 1, 3-8, 10-15). Rebuilt cross_day_features (305 rows / 43 NT8 days). Ran hardened forward pass on new days inline (no V2 needed). Refit DRS on FLAT target across IS+OOS (gbm_ev not available for new days without B7/V2). **Extended OOS Pearson +0.259 CI [+0.054, +0.472] SIGNIFICANT** (vs prior 23-day Pearson +0.139 CI [-0.047, +0.451] not sig). **Anti-predictive bias FIXED** (model now predicts $+495 mean on 10 negative days vs $+643 on positive days). Rank sizing rules: all 3 point-positive (+$29 to +$68/day), CIs just-cross-zero on lower bound (-$4 to -$10). Conservative `rank_0.7_1.3` has asymmetric risk/reward (-$4/day worst vs +$66/day upper CI). Path forward: shadow-deploy 30 days → conservative-live 30 days → scale. Lessons: 23-day verdict was premature; flat target more reliable than gbm_ev; user's "fix the format" intuition correct.
- **DRS CANONICAL VERDICT (morning, SUPERSEDED): DO NOT DEPLOY** (see [reports/findings/drs/2026-05-18_drs_canonical_verdict.md](reports/findings/drs/2026-05-18_drs_canonical_verdict.md)). Path A done (217 IS + 23 OOS days with honest gbm_ev hardened target). **IS walk-forward Pearson +0.191 CI [+0.098, +0.405]** strictly positive; **OOS sealed Pearson +0.139 CI [-0.047, +0.451]** crosses zero. Naive `clamp(pred/IS_mean, 0.5, 1.5)` rule on OOS: **-$333/day CI [-$523, -$163]** strictly NEGATIVE (baseline drift kills it). Rank-based rules robust to drift but ALL non-sig on OOS (-$20 to -$67/day point, CIs cross zero). **Anti-predictive on the 2 OOS negative days** (DRS predicted HIGHER for losers vs winners). DRS hypothesis not killed but doesn't survive sealed OOS — IS-OOS generalization gap is the wall. Useful features: overnight_gap_pct, prior_day_range_pct, days_since_fomc. Useless: dxy_*, vix_chg_prior, dow, is_fomc/cpi/nfp/opex (sparse binary). Next research direction: LLM-scored news headline intensity replacing binary event flags + collect more OOS days before re-evaluating. Production model artifact at `DATA/CROSS_DAY/drs_canonical_gbm.pkl` kept as research-only (NOT for deploy).
- **B-RENAME 2026-05-18**: Successful remaining-amplitude regressor PROMOTED from B10 → B9 (filling the slot freed when old B9 LSTM became C9). Failed bad-trade-cut DEMOTED to C11. New convention: candidates take lowest available C-slot; on validation, promote to lowest available B-slot. See [project_b_c_naming_convention.md](project_b_c_naming_convention.md). Current: B1-B9 ACTIVE (B9 = during-trade remaining-amplitude); B10, B11 RESERVED for next promotions; C12, C13 BUILT 2026-05-18 morning but both FAILED.
- **L5 PARADIGM BOUNDARY = R-TRIGGER IS STRUCTURALLY OPTIMAL FOR BINARY EXIT**: 5 independent attempts to beat R-trigger via binary cut/preempt/cap actions ALL fail walk-forward: morning trail-tightening with composite signals (-$0.29/leg), C11 bad-trade binary cut v1 (0/15 ops sig), C11 v2 cut-saves target (0/30 ops sig), C12 imminent-exit classifier (all delta strictly negative across 15 ops), C13 phase-gated pyramid cap (zero incremental over B9 alone). Only CONTINUOUS SIZING ON SIGNED AMPLITUDE prediction (B9) wins. Mechanism: continuous action keeps marginal predictions at size=1.0 (no Type 1 cost); only HIGH-CONFIDENCE predictions get cut/half/pyramid. Future L5 research must find a NEW action surface (NOT cut/preempt/cap-pyramid) to break the wall. Candidates for future: volatility-regime sideband, pyramid-confidence stacker, regime-clustered B9 variants.

## **CURRENT PRIORITIES (2026-05-17 sleep-run, FRESH)**
- **L5 DURING-TRADE PARADIGM VALIDATED via B9 (was B10), C11 (was B9 bad-trade) KILLED** (see [reports/findings/2026-05-17_b9_b10_l5_paradigm.md](reports/findings/2026-05-17_b9_b10_l5_paradigm.md) + [docs/L5_HYBRID_PIPELINE_SPEC.md](docs/L5_HYBRID_PIPELINE_SPEC.md)): Autonomous sleep-run built full pipeline. **B9 binary cut FAILED walk-forward both targets** — AUC 0.86-0.93 but Type 1 cost > Type 2 savings (mid-MAE trades recover). Confirms morning trail-tightening result: R-trigger structurally optimal under binary cut/hold. **B10 continuous remaining-amplitude regressor PASSED** — OOS sealed single-shot K=5: +$67/day CI [+$32, +$106] (14.1% lift over $475 baseline). Anti-doom slippage stress: +$38/day at $10/action. Action surface matters: continuous {cut, half, full, pyramid} only acts on high-confidence predictions, marginal cases stay at full size (no Type 1 cost). Production models: `reports/findings/regret_oracle/b10_remaining_amplitude_K{5,10,30,60,120}.pkl`. **NEXT (post-wake)**: build NT8 strategy T+25s timer + SIZE_QUERY IPC, build `live/L5_during_trade.py` sidecar, 30-day NT8 sim parity. Do NOT retune B10 on OOS — that contaminates.

## **CURRENT PRIORITIES (2026-05-17 evening)**
- **B/C NAMING CONVENTION** (see [project_b_c_naming_convention.md](project_b_c_naming_convention.md)): B-prefix = production candidates (active in stack, contiguous numbering, no gaps); C-prefix = failed candidates (research artifacts only, NOT in production). 2026-05-17 rename: old B9 LSTM leg-sizer → C9, old B10 LSTM direct-trade → C10 (both failed validation). B9 slot now RESERVED for new during-trade bad-trade detector. When a B fails, rename ALL files (`git mv b{N}_* c{N}_*`), update internal references, mark docstring "(failed candidate)". Next new during-trade model takes lowest available B-slot. Current stack: B1-B8 ACTIVE pre-entry, B9 IN DEVELOPMENT (bad-trade detector, first L5/during-trade), B10 RESERVED for B7-retarget remaining-amplitude.
- **DURING-TRADE B-STACK PARADIGM** (see [project_during_trade_b_stack.md](project_during_trade_b_stack.md)): User insight 2026-05-17 — all existing B1-B8 are SNAPSHOT-AT-ENTRY predictors; each has a during-trade analog that uses V2 + trajectory features at entry+K. Data processing inequality: P(outcome | features at K) >= P(outcome | features at entry). New L5 execution-layer paradigm. Enabler: `trade_trajectory_IS.parquet` (74,976 rows = 17,748 legs × 5 K horizons {5,10,30,60,120} in 5s units) built 2026-05-17 from `is_hardened_legs.csv` (275 days, $690/day flat IS baseline, zero trade management applied → labels honest). Operational analogs: B1→"opposite pivot forming?", B6→"next pivot direction from HERE", B7→"remaining amplitude" (= B10), B5→"where in MY leg?", B2→"faked out NOW?". Each unlocks different execution action (cut/flip/pyramid/tighten). **B9 (reclaimed slot, was LSTM leg-sizer → C9) = first L5 prototype: during-trade bad-trade detector**. Naive K=30-60 V2-only AUC 0.89-0.94, naive +$43-53/day on one val fold. Walk-forward CI is THE NEXT GATE — kill paradigm if CI crosses zero. Tools: `tools/{build_is_hardened_legs,build_trade_trajectory_dataset,trajectory_diagnostic,trajectory_diagnostic_v2}.py`.
- **NT8 ZIGZAG CALIBRATION = ATR(14) × 4 ON 1m BARS, PIVOTS ON 5s** (see [project_zigzag_calibration.md](project_zigzag_calibration.md)): Python `tools/build_zigzag_pivot_dataset.py` uses ATR period 14 on 1m bars × multiplier 4.0, pivot detection on 5s closes with min_bars=36. Neither deployed NT8 strategy matches: v1.0 raw is static R=30; v1.0.8-RC has UseDynamicR but ATR is on the pivot series (not 1m) AND defaults wrong (AtrLookbackBars=60 vs 14, AtrMultiplier=5.0 vs 4.0). `docs/nt8/ZigZagATR.cs` INDICATOR has the right architecture. To run a "raw" NT8 zigzag comparable to Python sim, need new `ZigzagRunnerNative_v1.0-RC.cs` consuming ZigZagATR.cs as pivot source. NT8 dev space cleaned 2026-05-17: 10 dead variants archived (3 indicators + 2 zigzag strategies + 1 NMP strategy + IPC reference kept).

## **CURRENT PRIORITIES (2026-05-16)**
- **REGRET ORACLE ARC = 6-LAYER ARCHITECTURE** (see [project_regret_six_layer_architecture.md](project_regret_six_layer_architecture.md)): L1 daisy-chain oracle DONE (7,925 trades / $1.04M/yr sequential ceiling) → L2 direction discrimination DONE (R²≈0.35 saturated; 82-86% accuracy in callable cells; 93% in extreme cells) → **L3 Bayesian Trade Archetypes via N-D Trajectory Clustering — PROTOCOL LOCKED 2026-05-16, BUILD PENDING** ([project_bayesian_archetypes_pending.md](project_bayesian_archetypes_pending.md), spec at `research/bayesian_archetypes/project.md`) → L4 selector/strategy MISSING → L5 execution model MISSING → L6 validation MISSING. When proposing new work, identify which layer it touches.
- **SIGNED_MFE PIVOT** (see [feedback_signed_mfe_pivot.md](feedback_signed_mfe_pivot.md)): For direction prediction on regret-oracle trades, target MUST be `signed_mfe = mfe_dollars × (+1 if LONG else −1)`. mfe_dollars alone hides direction signal — slope_15s_3m R² jumps 0.002 → ~0.20. Use signed_mfe for any direction work; use mfe_dollars only when specifically looking for magnitude amplifiers.
- **K-WAY R² SATURATION** (see [feedback_kway_r2_saturation.md](feedback_kway_r2_saturation.md)): Direction R² caps at ~0.35 on daisy-chain trades. k=5 with 2 bins = 0.348; stratified k=2 = 0.342-0.344. The 4-way and 5-way interactions add ~0. Don't escalate beyond k=3 with full bins — either reduce bin count or stratify. Stratification beats more features.
- **REGRET RESEARCH METHODOLOGY** (see [feedback_regret_research_methodology.md](feedback_regret_research_methodology.md)): The technical escalation ladder — 1D regression → quantile table → pair clusters+regression → triplet (only if pairs surface joints) → k=4/5 with reduced bins → stratified pair analysis. Levers when signal is weak: pivot target, stratify, add trajectory info. Lookahead audit MANDATORY on every selector-usable axis. Mandatory metrics per CLAUDE.md (mode + mean+CI, Wilson CI on pct_long).
- **USER COLLABORATION PROTOCOL** (see [user_collaboration_protocol.md](user_collaboration_protocol.md)): topic-at-a-time when designing; configurable defaults over preemptive engineering ("we won't know if we don't try"); critical-collaborator pushback expected; sleep-run handoffs are autonomous-with-journal/INDEX. Index of prompts-that-produced-breakthroughs for reading the user's mental model.
- **BINARY DIRECTION CLASSIFIER = L4 SIGNAL** (2026-05-16 late-late): `tools/regret_direction_classifier.py` on V2 entry features → AUC 0.864, Brier 0.142, no train-test gap. 88% acc at 40% coverage (threshold 0.85). Selector dial = \|P − 0.5\|. Massively beats cluster-routed approaches (R²=−0.05). Output: `reports/findings/regret_oracle/direction_classifier_v2_lr.npz`. **This is the operating signal for L4.**
- **LEAD-IN PCA REJECTED** (see [feedback_leadin_pca_rejected.md](feedback_leadin_pca_rejected.md)): Lead-in PCA signatures (centroid+direction in 184-D V2 space) HURT direction classifier at every lookback tested (60-bar/240-bar/720-bar). V2 entry features already encode the macro setup; lossy PCA summaries add noise the linear model overfits to. Next AUC lever = non-linear (GBM/CNN), not more features.
- **SCENARIO LSTM RESULT** (2026-05-16 autonomous-late): Multi-head LSTM (direction × duration × speed × trajectory) on lead-in K=60 bars × top-30 features → 2026 OOS dir 82.7% vs LR entry 81.0% (+1.7pp); duration tied; speed -4.3pp (LSTM HURTS); trajectory both below class-balanced baseline. **Information ceiling at ~83% direction accuracy confirmed for V2 entry features**, LSTM matches but barely exceeds LR. Lead-in trajectory carries minimal incremental signal. Trajectory bucket 84% MONOTONIC (oracle picks best extreme → most trades have MAE=$0); collapse to binary CLEAN/PULLBACK or drop head. Findings: `reports/findings/regret_oracle/2026-05-16_scenario_lstm.md`. Tools: `tools/scenario_{bucket_labeler,sequence_dataset,lstm_train,baseline_lr}.py`.
- **DIRECTION CLF ALONE = NOT A LIVE STRATEGY** (2026-05-16 honest-clean): See [feedback_direction_clf_alone_fails.md](feedback_direction_clf_alone_fails.md). Ran the LR direction classifier through the existing `training_iso_v2/` ticker+engine pipeline with tick-exact TP/SL exits (closing at exact threshold price via 5s OHLC, not bar close). All 72 grid configs (TP/SL × threshold × cadence) lose money OOS except one: T=0.95, TP=$20/SL=$5, 15m → +$2.54/day NET, **CI [−$4.22, +$10.34] crosses zero, NOT significant**. Direction acc 87% does NOT mean TP-hit rate 87% — classifier trained on hindsight-selected oracle bars; live firing at every 15m close means mid-move / end-of-move entries that rarely reach $20 favorable. **Entry timing is the unsolved bottleneck.** Three paths forward: (A) classifier as FILTER on existing tier strategies (recommended), (B) build separate entry-timing classifier, (C) price-action trigger + classifier routes. **Critical engine bug fixed**: existing HardStop/TakeProfit close at bar close after threshold cross → intrabar overshoot inflated $/day 20-30× in prior backtests. `training_iso_v2/exits_tick_exact.py` NEW with `TickExactTP/TickExactSL`; engine patched to honor `position.extras['_force_exit_price']`. `ledger.py` tracks `trough_pnl` (MAE). All future backtests must use tick-exact exits.
- **GOLDEN-MOMENT + DIRECTION COMBINED FAILS** (2026-05-16 golden-combined): See `reports/findings/regret_oracle/2026-05-16_golden_combined_failed.md`. Built entry-timing classifier on velocity-filtered positives (mfe_velocity > $5/min, ~2% positive rate, OOS AUC 0.81 GBM precision 14% at T=0.85). Paired with direction classifier. All 60 OOS grid configs LOSE money (-$15 to -$71/day NET, all CIs <0). **Killer insight**: at high P_timing bars, direction TP/(TP+SL) = 41% — BELOW 50% baseline. Direction clf 81% acc was on AVERAGE oracle bars (smooth setups); high-timing bars are extremes/inflections where direction is genuinely uncertain. The two V2-feature classifiers are anti-correlated where it matters most. **Velocity filter validated**: lower SL% (7%) than magnitude filter (11%), comparable ceiling. **Critical dataset bug fixed**: V2 layered has sub-5s irregular timestamps, `iloc[::12]` doesn't give clean 1m closes — must filter `ts%60==0` to match ticker's `is_1m_close`. **Tools**: `tools/{build_golden_entry_dataset,train_golden_entry_classifier,precompute_golden_timing,golden_ceiling_calc,golden_combined_kpi}.py`. **Next**: Path A — direction classifier as filter on existing tier strategies (FADE_CALM, CASCADE, MA_ALIGN). Tier-fire bars are price-action moments (not extremes), so direction clf's 81% acc should hold there.

## **CURRENT PRIORITIES (2026-05-10 evening)**
- **PROBABILITY TABLE SELECTION BIAS** (2026-05-10 evening): See [feedback_probability_table_selection_bias.md](feedback_probability_table_selection_bias.md). Empirical P_revert tables built by binning bars OVERSTATE the entry-strategy win rate. Validated: "strongest cell" P=0.71 produced 46-48% actual win rate when firing at first-bar-of-condition. Cusp-on-z (z stopped falling) recovers ~12pp but still IS-only edge (58.6% IS → 35.9% OOS).
- **CUSP/DECAY framework status** (2026-05-10): User's manual picks captured $827/day oracle on 4 marked days (cusp_marker tool built this session). Five simulator variants tested — all failed to translate to OOS edge. v5 pure-prob-gate is marginal exception: +$3.80/day at 52.1% OOS win rate (1:1 R/R). Framework is real (visible in IS) but doesn't generalize without regime gate or better cusp discriminator. Tools: `tools/sim_decay_rules.py`, `tools/sim_prob_gate_only.py`, `tools/sim_strongest_cell.py`, `tools/cusp_marker.py`, `tools/phase_gate.py`. Picks at `DATA/cusp_picks/`. Full results: `docs/daily/2026-05-10.md`.
- **3-BODY PROBABILITY = META-ROUTER** (per docs/daily/2026-05-09.md lines 906-984): Bayesian table is a Level-1 gate routing between {RIDE, BOW, NORMAL}. Cusp/fade only fires when P_cascade is LOW. Today's cusp_marker work = Level-2 logic; Level-1 still needs proper chord-based gating using `reports/findings/segments/bayes_table_v0_location/joint_5axis.csv` (6 stable reverter cells found, all small-n).

## **CURRENT PRIORITIES (2026-05-10)**
- **5s LEVEL IS INHERENTLY NOISE — substrate not predictor** (2026-05-10 morning): See [feedback_5s_inherently_noise.md](feedback_5s_inherently_noise.md). Confirmed by reclassifying 9,561 NOISE-after-STEEP_LINEAR_DOWN notes against 20-primitive SeedPrimitiveLibrary at r>=0.75 — 53.8% remained NOISE residual; P(fwd_up) uniform across all 19 classifiable sub-shapes (range 0.31-0.45) regardless of inner geometry. Even SYMMETRIC_V_UP under STEEP_LINEAR_DOWN parent → 34% UP. Parent context dominates; inner 5s shape carries no incremental signal. RULE: anchor predictions at measure (15s) / sub_motif (1m) / motif (5m), not note (5s). Note is a chord component, not a prediction target.

## **CURRENT PRIORITIES (2026-05-09)**
- **5-LEVEL HIERARCHICAL SEGMENTATION SUBSTRATE BUILT** (2026-05-09 evening): See [project_5level_segmentation_substrate.md](project_5level_segmentation_substrate.md). Recursive 2D-shape segmenter at 15m/5m/1m/15s/5s — 2,047 phrases / 6,276 motifs / 21,048 sub_motifs / 70,782 measures / 203,801 notes across 345 days. Parametric pipeline: bulk → stepped EDA → stepped surface regression (forward-return heatmaps shape×length, shape×sigma, shape×parent) → Bayesian tables (Beta(1,1) Jeffreys posteriors). 60 strong cross-level findings. Strongest: NOISE-after-STEEP_LINEAR_DOWN at note → 35.5% UP (n=9,539, CI [0.345, 0.365]); FLATLINE-after-rally at motif → 74.6% UP; STEEP_CONCAVE_UP within STEEP_LINEAR_UP at sub_motif → 68.4% UP. Substrate is the lookup foundation for the meta-router's P_cascade(t). Pending: TOD axis + calendar-event axis on every cell, OOS sign-stability per cell, per-tier oracle PnL per chord, V0 meta-router prototype.
- **ORIGINAL BAYESIAN BRAIN ARCHITECTURE DOCUMENTED** (2026-05-09 evening): See [project_original_bayesian_brain_architecture.md](project_original_bayesian_brain_architecture.md). User-described 8-step Feb 2026 pipeline: (1) scan IS → (2) build templates → (3) K-means distill → (4) scrub → (5) mount on Bayesian brain → (6) oracle + regret → (7) 9-gate cascade → (8) OOS validation. The architecture worked; the K-means step on multi-D feature vectors was the red flag (centroids averaged unrelated patterns into the same cell, false-merging real signals). V0 build replaces step 3 with quantile-bucketing on 5 explicit statistical primitives (slope/curv/z/sigma_rank/r2adj). Brain code deleted in commit 23db222f; modern `training_iso_v2/bayesian_table.py` is the surviving infrastructure (regime keying — to be replaced by chord). 9-gate cascade became 9 ExNMP tiers; new meta-router (Level 1) replaces gate-cascade with P_cascade-based routing.
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
