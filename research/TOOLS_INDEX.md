# Tools Inventory

Complete index of everything under `tools/`. Grouped by function. Each entry: **one-line purpose** + usage.

---

## 1. Data Pipeline — Atlas / Feature Building

| Tool | Purpose |
|---|---|
| `atlas_rebuild.py` | Clean 1s tick spikes + validate all TFs against 1s + rebuild ATLAS_FEATURES in one step |
| `build_feature_atlas.py` | Pre-compute 13D features for all TFs → `DATA/ATLAS_FEATURES/{tf}/YYYY_MM.parquet` |
| `build_timeframes.py` | Build 5s, 15s, 30s from 1s and 3m–30m from 1m; validates against 60-bar control aggregates |
| `clean_tick_spikes.py` | Remove single-bar price spikes (> THRESHOLD ticks from neighbors) via interpolation |
| `databento_to_atlas.py` | Convert Databento downloads → ATLAS parquet. Auto-detects schema (trades, 1s, 1m, 1h, 1d) |
| `rebuild_atlas_databento.py` | Rebuild ATLAS from Databento `.dbn.zst` (front-month MNQ only) |
| `nt8_export_to_atlas.py` | NT8 `.txt` exports → `DATA/ATLAS/{tf}/YYYY_MM_DD.parquet` |
| `nt8_to_atlas.py` | NT8 tick export → 1s ATLAS parquet |
| `nt8_to_parquet.py` | NT8 tick data → ALL TFs (1s through 1W) |
| `convert_nt8_atlas.py` | NT8 history CSV → ATLAS parquet |
| `rebuild_features.py` | Wipe + regenerate `FEATURES_5s/` for an atlas (warm-starts from sibling checkpoint) |
| `setup_oos_atlas.py` | Split ATLAS into IS (Jan-Nov 2025) + OOS (Dec 2025–Feb 2026) |
| `validate_data.py` | Cross-check every TF's OHLC against 1s ground truth; fix mismatches |

## 2. Labeling — Human-in-the-Loop

| Tool | Purpose |
|---|---|
| **`trade_marker.py`** | **Manually mark trades on price chart: click start / click end, direction auto. Saves to `DATA/regime_seeds/seeds_YYYY-MM-DD_multi.json`** |
| `peak_marker.py` | Manually mark peaks on a chart. Single click = mark. Saves JSON |
| `dmi_peak_marker.py` | Mark peaks on DMI/Volume chart (not price) — see exhaustion directly |
| `regime_labeler.py` | Step through I-MR regime segments one at a time. Y/N per segment. Saves seeds |
| `draw_levels.py` | Interactive level drawing (support/resistance) on 1m chart. Click to add horizontal lines |
| `swing_inspector.py` | Grade continuous swing groups (~10 trades per snapshot) |
| `seed_inspector.py` | Step through auto-detected I-MR seeds, accept/reject each |

## 3. Auto Seeds / Levels

| Tool | Purpose |
|---|---|
| `auto_swing_marker.py` | ZigZag-based swing detector calibrated from human seeds. Outputs seed JSON matching `trade_marker.py` format |
| `auto_levels.py` | Automatic support/resistance detection (learned from 31 weeks of hand-drawn levels) |
| `auto_seeds_day_chart.py` | Visualize auto seeds (ZigZag) vs physics peaks on single-day chart |
| `pivot_seed_scanner.py` | Find price pivots, measure swings, capture state at each; outputs REAL vs FAKEOUT classes |
| `pivot_seed_scanner_mtf.py` | MTF version: 1s pivots, 1m exhaustion confirm, 15s state capture |
| `build_peak_seeds.py` | Convert pivot scanner output → auto-swing seed JSON |
| `imr_to_seeds.py` | Convert I-MR regimes → seed JSON, optionally filtered by human-seed calibration |

## 4. RM Pivot Research (current session)

| Tool | Purpose |
|---|---|
| `measure_rm_pivot_direction_cohen_d.py` | Cycle 1: Cohen d of residual sign at RM zigzag pivots vs next-leg direction |
| `measure_rm_pivot_entry_direction.py` | Cycle 3: signal portfolio — direction HR + turning-point + oracle ceiling + daily stack |
| `forward_pass_rm_pivot.py` | Cycle 2 naive forward pass: enter at RM pivot, exit at next pivot |
| `sweep_phantom.py` | Stepwise sweep of phantom-entry params (wait bars, min-favorable, SL variations) |
| `time_to_wrong.py` | How fast do losers declare? Seconds to first cross of ±$1/$3/$5/$10 |
| `train_pivot_direction_nn.py` | Train CNN on 91D features at RM pivots → P(win). 6×15 grid, walk-forward |
| `train_tier_direction_nn.py` | Same CNN architecture on 9-tier trades (`iso_is.pkl`). Smaller, more regularized |
| `apply_pivot_nn_filter.py` | Post-hoc apply trained NN to trades pickle; classify TAKE/FLIP/SKIP; report uplift |
| `chord_ratio_analysis.py` | Two-chord ratio (price path vs RM chord) — classifies NOISE vs TREND regime at pivot |
| `cord_length_1m.py` | 1m price cord length = oracle upper bound on extractable PnL at threshold R |
| `cord_length_regression.py` | RM cord length — the honest ceiling (smoother, lower) |
| `cord_tradeable.py` | Realistic capture accounting for 2R retracement tax at zigzag confirmation |
| `pivot_daily_distribution.py` | Per-day PnL distribution for pivot-physics chain simulator |
| `pivot_zero_day_diagnosis.py` | Classify zero-PnL days (no trades vs wash vs small-positive) |
| `pivot_entry_variants.py` | Test 3 entry variants with identical exit physics |
| `pivot_accuracy_stratified.py` | Pivot direction accuracy stratified by chord ratio AND wick |

## 5. Pivot Forward Pass Variants

| Tool | Purpose |
|---|---|
| `pivot_physics_exit.py` | Physics-only exit sim — 1s pivot entry + 1m signal exit + 30s sniper, no SL |
| `pivot_physics_chains.py` | Same as `pivot_physics_exit` + chain multiplier (concurrent positions) |
| `pivot_residual_forward.py` | Zigzag + residual direction, real-time (no lookahead). Supports TP/SL sweep |
| `pivot_residual_sim.py` | Oracle-pivot sim (w/ lookahead) — baseline edge ceiling |
| `pivot_1s_forward.py` | Full 1s forward pass: 1s pivot detection AND 1s execution |
| `pivot_forward_1s.py` | Pivot-residual forward with 1s slippage resolution (OHLC intra-bar) |
| `regression_pivot_forward.py` | REGRESSION-LINE pivot forward pass with 1s slippage |
| `chart_1s_trades.py` | Chart 1s-pivot forward pass trades on a single day |
| `chart_rm_trades.py` | Chart RM-slope trades: price+RM+trade segments, slope panel |
| `chart_strategy_comparison.py` | CURRENT vs IDEAL (cusp → mean cross) strategies side-by-side |
| `chart_regression_z.py` | Regression mean + z overlay on price (zoomed-out) |
| `pivot_inspector.py` | Visualize real vs fakeout pivots color-coded on price chart |

## 6. Physics / Nightmare Engines

| Tool | Purpose |
|---|---|
| `nightmare_ticker.py` | Zero-lookahead forward pass. 1s → 1m aggregation, SFE warmed 300 bars, no peeking |
| `nightmare_runner.py` | Lightweight per-day wrapper around `nightmare_ticker` |
| `nightmare_forward_pass.py` | Nightmare Protocol forward pass with grounded features |
| `nightmare_oos_ticker.py` | Same ticker, OOS data |
| `nightmare_eda.py` | Deep analysis of losing exit types from a ticker run |
| `strategy_ticker.py` | Zero-lookahead forward pass: 79D + NN + Brain integration |
| `physics_day_chart.py` | Chart one day's physics (price + features) |
| `physics_exit_chart.py` | Exit chart — hold until physics signals done |
| `physics_failure_analysis.py` | Where does physics thesis fail? |
| `physics_funnel_oos.py` | IS-seed-physics → OOS match. Enter opposite for exhaustion |
| `physics_oos_full.py` | Full OOS physics pipeline |

## 7. Engine Variants / Entry Tests

| Tool | Purpose |
|---|---|
| `blended_test.py` | Blended engine test (one NMP + tiered exits) |
| `exnmp_trio_test.py` | Run all 3 ExNMP strategies (KillShot, Overshoot, Cascade) sequentially |
| `killshot_test.py` | NMP+wick rejection (KillShot engine) on IS/OOS |
| `overshoot_test.py` | NMP base + hold through mean for momentum overshoot |
| `wick_overshoot_test.py` | Kill shot entry + overshoot exit + breakeven regret |
| `saturation_sim.py` | Fixed TP / SL / timeout on every trade — saturation strategy |
| `eda_new_entries.py` | Stand-alone EDA of new entry strategies (REGIME_FLIP, MTF_EXHAUSTION, etc.) |
| `regret_new_entries.py` | Regret analysis on new entries |

## 8. Tier-Specific Analysis

| Tool | Purpose |
|---|---|
| `tier_eda.py` | Tier-by-tier surgical EDA. Segment / separator / peak / regime shift analysis |
| `tier_eda_killshot.py` | KILL_SHOT specific EDA — what separates good from bad |
| `tier_exit_physics.py` | Full physics dump per tier — everything for exit Q2/Q3 |
| `tier_daily_concentration.py` | Per-tier Pareto of daily PnL. Is edge spread or concentrated? |
| `tier_day_classifier.py` | Find day-level features separating BLEED from HARVEST days for a tier |
| `tier_day_rule_backtest.py` | Apply combined-feature day rule and measure $ lift |
| `tier_lookback_eda.py` | Pre-entry physics (10 min lookback) for winners vs losers |
| `tier_segment_diagnostic.py` | Split IS chronologically → stability of tier performance |
| `tier_sequence_analysis.py` | Do tier firings predict other tier firings? |
| `tier_signal_conflicts.py` | When do multiple tiers fire on the same bar? Which directions? |
| `tune_tier_thresholds.py` | Suggest threshold changes to improve tier WR |
| `run_tier_isolated.py` | Run each tier in isolation (only one tier fires) |
| `run_iso_tiers_isolated.py` | Isolated run for 5 NMP tiers |
| `iso_tier_audit.py` | Measure each nn_v2 tier on post-lookahead-fix features |
| `iso_tier_eda.py` | Max-fill tier forward pass — every tier evaluates every bar |
| `regret_on_isolated.py` | Per-tier regret table (actual vs optimal, capture %) |
| `corrected_regime_discovery.py` | CART on best_action labels — find counter-direction regimes |

## 9. Regret / Post-Run Analysis

| Tool | Purpose |
|---|---|
| `regret_analysis.py` | Post-run regret: how much money each exit reason left on the table |
| `analyze_pnl.py` | Quick PnL analysis on trades CSV |
| `analyze_gates.py` | Gate threshold analyzer — oracle-driven optimal gate settings |
| `trade_review.py` | Comprehensive post-run trade review tool |
| `hourly_oos_report.py` | PnL by hour for live-trading comparison |
| `sunday_hourly_eda.py` | Which Sunday hours are profitable? |
| `daily_hourly_pnl.py` | Daily mode/median/mean + hourly contribution. Revenue-stream view |
| `pnl_mode_buckets.py` | Bucket trades into 10 tick-aligned PnL ranges → find mode |
| `pnl_tier_distribution.py` | Per-tier + per-day aggregates + histogram |
| `giveback_analysis.py` | After peak PnL, how many bars until break-even? |
| `maxfill_regret.py` | Regret on max-fill trades (oracle-optimal PnL per tier) |
| `peak_capture_regret.py` | How much of each trade's 20-bar peak do we capture? |
| `pattern_regret_report.py` | Regret by pattern assignment |
| `pareto_loss_concentration.py` | 80/20 — which trades dominate total losses? |
| `winner_maxout_loser_rehab.py` | Trail stop on winners + direction flip on losers |
| `big_loss_entry_signature.py` | Features at entry that predict catastrophic losses (Cohen d) |
| `big_loss_physics.py` | When do catastrophic losers tip their hand (MAE / running PnL)? |

## 10. Peak Research

| Tool | Purpose |
|---|---|
| `peak_marker_analysis.py` | Analyze 60s pre-peak features from human-marked peaks |
| `peak_sequence_analysis.py` | −30 to +30 bars around peaks — what do velocity/volume/std/dmi do? |
| `peak_prediction_accuracy.py` | Does peak-detection raise reversion accuracy? (50.8% → 65%?) |
| `peak_prediction_layered.py` | Baseline + DMI + variance_ratio filter layers |
| `peak_verification_chart.py` | Visual verification — correct vs wrong peak predictions |
| `peak_template_research.py` | Find peaks, cluster 10-bar approach with UMAP+HDBSCAN |
| `peak_bucket_lifecycle.py` | Track trades through peak buckets over lifetime |
| `peak_signature_cluster.py` | Multiple exit signatures per tier (clustered) |
| `killshot_peak_physics.py` | 5s path + peak physics for KILL_SHOT trades |
| `milestone_thresholds.py` | Data-defined bar-N peak thresholds (not arbitrary) |

## 11. Level / Zone Analysis

| Tool | Purpose |
|---|---|
| `level_cascade.py` | Top-down S/R validation (4h down to 1s) |
| `level_shapes.py` | Classify price action at level touches (REVERSAL / BREAKOUT / …) |
| `zone_analysis.py` | Oscillation/variance/daily ranges inside level zones |
| `visual_shape_cnn.py` | 2D CNN on candlestick images — classify level touches |
| `bar_high_levels.py` | Level-related bar analysis |

## 12. Multi-TF / Resonance

| Tool | Purpose |
|---|---|
| `tf_pair_validation.py` | Parent-child TF peak validation — does parent agree? |
| `resonance_cascade_research.py` | Peak detection on all TF pairs → crash/rally signal |
| `amplifier_probability.py` | CASCADE-solo vs CASCADE-confirmed — statistical confidence |
| `measure_oscillations.py` | Natural oscillation period + amplitude per TF |

## 13. Feature / Physics EDA

| Tool | Purpose |
|---|---|
| `feature_eda.py` | Modules E1-E7 of multi-TF feature EDA |
| `feature_price_relationship.py` | 12 features × scatter/binned/correlation vs next-bar price |
| `feature_response_surface.py` | Which features (and combos) separate winners from losers |
| `mv_response_surface.py` | Multivariate (per-tier) feature interactions |
| `gate_interaction_matrix.py` | C&E matrix empirical validation — X → Y responses |
| `path_features_eda.py` | 12 5s bars preceding 1m entry — path-derived metrics |
| `bar_color_payoff.py` | Green/red bar color vs next-bar payoff |
| `bar_directional_wick.py` | Directional wick analysis (upper vs lower, rejection vs support) |
| `bar_flip_size.py` | Size of bar flips across TFs |
| `bar_wick_continuation.py` | Big-body (conviction) vs big-wick (rejection) — which continues more? |
| `movement_direction_eda.py` | Direction-prediction EDA (single + polynomial features) |
| `movement_z_stratified.py` | Direction stratified by z-score (mean-reversion pockets) |
| `tag_15_movements.py` | Tag every $15 / 8-min movement opportunity in raw 5s |
| `residual_by_sr_context.py` | Residual direction STRATIFIED by S/R level proximity |
| `minimum_prediction_window.py` | Given N bars of 1s data, at what N does sign prediction exceed 50%? |
| `slope_eda.py` | Regression mean kinematics — β (slope) + γ (curvature) via 30-sample OLS |
| `imr_analysis.py` | I-MR (moving range) analysis on trade replays |
| `imr_golden_path.py` | I-MR chart + golden-path overlay (1m resolution) |
| `imr_trade_chart.py` | Single-trade chart with entry/exit + MFE/MAE + physics subplots |
| `long_trade_path_eda.py` | 79D trajectory through 18+ min trade paths |
| `loser_cliff_eda.py` | Natural "dead" timescale for losers (bar where peak stalls) |
| `loser_physics.py` | Playbook that built ExNMP tiers — when flipping rescues losers |

## 14. Charts / Visualizations

| Tool | Purpose |
|---|---|
| `trade_visualizer.py` | Plots price waveform with entry/exit markers |
| `session_overlay.py` | 1h candlesticks + adaptive Fibs + 1m trades |
| `dmi_session_chart.py` | DMI-focused session chart |
| `dmi_swing_plot.py` | DMI swing plot |
| `dmi_imr_chart.py` | DMI I-MR chart (SPC-style control limits) |
| `dmi_peak_overlay.py` | Overlay human peaks on DMI + Volume |
| `dmi_se_overlay.py` | DMI + SE bands + Volume overlay |
| `pattern_map.py` | Where each detected pattern sits on price waveform |

## 15. NN Training

| Tool | Purpose |
|---|---|
| `train_pivot_direction_nn.py` | CNN on 91D at RM pivots → P(win). Walk-forward split |
| `train_tier_direction_nn.py` | CNN on 91D at tier entries → P(win). More regularization |
| `seed_oracle_trainer.py` | Entry classifier from manual seeds. 192D features, multi-TF |
| `shape_primitive_builder.py` | UMAP+HDBSCAN on entry+exit primitives |
| `visual_shape_cnn.py` | 2D CNN on candlestick images for level-touch classification |

## 16. Strategy Discovery

| Tool | Purpose |
|---|---|
| `strategy_miner.py` | Data-driven entry discovery — feature threshold scans |
| `derive_physics.py` | Extract entry/exit rules from corrected trades; rank features by discrimination |
| `template_instruction_test.py` | Apply template-direction + duration rule, measure |
| `seed_funnel_test.py` | Feature-trajectory funnel → template-match entry |
| `seed_funnel_oos.py` | OOS seed funnel: IS seed magnitude+duration profile match |
| `seed_pattern_analyzer.py` | Extract price waveforms from seeds; shape classification + cross-TF nesting |
| `exit_physics_eda.py` | What does 79D look like at regret's optimal exit? |
| `hypothesis_test.py` | Apply candidate filters → measure $/day impact |
| `cascade_order_optimizer.py` | Find TIER_PRIORITY order that maximizes PnL under cascade |

## 17. Lookahead / Parity / Validation

| Tool | Purpose |
|---|---|
| `compare_lookahead_impact.py` | Diff new (honest) vs old (inflated) blended trades |
| `lookahead_impact.py` | Per-day tier distribution vs archived baseline |
| `parity_check.py` | Live log vs baseline forward-pass timestamp-by-timestamp |
| `parity_validate.py` | FEATURE PARITY + TRADE PARITY in one check |
| `live_parity_check.py` | Build parity features from live bars, compare |
| `validate_sfe_parity.py` | SFE feed_bar() vs batch_compute_states (ground truth) |
| `generate_training_labels.py` | Forward PnL at 5/15/30/60/180 bars per 1m bar (no lookahead) |

## 18. Checkpoints / Snapshots

| Tool | Purpose |
|---|---|
| `checkpoint_viewer.py` | Dump checkpoint contents human-readable |
| `inspect_templates.py` | Load template checkpoint → human-readable report |
| `build_checkpoint.py` | Build checkpoint.json at specific end-date cutoff |
| `shard_reader.py` | Quick summary of in-progress signal-log shards |
| `tree_health.py` | Check tree leaf distinctness (approach/entry/exit 79D) |
| `precompute_live_states.py` | Pre-compute MarketState per TF; saved pickle for fast live launch |

## 19. Utilities / Support

| Tool | Purpose |
|---|---|
| `golden_path.py` | Y10/Y11/Y12 computation on 1s ATLAS. Oracle optimal segments |
| `equity_risk_simulator.py` | Dynamic position sizing from $10 floor |
| `l2_risk_budget.py` | How much risk to capture $30+ trades? MFE vs MAE cost |
| `standalone_research.py` | Orchestrator for `tools/research/` subpackage modules |
| `run_analytics.py` | Mini analytics runner on existing checkpoints |
| `seed_variance_check.py` | Slippage RNG seed variance (C3 phantom on IS, seeds 1-5) |
| `z_range_filter_backtest.py` | 1h z-range filter validation on blended forward-pass |

## 20. Calibration / Trajectories

| Tool | Purpose |
|---|---|
| `calibrate_trajectories.py` | Per-TF TrajectoryPredictor calibration (each TF answers different question) |
| `regression_line_cohen_d.py` | Regression-line Cohen d at zigzag pivots (smoothed signal direction) |
| `regression_line_cohen_d_sr.py` | Same + S/R features from 5 prior business days |
| `measure_bad_trade_holds.py` | Hold times: winners vs losers cohort |

---

## Notable current-session additions

Files added or heavily used in current research (RM pivot direction):

```
tools/measure_rm_pivot_direction_cohen_d.py   # Cycle 1
tools/forward_pass_rm_pivot.py                # Cycle 2
tools/measure_rm_pivot_entry_direction.py     # Cycle 3
tools/sweep_phantom.py                        # Phantom wait sweep
tools/time_to_wrong.py                        # Time-to-failure diagnostic
tools/train_pivot_direction_nn.py             # CNN training
tools/train_tier_direction_nn.py              # Tier CNN training
tools/apply_pivot_nn_filter.py                # Post-hoc filter apply
tools/chord_ratio_analysis.py                 # Regime (noise/trend) classifier
tools/cord_length_1m.py                       # Price cord oracle ceiling
tools/cord_length_regression.py               # RM cord oracle ceiling
tools/cord_tradeable.py                       # Realistic capture after 2R tax
tools/pivot_daily_distribution.py             # Per-day PnL distribution
tools/pivot_zero_day_diagnosis.py             # Zero-PnL day classifier
tools/chart_rm_trades.py                      # Per-day RM-slope trade chart
tools/chart_strategy_comparison.py            # Strategy comparison chart
tools/chart_1s_trades.py                      # 1s-pivot forward pass chart
```

And two training files in `training_RM_physics/`:

```
training_RM_physics/rm_physics_engine.py      # RM pivot + NN filter engine
training_RM_physics/ticker_1s.py              # 1s ticker (1s price, 5s features, 1m bar_data)
training_RM_physics/nn_direction.py           # Shared: feat_to_grid + PivotCNN + load_nn_filter
training_RM_physics/run_rm.py                 # Pipeline driver
```

---

## How to navigate this going forward

- **Before building a new tool**, grep this index: `grep -i <keyword> research/TOOLS_INDEX.md`.
- **Before any analysis**, ask "is there already a tool that answers this?"
- Every new tool added to `tools/` should land here too — keep the index live.
