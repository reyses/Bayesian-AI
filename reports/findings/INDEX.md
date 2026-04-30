# Reports/Findings Master Index

> Navigation aid for the 125+ markdown reports + 21 subdirectories in this folder.
> Most recent overnight (2026-04-29) and morning (2026-04-30) work indexed first.

---

## 🌟 Start here (most recent + most actionable)

| File | What's inside | Read time |
|---|---|---:|
| `MORNING_BRIEFING_2026-04-30.md` | 1-page TL;DR with $165k headline + decision tree | 1 min |
| `HEADLINE_VALIDATION_2026-04-30.md` | Sanity-check the $165k across 3 sources + holdout | 2 min |
| `OVERNIGHT_SUMMARY_2026-04-29.md` | Full overnight synthesis, 6 analyses linked | 5 min |
| `TIER_EDA_FIX_PLAN_2026-04-29.md` | Per-tier diagnosis + remediation | 5 min |

---

## Today's work (2026-04-30)

### Validation tests
- `base_nmp_param_comparison/2026-04-30_results.md` — overfit hypothesis CONFIRMED. Standalone BaseNmpRunner loses money on full 14 months across all 3 param sets including SFE canonical. Tier-context required.

---

## Overnight work (2026-04-29) — by topic

### Tier system × regime × LinReg slope filter (the headline finding)
- `tier_pnl_by_regime/2026-04-29_summary.md` — main cross-tab, profitable subsets identified
- `tier_pnl_by_regime/2026-04-29_10_tier_linreg_slope_filter.md` — **the +$33k filter breakdown**
- `tier_pnl_by_regime/2026-04-29_12_slope_filter_train_test.md` — **70/30 holdout validation**
- `tier_pnl_by_regime/2026-04-29_09_phase_winners.md` — best tier per 1W phase
- `tier_pnl_by_regime/2026-04-29_07_tier_x_1w_phase.csv` — tier × phase data
- `tier_pnl_by_regime/2026-04-29_13_fade_calm_dropout.md` — FADE_CALM regime-dependence solved
- `tier_pnl_by_regime/2026-04-29_14_all_tiers_entry_envelope.md` — per-tier signature analysis
- `tier_pnl_by_regime/2026-04-29_08_tier_timeline.png` — visual: cumulative tier PnL

### Strategy PnL diagnosis (v1.0.4 baseline)
- `strategy_pnl_by_regime/2026-04-29_v104_pnl_by_regime.md` — DOWN regimes = 101% of losses

### Peak-feature relationships (the canonical 91D space)
- `peak_feature_overlay/2026-04-29_summary.md` — top features by H/L separation
- `peak_feature_overlay/2026-04-29_01_effect_size_table.csv` — Cohen's d for all 91 features
- `peak_feature_overlay/2026-04-29_02_per_feature_dists.png` — top-24 distributions

### Macro segments + zones (3-body framing)
- `macro_segments/2026-04-29_summary.md` — full output of macro_slope_segmenter
- `macro_segments/1W_chart.png`, `1D_chart.png`, `4h_chart.png` — visualizations
- `macro_segments/1W_segments.csv`, etc. — segment data + zone behavior

### Cross-TF nesting (UP/DOWN asymmetry)
- `regime_eda/2026-04-29_cross_tf_nesting_full_14mo.md` — UP-legs 1.2-1.85× more sub-pivots
- `regime_eda/2026-04-29_cross_tf_nesting_full_14mo.png` — visualization
- `regime_eda/2026-04-29_cross_tf_direction_agreement_full_14mo.csv` — agreement %

### Auto peak detection (multi-TF)
- `regime_eda/2026-04-29_auto_peaks_zigzag_4h.md` — calibration F1 results
- `regime_eda/2026-04-29_auto_peaks_zigzag_1h.md`
- `regime_eda/2026-04-29_auto_peaks_zigzag_15m.md`

### Standalone research (regression on features)
- `../tools/standalone_report.txt` — full output (regression + regimes + oracle)

---

## Earlier findings (pre-2026-04-29) — by topic

### v1.0.x backtests
- `2026-04-28_v106_full_backtest.md` — v1.0.6-RC full 14-month: -$75/day (window-fit GA)
- `trades_v1.0.4_playback_trajectory_analysis.md` — bar-by-bar v1.0.4 path
- `2026-04-27_v104_eda_phase1.md` — hold-time cliff, hard-stop sweep
- `2026-04-27_v104_phase2_regression.md` — feature regression on trade outcomes

### Earlier baselines
- `2026-04-17_killshot_peak_physics.md` — peak physics test (DISPROVED)
- `2026-03-23_peak_research_grounded.md` — original peak research

### Tier-specific deep dives
- `peak_capture_regret_FADE_AGAINST.md`
- `peak_capture_regret_RIDE_AGAINST.md`
- `peak_capture_regret_KILL_SHOT_ACTIVE.md`
- `peak_capture_regret_KILL_SHOT_CALM.md`
- `peak_capture_regret_NMP_FADE.md`
- `peak_capture_regret_MTF_EXHAUSTION.md`
- `peak_lifecycle_FADE_AGAINST.md`
- `peak_lifecycle_KILL_SHOT.md`

---

## Subdirectories

| Directory | What's inside |
|---|---|
| `tier_pnl_by_regime/` | All tier × regime cross-tabs from overnight |
| `strategy_pnl_by_regime/` | v1.0.4 PnL × regime analysis |
| `peak_feature_overlay/` | Peak × feature effect-size analysis |
| `macro_segments/` | Slope-segmenter outputs (1W/1D/4h) |
| `regime_eda/` | Auto-peaks calibration, cross-TF nesting, regime visualizations |
| `linreg_eda/` | Earlier LinReg + zigzag charts |
| `base_nmp_param_comparison/` | Today's overfit-hypothesis test |
| `fake_vs_real_peaks/` | Old peak-classification work |
| `regret/`, `regret_v2/` | Counterfactual regret analyses |
| `mtf/` | Multi-TF analyses |
| `volume_research/` | Volume-based research |

---

## How to navigate

- **Want the answer?** Read `MORNING_BRIEFING_2026-04-30.md`.
- **Want details on the +$165k figure?** Read `OVERNIGHT_SUMMARY_2026-04-29.md` then `tier_pnl_by_regime/2026-04-29_10_tier_linreg_slope_filter.md`.
- **Want per-tier remediation?** Read `TIER_EDA_FIX_PLAN_2026-04-29.md`.
- **Suspicious of a number?** Pull the underlying CSV from the right subdir.
- **Looking for an old result?** Browse this index by topic, or search filename: `find reports/findings -name "*topic*"`.
