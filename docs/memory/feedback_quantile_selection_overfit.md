---
name: Quantile-cell selection overfits massively without OOS validation
description: Discovered 2026-05-03 in Layer C1 triplet EDA — fine-grained quantile partitions on IS create cells that look like signal but are 75% noise. Always OOS-validate before quoting cell metrics.
type: feedback
originSessionId: e1202bdb-1787-4774-b48b-b312af212043
---
When evaluating quantile-binned feature combinations against a forward-return target,
**do not quote cell-mean / lift / WR statistics from IS as if they are real**. Run an
OOS validation pass *first*.

**Why**: Layer C1 in 2026-05-03 evaluated 270 triplets × 6 regimes × 27 cells (3 quantiles
per feature) = ~43k IS cells. Top |lift| cells reached **+100 ticks** above regime baseline
on IS. OOS validation (same IS-derived quantile boundaries applied to OOS data) showed
**only 25.8% of top-K cells survived**:

- 49 of 120 cells flipped sign on OOS
- 34 had |OOS lift| < 30% of |IS lift|
- The +99.9 IS-lift cell (`hurst_w_15m + price_sigma_w_5m + vol_velocity_w_15m UP_SMOOTH (1,2,1)`)
  collapsed to **−3.3** OOS — pure noise dressed up as signal
- True survivors had n_IS ≥ 200 AND a structurally sensible composition (1h structure
  anchor + amplitude carrier + reversion-context companion)

**How to apply**:

1. **Never quote a cell-mean lift from IS without OOS confirmation.** A cell that picks
   the top 1/27th of bars in a 270-triplet sweep is in the top 0.014% of all evaluated
   cells. Selection bias is enormous.
2. **Survival rule for cell-mean signals**: sign(OOS lift) == sign(IS lift) AND
   |OOS lift| ≥ 0.30 × |IS lift| AND n_oos ≥ 20.
3. **Going deeper amplifies the problem.** Layer C1 had 27 cells per triplet; Layer C2
   (4-feature) would have 81. Combinatorial cell-count growth turns spurious patterns
   into "winning" cells. Stop at C1, consolidate survivors, then test orthogonal feature
   combinations rather than adding more features.
4. **Trust large-n cells over high-lift cells.** In the C1 OOS test, the survivors had
   n_IS = 88 to 371 and modest IS lift (~+65 to +120). The biggest IS lifts (98, 95, 94)
   sat on n_IS = 96 to 123 and ALL flipped sign on OOS.
5. **Regime-conditional structural composition matters.** Survivors all had pattern:
   1h-window anchor (`vol_mean_w_1h`, `hurst_w_1h`, `bar_range_1h`) + low-TF amplitude
   (`swing_noise_w_1m`, `bar_range_5m`) + reversion-context (`z_se_w_15m`,
   `reversion_prob_w_15m`). Pure velocity/sigma combinations did not survive.

This applies to any future EDA layer: chord-finder, contextualizer, multi-feature
quantile-binned analysis. Always: pre-register a survival rule, validate on
held-out data, report only the survivors.
