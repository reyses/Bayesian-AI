# 2026-05-17 — DRS Feasibility Verdict (Phase 1B)

## TL;DR

Cross-day features have **weak but real predictive signal** for day-level
PnL on the peeky proxy target. **OOS Pearson R = +0.253, 95% CI [+0.181,
+0.447]** across 5-fold walk-forward on 228 days. CI is strictly positive
but lower bound sits just under my "strong" threshold of +0.20.

MAE lift over persistence baseline: **+35.4%** ($2,688 → $1,738), which IS
meaningful. The DRS isn't dead, but the effect size is more modest than
the +$200-500/day I projected. Realistic expectation now: **+$100-200/day
mean lift** plus tail-risk reduction.

**Recommendation**: proceed to Path A for honest target. Outcome unlikely
to flip the verdict but will give a defensible deployment number.

## Result table

| Metric | Value |
|--------|-------|
| Days included | 228 (after NaN drop, mostly missing VIX/DXY at boundaries) |
| Walk-forward folds | 5 (38 test days each, sequential) |
| OOS Pearson R | **+0.253** |
| 95% bootstrap CI on Pearson | **[+0.181, +0.447]** |
| OOS R² | +0.031 (low — heavy variance) |
| OOS MAE | $1,738 |
| Persistence-baseline MAE | $2,688 |
| MAE lift vs persistence | **+35.4%** |

Per-fold Pearson:
- Fold 1: NaN (only 38 train days, constant predictions)
- Fold 2: +0.295
- Fold 3: +0.486
- Fold 4: +0.705
- Fold 5: +0.192

High variance across folds suggests regime-specific signal: the model
works well in some periods, poorly in others. This is expected from a
heavy-tail-distributed strategy on a small sample.

## Feature importance (permutation, last fold)

**Positive contributors** (permuting hurts):
- `overnight_gap_pct`: +$89.5 MAE delta (by far the dominant signal)
- `days_since_fomc`: +$34.2
- `days_to_next_fomc`: +$16.7

**Negligible** (sparse binary):
- `is_fomc`, `is_cpi`, `is_nfp`, `is_opex` (all ≤0.0 — small N, not informative)

**Apparently harmful** (permuting helps — sign of overfitting):
- `vix_close_prior`, `dxy_close_prior`, `prior_day_range_pct`, `prior_day_c2c_pct`
- These features make the model fit training noise. With less data they
  could help; with 190 train days they hurt. **Likely candidates for
  removal in v2.**

## Interpretation

1. **Overnight gap is the primary signal.** No surprise: big overnight
   gaps correlate with high-volatility days, which are good for a
   zigzag-style strategy that benefits from price movement.

2. **FOMC-proximity matters more than FOMC itself.** The day-of-FOMC
   binary doesn't help much (sparse), but days-since/to-next-FOMC do.
   This is the Fed cycle effect — pre-meeting calm vs post-meeting
   continuation, etc.

3. **VIX and DXY DIDN'T help in this run.** Two interpretations:
   a) The peeky proxy is noisy enough that VIX/DXY contributions don't
      separate from noise.
   b) The model overfits to them with only 190 train days; needs more
      regularization or feature engineering (e.g., VIX percentile rather
      than absolute).
   The Path A canonical run should clarify which is true.

4. **Persistence beat is real.** The DRS GBM improves MAE 35% over
   "predict yesterday's pnl for today" — that's a non-trivial lift
   even if Pearson is only 0.25.

## Why the lower CI bound dropped to +0.18

Because Pearson aggregates across folds with very different N (38 each)
and one fold with constant predictions (NaN), the bootstrap CI is wider
than typical. With Path A (real hardened target) the variance should
reduce, possibly tightening the CI into the "strong" regime.

## Recalibrated impact expectation

Earlier (pre-data) I projected +$200-500/day. Based on Phase 1B's signal
strength:

- Pearson 0.25 → maybe 25% of theoretical max in size-modulated PnL
- On hardened floor of ~$600-700/day, a 15-30% conditional lift = **+$90-210/day**
- Plus 1-3 bad days/year filtered = $2-5K additional via tail-risk reduction

Still a meaningful improvement, but the user should expect maybe **+15-25%
annualized revenue** (not the +30-70% I projected). Roughly $25-50K/year
on the floor.

The "transformative" framing the user pushed back with **still holds in
direction but the magnitude is more modest than projected.**

## What to do next

**Recommendation: proceed to Path A.** Reasons:

1. The peeky proxy is noisy — a real hardened target may show cleaner
   signal (CI could move into +0.20 strong territory).

2. If Path A reproduces the +0.25 Pearson, that confirms the feature set
   has reliable signal and we can begin DRS deployment planning at a
   conservative +$100-150/day expectation.

3. If Path A shows much WEAKER signal (Pearson drops below CI lower
   bound), that's a useful negative — abandon the DRS hypothesis and
   look elsewhere (probably the LLM news feature for one specific
   high-value day type).

## Files

- Input: `DATA/CROSS_DAY/cross_day_features.parquet`
- Input: `DATA/CROSS_DAY/day_pnl_proxy.parquet`
- Output: `reports/findings/drs/2026-05-17_feasibility_gbm.txt`
- Output: `reports/findings/drs/2026-05-17_feasibility_gbm_preds.csv`
- Path A infrastructure ready: `tools/sourcing/DRS_PATH_A.md`
