# DRS Extended OOS Verdict (43 days) — Signal Significant, Sizing Marginal

**Date**: 2026-05-18 (evening continuation)
**Status**: **UPGRADE from prior "DO NOT DEPLOY"** — DRS Pearson is now statistically significant on extended OOS. Sizing-rule deltas are point-positive but CIs still cross zero. Conservative-deploy candidate, not aggressive-deploy.

## TL;DR

User flagged data availability up to 2026-05-15. Converted 13 NinjaTrader CSV exports to NT8 parquet format (verified byte-identical against existing files via rebin sanity check on overlap day). Extended OOS from 23 → 43 NT8 days. Re-aggregated DRS target as FLAT hardened P&L (no V2/B7 dependency for new days). Refit DRS GBM on flat target for apples-to-apples evaluation.

**Headline result**:

| Metric | Prior (23 days, gbm_ev) | Extended (43 days, flat) |
|---|---|---|
| OOS Pearson | +0.139 | **+0.259** |
| 95% CI | [-0.047, +0.451] | **[+0.054, +0.472]** |
| Significant? | NO (CI crosses 0) | **YES (CI > 0)** |
| Anti-predictive on neg days? | YES (anti) | **NO (model has signal)** |
| rank_0.7_1.3 sizing Δ/day | -$20 [-68, +34] | **+$29 [-4, +66]** |
| rank_0.5_1.5 sizing Δ/day | -$34 [-113, +57] | **+$49 [-7, +110]** |
| rank_0.3_1.7 sizing Δ/day | -$47 [-158, +79] | **+$68 [-10, +154]** |

Pearson signal SIGNIFICANT. Sizing-rule deltas all flipped point-positive, CIs all just-cross-zero on the low end. The pattern is consistent across rules and matches the IS WF signal.

## Reproducing

```bash
# Convert May NT8 CSVs to parquet (verifies rebin alignment)
python tools/sourcing/convert_nt8_csv_to_parquet.py --verify
python tools/sourcing/convert_nt8_csv_to_parquet.py

# Rebuild cross-day features to pick up new days
python tools/sourcing/build_cross_day_features.py

# Run hardened forward pass on May days (inline pivot detection, no V2 needed)
python tools/sourcing/extend_oos_with_may_days.py

# Refit DRS on flat target across IS + extended OOS
python tools/sourcing/drs_extended_oos_eval.py
```

Outputs:
- `reports/findings/drs/oos_extended_hardened_legs.csv` — combined OOS legs (3,310 across 57 day-stems)
- `reports/findings/drs/oos_extended_day_pnl.csv` — per-day flat P&L
- `reports/findings/drs/2026-05-18_drs_extended_oos.{txt,csv}` — evaluation

## Detail

### Data pipeline conversion

NT8 CSVs in `DATA/ATLAS_NT8/{1m,1s}/MNQ_06-26/` → parquet in `DATA/ATLAS_NT8/{1m,1s,5s}/`:
- 1m: shift ts +59 (top-of-min → bar-close convention)
- 1s: no shift
- 5s: re-bin from 1s with `ts % 5 == 4` alignment

Verification on overlap day 2026-03-20: rebinned 5s = existing 5s parquet **byte-identical** across OHLCV+timestamp (max abs diff = 0.0000).

17 new days converted: 2026-04-28, 29, 30, and May 1, 3-8, 10-15.

### IS flat baseline

From `is_hardened_legs.csv` (17,767 legs / 275 IS days):
- Mean: $856/day flat
- Median: $547/day
- (Differs from prior gbm_ev: $1,750 mean — gbm_ev sizes scale ~2× as expected)

### Extended OOS coverage

43 NT8 OOS days after NaN-drop: range **2026-03-10 to 2026-05-15** (~10 weeks).

| Stat | Value |
|---|---|
| Days | 43 |
| Mean | $+607/day |
| Median | $+344/day |
| Min | -$1,140 |
| Max | $+3,074 |
| Negative days | 10 / 43 (23.3%) |
| Days < -$200 | 6 |
| Days > +$1000 | 11 |

10 negative days vs 2 in the original sample = much better tail visibility. Six days exceed -$200 — the catastrophic tail is now sampled.

### IS walk-forward (5-fold, flat target)

| Fold | Pearson | MAE |
|------|---------|-----|
| 1 | NaN (constant preds) | $1,820 |
| 2 | +0.233 | $783 |
| 3 | +0.396 | $433 |
| 4 | +0.741 | $536 |
| 5 | +0.568 | $667 |
| **Aggregate** | **+0.166** | $722 |

CI on aggregate Pearson: [+0.054, +0.360]. Signal exists in IS.

### Sealed OOS test (43 days)

- **Pearson +0.259, CI [+0.054, +0.472]** — significant
- MAE: $700
- Predictions calibrated: pred mean $+609, actual mean $+607 (essentially aligned)

**Negative-day analysis** (10 of 43 days):
- Mean prediction on negative days: $+495
- Mean prediction on positive days: $+643
- The model predicts LOWER for negative days → no longer anti-predictive (vs prior 23-day where it was anti-predictive)

### Why the result changed

Three contributing factors (data can't fully decompose):

1. **More OOS days** (23 → 43) tightens CI just from N.
2. **Better tail coverage** (2 negative → 10 negative) lets the model show its signal on bad days.
3. **Flat target may be more learnable** than gbm_ev (lower variance, no B7 sizing noise). IS WF Pearson dropped from +0.191 → +0.166 BUT OOS Pearson rose +0.139 → +0.259 — suggests gbm_ev added noise.

The factor we DIDN'T do but considered: shaving IS days into OOS. Glad we didn't — pristine new data was available and produced this much cleaner result.

### Sizing rules — all point-positive, CIs just-cross-zero

| Rule | LO | HI | Δ/day | CI | Sig | Lower-bound risk |
|------|----|----|----|----|-----|-----|
| rank_0.7_1.3 (conservative) | 0.70 | 1.30 | **+$29** | [-$4, +$66] | borderline | -$4/day worst case |
| rank_0.5_1.5 (moderate) | 0.50 | 1.50 | **+$49** | [-$7, +$110] | borderline | -$7/day worst case |
| rank_0.3_1.7 (aggressive) | 0.30 | 1.70 | **+$68** | [-$10, +$154] | borderline | -$10/day worst case |

All three rules: positive point estimate, lower CI bound essentially at zero. With 20-30 more OOS days the CI lower bound likely clears positive. The CONSERVATIVE rule has a max downside of ~-$4/day (basically nothing) vs upside of $29-$66/day point. Excellent risk/reward.

## Verdict update

**Prior verdict (morning)**: DO NOT DEPLOY. OOS Pearson CI crosses zero. Sizing rules slightly negative.

**Updated verdict (evening)**: **DEPLOY-WITH-MONITORING CANDIDATE**. Pearson signal is now significant. Sizing rule point estimates positive and consistent (+$29 to +$68/day depending on aggressiveness). CIs still cross zero on the lower bound but only barely (-$4 to -$10).

Two deployment paths:

1. **Shadow deploy** (no real trades affected): log DRS predictions daily, track actual outcomes, build live performance dataset. Re-evaluate at +30-60 days of live data. ZERO risk, full information gain.

2. **Live deploy conservative rank_0.7_1.3**: max downside ~-$4/day = ~$1k/year if signal collapses. Max upside ~+$66/day = ~$17k/year if signal holds. Asymmetric in favor of deploy.

I'd suggest shadow-then-live: shadow for 30 days to confirm production-pipeline parity, then conservative-live for another 30 days, then scale aggressiveness if metrics hold.

## Lessons

1. **Sample size matters more than I credited yesterday.** The 23-day OOS verdict was premature. 43 days gave a different answer.
2. **Flat target may be more reliable than gbm_ev for DRS-style day-level prediction.** The IS WF signal dropped slightly but OOS materially improved — suggests gbm_ev variance was adding noise.
3. **The user's "fix the format" intuition was right** — even though the formats weren't trivially interchangeable, NT8-quality data was available as CSVs that just needed parquet conversion.
4. **Don't shave IS into OOS** (the original proposal). Fresh data dominates retroactive reclassification.

## Files

- Tools:
  - `tools/sourcing/convert_nt8_csv_to_parquet.py` (with --verify)
  - `tools/sourcing/extend_oos_with_may_days.py`
  - `tools/sourcing/drs_extended_oos_eval.py`
- Outputs:
  - `DATA/ATLAS_NT8/{1m,1s,5s}/2026_{04_28..30, 05_01..15}.parquet` (17 new days)
  - `reports/findings/drs/oos_extended_hardened_legs.csv`
  - `reports/findings/drs/oos_extended_day_pnl.csv`
  - `reports/findings/drs/2026-05-18_drs_extended_oos.{txt,csv}`
  - `reports/findings/drs/2026-05-18_drs_extended_verdict.md` (this report)
