# DRS Canonical (Path A) — IS-OOS Gap Verdict

**Date**: 2026-05-18
**Status**: DO NOT DEPLOY — IS-OOS generalization gap is too large to justify production use.

## TL;DR

Path A canonical run produced honest `target_day_pnl` for 217 IS days
(ATLAS) + 23 OOS days (NT8). Refit DRS GBM on this real target.

- **IS walk-forward Pearson**: +0.191, 95% CI [+0.098, +0.405] — weak but
  strictly positive
- **OOS sealed Pearson**: +0.139, 95% CI **[-0.047, +0.451]** — CROSSES ZERO
- **Naive `clamp(pred/IS_mean, 0.5, 1.5)` sizing rule on OOS**:
  **-$333/day, CI [-$523, -$163]** — actively destroys value
- **Rank-based sizing rules on OOS**: ALL non-significant, point
  estimates -$20 to -$67/day

The DRS hypothesis is not killed but it is NOT deployable on current
features. The 23-day OOS sample is small but the unanimous direction
(all rules → slight negative or non-significant) is informative.

## Reproducing

```bash
# Path A: build honest target (~15-30 min total)
python tools/sourcing/drs_a_step1_predict_b_models_is.py
python tools/sourcing/drs_a_step2_build_cloud_is.py
python tools/sourcing/drs_a_step3_forward_pass_is.py
python tools/sourcing/drs_a_step4_aggregate_day_pnl.py

# DRS GBM on real target
python tools/sourcing/drs_canonical_gbm.py
python tools/sourcing/drs_rank_sizing_eval.py
```

Outputs:
- `DATA/CROSS_DAY/cross_day_features_with_target.parquet` — 240 days with real target
- `DATA/CROSS_DAY/drs_canonical_gbm.pkl` — full-IS production model
- `reports/findings/drs/2026-05-18_drs_canonical_gbm.txt` — feasibility result + IS WF
- `reports/findings/drs/2026-05-18_drs_rank_sizing.{txt,csv}` — rank rule grid

## Detailed results

### IS walk-forward (5 folds, 205 days after NaN-drop)

| Fold | n_train | n_test | Pearson | MAE |
|------|---------|--------|---------|-----|
| 1 | 34 | 34 | NaN (constant preds) | $3,380 |
| 2 | 68 | 34 | +0.302 | $1,183 |
| 3 | 102 | 34 | +0.520 | $731 |
| 4 | 136 | 34 | +0.725 | $925 |
| 5 | 170 | 34 | +0.570 | $1,180 |
| **Aggregate** | | | **+0.191** | **$1,480** |

Persistence baseline MAE: $2,139. DRS MAE lift: **+30.8%**.

### Rank-based sizing on IS WF (proper fold-internal ranking)

| Rule | LO | HI | mean Δ/day | 95% CI | sig |
|------|----|----|-----------:|--------|-----|
| rank_0.7_1.3 | 0.70 | 1.30 | **+$249** | [+$113, +$424] | **YES** |
| rank_0.5_1.5 | 0.50 | 1.50 | +$415 | [+$189, +$706] | YES |
| rank_0.3_1.7 | 0.30 | 1.70 | +$582 | [+$264, +$988] | YES |
| rank_0.0_2.0 | 0.00 | 2.00 | +$831 | [+$378, +$1,412] | YES |

All 4 rules significant on IS. Effect scales with aggressiveness.

### Sealed OOS single-shot test (23 NT8 days, 2026-03-20 to 2026-04-26)

| Rule | LO | HI | mean Δ/day | 95% CI | sig |
|------|----|----|-----------:|--------|-----|
| rank_0.7_1.3 | 0.70 | 1.30 | -$20 | [-$68, +$34] | no |
| rank_0.5_1.5 | 0.50 | 1.50 | -$34 | [-$113, +$57] | no |
| rank_0.3_1.7 | 0.30 | 1.70 | -$47 | [-$158, +$79] | no |
| rank_0.0_2.0 | 0.00 | 2.00 | -$67 | [-$226, +$113] | no |

ALL OOS rules: non-significant, point estimates slightly NEGATIVE,
upper CI bounds always positive.

### The 23-day OOS picture

23 NT8 OOS days (gbm_ev hardened):
- Mean: $+1,086/day
- Median: $+767
- Negative days: 2 of 23 (8.7%)
- DRS predictions: mean $+1,103 (close to actual mean)
- BUT on the 2 negative days, DRS predicted **HIGHER** than positive days
  (pred mean $+1,466 vs $+1,069 — anti-predictive)

The model has signal at the day-quality LEVEL but is misranking the
catastrophic tail. This is the same failure mode as direction prediction
on V2 features: average accuracy is fine, extreme cases are unreliable.

### Feature importance (permutation, last IS fold)

| Feature | ΔMAE |
|---------|-----:|
| overnight_gap_pct | +$82 |
| prior_day_range_pct | +$78 |
| days_since_fomc | +$62 |
| prior_day_c2c_pct | +$23 |
| overnight_range_pct | +$18 |
| vix_close_prior | +$17 |
| is_fomc | $0 |
| is_cpi | $0 |
| is_nfp | $0 |
| is_opex | $0 |
| dxy_close_prior | -$1 |
| dxy_chg_prior | -$4 |
| days_to_next_fomc | -$28 |
| vix_chg_prior | -$33 |
| dow | -$51 |

Overnight gap + prior day range + days_since_fomc are the only durably
useful features. VIX changes, DXY, day-of-week, and the day-of-event
binaries all hurt or contribute nothing.

## Why IS works and OOS doesn't

Three hypotheses (data can't distinguish on N=23 OOS):

1. **Regime change**: IS = 2025 calendar year (Q1-Q4); OOS = 2026 Mar-Apr.
   Different vol regime, different Fed cycle position, possibly different
   correlation structure between cross-day features and intraday outcome.

2. **OOS sample too small**: 23 days is barely enough for a CI to be wide
   and uninformative. With 100+ OOS days the CI might tighten enough to
   distinguish signal from noise.

3. **Feature staleness**: VIX/DXY/calendar features perform best when the
   model trains on data with similar joint distribution. The 2026 macro
   regime may have shifted enough (post Fed-pivot, tariff news, etc.) that
   the IS-learned mapping doesn't transfer.

## Decision: do NOT deploy DRS v1

Per anti-doom-cascade rule from CLAUDE.md: report multiple gap
assumptions. Here:

- If true delta = IS WF point (+$249/day): deploying recovers most of
  the foregone alpha. BUT we have to believe IS signal generalizes.
- If true delta = OOS point (-$20 to -$67/day): deploying destroys value.
- If true delta = 0: deploying does nothing and adds operational risk.

The OOS evidence STRONGLY suggests the IS signal does not generalize.
Burden of proof is on DRS to clear OOS bar before deployment.

## Recommended next steps (per handoff)

1. **Phase 2 — LLM news features**: targeted retrieval of FOMC/CPI/NFP
   headline risk score. The current binary is_fomc/is_cpi/is_nfp features
   have ZERO permutation importance because they're sparse and don't
   capture HEADLINE INTENSITY. An LLM-scored intensity might be the
   orthogonal feature DRS needs.

2. **Collect more OOS data**: wait until 60-90 more NT8 days accumulate
   from sim or live trading, then re-run the sealed OOS test. If Pearson
   tightens above zero with positive CI, DRS comes back to life.

3. **DRS-as-VETO not multiplier**: the OOS catastrophic-day analysis
   showed DRS is ANTI-PREDICTIVE on the 2 negative days. So even the
   "skip bad days" framing fails — we'd skip wrong days. Consider a
   different veto mechanism (e.g., is_fomc + vol_regime composite hard
   rule instead of learned model).

4. **Pivot to other layers**: the during-trade L5 paradigm (B9) shipped
   yesterday with +$67/day OOS CI [+$32, +$106]. Continue building L5
   variants (with new action surfaces, NOT cut/preempt). DRS as a
   day-level layer remains future work.

## Files

- Path A target: `DATA/CROSS_DAY/cross_day_features_with_target.parquet`
- Production model: `DATA/CROSS_DAY/drs_canonical_gbm.pkl` (kept as
  research artifact, NOT for deployment)
- Tools: `tools/sourcing/drs_canonical_gbm.py`,
  `tools/sourcing/drs_rank_sizing_eval.py`
- Reports: `reports/findings/drs/2026-05-18_drs_canonical_gbm.txt`,
  `reports/findings/drs/2026-05-18_drs_rank_sizing.txt`
- Handoff: `reports/findings/drs/2026-05-18_handoff.md`
- Feasibility (Phase 1B): `reports/findings/drs/2026-05-17_feasibility_verdict.md`
