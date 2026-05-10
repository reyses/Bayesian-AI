---
name: Outlier-day dominates total-PnL optimizer
description: 2026-05-04 lesson — total-PnL grid search hides one freak day's lottery payoff as a "+$713/day OOS uplift". Always bootstrap-CI the delta and use median-day or trimmed-mean objective.
type: feedback
originSessionId: e1202bdb-1787-4774-b48b-b312af212043
---
## The rule

When optimizing exit-threshold parameters on regret-replayed paths, **NEVER
maximize total summed PnL across all paths**. Single-day tail events
(macro/news days with $1k+/contract ranges) dominate the sum and produce
threshold combos that look great in IS but only "work" because the same
freak day exists in OOS too.

Use **median daily PnL** or **trimmed mean of daily PnL** as the objective
instead. They're robust to single-day outliers and reveal the real signal.

**Why:** During the 2026-05-04 V2-native pipeline run, the grid-search
optimizer (objective='total') produced an OOS result of **+$713.26/day**
across 68 days. Bootstrap 95% CI on the delta vs baseline: **[-$3.97,
+$2208.64]** — NOT statistically significant.

Forensics revealed:
- **97% of the uplift came from ONE day: 2026-03-20**
- That day: 2026-03-20 had a $666 range ($1,333/contract = ~2666 ticks)
- VEL_BODY_CHORD with `tp_pts=15` ($30 TP) hit TP repeatedly = lottery payoff
  of +$49,007 in one session
- The other 67 OOS days NET to ~-$500 total
- Top-1 day concentration = 97% (anything > 30% is suspect)

The same OOS-overfit pattern appeared again with the Bayesian regime-only
formulas (+$716/day OOS, 97% from same 2026-03-20). Threshold derivation
methodology didn't matter; *strategy selection* did. VEL_BODY_CHORD on 67/68
OOS days was negative; on 2026-03-20 it printed.

## How to apply

1. **Default objective**: `--objective median_day` in
   `training_v2/threshold_optimizer.py`. (Or use `threshold_bayesian.py`
   which never grid-searches.)
2. **ALWAYS bootstrap-CI the OOS delta** vs baseline before claiming uplift.
   Use 4,000 paired bootstrap resamples on per-day PnL deltas. CI [a, b]:
   if a > 0, significant; if a ≤ 0 ≤ b, not significant.
3. **Top-1 day concentration check**: if the top single OOS day contributes
   >30% of total uplift, the result is dominated by tail events — investigate
   that day's market action and decide whether the strategy generalizes.
4. **Verdict on VEL_BODY_CHORD (2026-05-04)**: KILLED PERMANENTLY. It's a
   lottery-day artifact, not a real strategy. Removed from
   `training_v2/run.py` default strategy list.

## Anti-pattern

```
# Bad: total summed PnL on simulated paths
optimize_cell(labels, objective='total')  # picks IS-best argmax — overfits

# OK: median daily PnL
optimize_cell(labels, objective='median_day')  # tail-event resistant

# Best: derive from cell distribution moments — no search at all
threshold_bayesian.derive_thresholds(labels, q_tp=0.30, q_sl=0.70, ttp_factor=1.5)
```
