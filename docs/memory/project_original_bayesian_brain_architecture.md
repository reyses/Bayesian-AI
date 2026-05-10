---
name: Original Bayesian Brain architecture (Feb 2026, deleted/superseded — substrate was wrong, design was right)
description: 7-step pipeline scanning IS to build trade templates, K-means distill to canonical set, mount on Bayesian brain, track via oracle + regret, fire via 9-gate cascade. The architecture worked; templates were "obscured / multi-dimensional concepts" which was the red flag.
type: project
---

**Documented 2026-05-09 evening from user description.**
**Source files were deleted in commit 23db222f ("chore: delete dead engine code")**
but the architecture is the basis for the V0 build. Reviving the design with
corrected substrate (primitive chord instead of K-means'd multi-D template).

## The original 8-step pipeline

```
1. SCAN IS              walk in-sample data bar by bar
2. BUILD TEMPLATES      extract trade-pattern signatures around each entry
3. K-MEANS DISTILL      cluster template signatures into a smaller canonical set
4. SCRUB                clean noisy / low-confidence clusters
5. MOUNT ON BRAIN       load distilled templates as Bayesian-table keys;
                        each (template, direction) cell holds win/loss + $/trade
6. ORACLE + REGRET      post-hoc: did the trade fire the right way?
                        regret = counterfactual analysis on each decision
                                 (what would have happened with other actions)
7. 9-GATE CASCADE       at each candidate entry, 9 ordered gates each gates
                        the trade; only signals that pass all 9 gates fire
8. OOS VALIDATION       carry the IS-fit brain forward on out-of-sample data,
                        re-run the 9-gate cascade against the brain's posteriors,
                        measure how well the IS-derived edge generalized
                        (this is the "did the substrate hold" check that any
                        IS-fit table requires before deployment)
```

## Why it failed (the red flag, in hindsight)

The "obscured / multi-dimensional concepts" were the templates themselves.
Step 3 (K-means on multi-dimensional template signatures) collapsed distinct
real patterns into clusters whose CENTROIDS no longer represented the
patterns clearly.

Concretely:
- Templates were built from many features at once (a 16D, 23D, or 79D
  feature vector at the entry bar)
- K-means partitioned those vectors into K buckets
- The centroid of a K-means bucket is not a recognizable pattern — it's
  an averaged blob of all the patterns that happened to land near each
  other in 16D / 23D / 79D space
- Two physically different setups could land in the same bucket (false
  merge) or one setup could split across buckets (false split)
- Bayesian table keyed on bucket id therefore averaged unrelated trades
- Outcome estimates were noisy; gate-cascade signals fired on patterns
  that didn't actually exist as coherent regimes

This is the substrate problem. The architecture was right; the keys were
abstract enough that they no longer corresponded to discoverable patterns.

## Why the V0 build (chord-keying) corrects this

The 5 primitive chord axes from `tools/event_bucket_15m_crm.py` are NOT
abstract:

  slope_q       (5 bins)   sign + magnitude of 1h-lookback slope of M_close_15m
  curvature_q   (3 bins)   sign + magnitude of slope-of-slope
  z_close_q     (5 bins)   (5s_close − M_close_15m) / SE_close_15m
  sigma_rank_q  (5 bins)   rolling 60min percentile of SE_close_15m
  r2adj_q       (5 bins)   R² adjusted of 5min linear fit to 5s closes

Each axis is a single statistical measurement with a clear physical
interpretation. The chord (slope_q3, curv_q1, z_q4, sigma_q2, r2adj_q5)
describes a SPECIFIC, recoverable pattern: 'no trend, decelerating curve,
price slightly above mean, low band-width, smooth/predictable'.

There is no centroid averaging. Two events with the same chord WERE
empirically in the same statistical phase. Bayesian table keyed on chord
therefore aggregates trades that genuinely share context.

Up to 5×3×5×5×5 = 1,875 cells from 5,340 IS macro events ≈ mean 3
events/cell — needs hierarchical shrinkage (cell → axis-marginal →
universal), which is already implemented in
`training_iso_v2/bayesian_table.py`.

## What survived from the original architecture

Almost everything except the K-means step:

| step in original         | mapping in V0                                                                  |
|--------------------------|--------------------------------------------------------------------------------|
| 1. scan IS               | same — walk 5,340 IS macro events                                              |
| 2. build templates       | replace with: extract 5-axis chord at event entry                              |
| 3. K-means distill       | DROP — quantile bucketing already produces canonical cells                     |
| 4. scrub                 | replace with: hierarchical shrinkage of thin cells toward parent               |
| 5. mount on brain        | extend `training_iso_v2/bayesian_table.py` keying                              |
| 6. oracle + regret       | reuse — `training_iso_v2/regret.py` already produces per-event labels          |
| 7. 9-gate cascade        | maps to: meta-router + per-tier filter cascade (the 2026-05-09 architectural lock) |
| 8. OOS validation        | OOS sign-stability per axis + per-cell P(_) divergence test; re-run meta-router and tier-filter cascade on the OOS macro-event population (already in todo list as the validation pass) |

The 9-gate cascade became the 9 ExNMP tiers (FADE_CALM, FADE_MOMENTUM,
RIDE_CALM, RIDE_MOMENTUM, FADE_AGAINST, RIDE_AGAINST, KILL_SHOT, CASCADE,
FREIGHT_TRAIN — see `training_iso_v2/strategies/`). Each tier IS a
parameterized gate. The new meta-router (Level 1, P_cascade-based)
selects which tiers are even eligible, then per-tier filters (Level 2)
gate firing.

## What this means for the user's pattern recognition

The user's instinct ('original template work will work but not in the
context we were doing') was correct. The exploratory work today produced:

  - Statistical primitive axes that don't average across distinct patterns
  - Event-segmented bucketing on 1h HL ±3σ macro events (3,431 IS)
  - Oracle-driven failure-mode framing
  - Meta-router architecture that maps to the original 9-gate role
  - Hierarchical shrinkage that replaces the K-means distill step

All of which corrects the SUBSTRATE problem that killed the original brain
without changing the architectural design.

## Files / commits to reference

- Feb 2026 brain (deleted): commit befdc2df modified `core/bayesian_brain.py`
  which tracked `(StateVector, template_id, direction) -> {wins, losses, $/trade}`
- Feb 2026 templates: commits 8232a1c6 (TF-bucketed clustering), 82fc8478
  (counter-trend template analysis), 4f9334c6 (trade duration fractal analysis)
- Deletion: commit 23db222f ('chore: delete dead engine code -- core/ down to 5 files')
- Modern Bayesian table: `training_iso_v2/bayesian_table.py` (regime keying — to be replaced by chord)
- Modern regret: `training_iso_v2/regret.py` (oracle + counterfactual labels)
