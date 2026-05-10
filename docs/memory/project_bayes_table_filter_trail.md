---
name: Bayesian table = filter + trail stop for the 9 ExNMP tiers
description: Architectural lock 2026-05-10 — per-cell direction/duration/magnitude/decay tables become context filter + adaptive trail for all 9 tiers
type: project
---

# BAYESIAN TABLE → FILTER + TRAIL STOP FOR 9 TIERS

## Architectural lock (2026-05-10)

User: "and this becomes the filter and trailstop of the 9 tiers"

## 3-STATE simplification (2026-05-10, refining the lock)

User: "5s CRM has only 3 possible actions continue direction, flatline or
reverse, out of these 3 only one is adverse"

At any 5s bar the CRM has 3 next-state outcomes:
  CONTINUE  — same slope sign         → favorable to current position
  FLATLINE  — slope ≈ 0               → neutral, no PnL change
  REVERSE   — slope flips             → ONLY ADVERSE outcome

This collapses the filter+trail problem to a single per-cell statistic:
  P(reverse | current_cell, t_in_trade)

The duration_per_axis.csv table IS this survival function:
  P(reverse by t) = 1 - P(duration >= t)

The decay_per_axis.csv peak_horizon column = bar to start tightening trail.

No magnitude branching needed — REVERSE is the only adverse outcome to
defend against. Magnitude tables become input to position sizing, not exit
logic.

Structural EV note: with only 1/3 outcomes adverse, even uniform odds
give break-even. The Bayesian table tilts cells away from uniform — most
cells have P(continue) > P(reverse) by a margin that the table quantifies.

The Bayesian probability table built 2026-05-09→10 (per-primitive-cell direction
+ duration + magnitude + decay) is NOT a standalone strategy. It is:

1. **CONTEXT FILTER** — gates when each of the 9 ExNMP tiers fires
2. **ADAPTIVE TRAIL/STOP** — calibrates exits per cell

The 9 tiers (FADE_CALM/MOMENTUM/AGAINST, RIDE_CALM/MOMENTUM/AGAINST,
KILL_SHOT, CASCADE, FREIGHT_TRAIN) keep their existing entry rules. Each
gets the table as a wrapper that consults cell statistics at entry and
during the trade lifecycle.

## Composition (per 2026-05-09 context-filter-vs-tier lock)

```
Strategy = TIER_entry AND f1(state) AND f2(state) AND filter_bayes(at_bar_cell)
                                                                      AND
trail_bayes(cell, t_in_trade, current_PnL)
```

Multiplicative gate. The Bayesian filter is one of f_i. The trail logic
replaces (or overlays on) existing fixed-dollar trail stops.

## Filter logic

At each tier entry candidate:
1. Compute at-bar primitives (slope_q, curv_q, z_close_q, sigma_rank_q, r2adj_q)
   from `tools/event_bucket_15m_crm.py:_compute_day_crm_features` — this is
   already lookahead-clean and reusable.
2. Look up cell in `BLEED_tier_x_sub_motif_x_measure.csv` for THIS tier.
   If cell is in the bleed-set: SKIP.
3. Look up cell in `p_up_per_axis.csv` for direction prior. If direction
   prior opposes the tier's intended direction: SKIP.
4. Else: PROCEED with the entry.

## Trail/stop logic

Once tier fires:
1. Initial hard stop: `entry_price ± stop_z * SE_anchor` where stop_z comes
   from `magnitude_per_axis.csv` (q90 of excess past trigger ≈ +2σ for
   most cells per user hypothesis).
2. Adaptive trail: query `decay_per_axis.csv` for cell's peak_horizon and
   decay_to_60m. Two archetypes:
   - SUSTAINED (peak=60m, decay=0): loose trail, ride to time-stop or
     band-mean cross
   - SPIKE_REVERT (peak=5-15m, decay<<0): tighten aggressively past
     peak_horizon, exit when expected remaining PnL approaches zero
3. Exit triggers:
   - hard stop hit
   - peak_horizon reached AND decay sign turning negative
   - price reverts to band-mean (M_close cross)
4. Optional: arm opposite-leg entry on exit (if exit-time cell has stable
   counter-edge in the same table)

## Loop architecture

```
Always-in-market (when both directions have stable cells):
  [tier ENTRY] → [bayes FILTER pass] → IN TRADE
       ↓
  [bayes TRAIL/STOP fires] → EXIT
       ↓
  [exit-bar primitive lookup] → opposite-direction cell stable?
       ↓                              ↓
       YES → arm opposite leg         NO → wait for next tier signal
```

The filter and trail share the same lookup table; the tier provides the
entry signal; the exit's primitive vector becomes the next entry decision.

## Required infrastructure

```
training_iso_v2/filters/                              (new dir)
    bayes_table_lookup.py        at-bar feature compute + cell lookup helper
    bayes_filter.py              SkipFilter wrapping any tier with the table

training_iso_v2/exits.py         (extend with bayes_trail exit class)
    BayesAdaptiveTrail           reads cell decay/magnitude, sets trail
                                  per archetype (sustained / spike_revert)

training_iso_v2/strategies/<each_tier>.py    (modify each)
    self.bayes_filter = BayesFilter(tier=self.name, table_dir=...)
    on_candidate(state): if not self.bayes_filter.allow(state): return None
    on_trade(state, trade): self.bayes_trail.update(state, trade)
```

## Tables consumed

```
reports/findings/segments/diagnostic_tier_bleed/
    BLEED_tier_x_sub_motif_x_measure.csv     per-tier skip-cell list

reports/findings/segments/bayes_table_v0_location/
    p_up_per_axis.csv                        direction prior per cell
    duration_per_axis.csv                    P(continues N min) per cell
    magnitude_per_axis.csv                   stop_z, target_z per cell
    decay_per_axis.csv                       peak_horizon, decay per cell
    actionable_ride_exit_table.csv           pre-joined fast-lookup
```

All tables keyed on (side, anchor, axis, bin) at min_n=10 IS samples.

## Validation path before deploy

1. Wire into ONE tier first (suggest NMP_FADE_RAW — biggest bleed savings
   $7,970 OOS uplift if filter works as expected)
2. Run full IS+OOS through `training_iso_v2.run_iso --tiers NMP_FADE_RAW`
   with and without bayes filter
3. Compare: $/day, day-WR, max DD, OOS gap from IS
4. If filter produces stable improvement (CI excludes 0): repeat for next
   tier in bleed-savings-rank order:
       NMP_FADE_RAW → FADE_AT_BAND → FADE_MOMENTUM → FADE_CALM → KILL_SHOT
       → CASCADE → FADE_AGAINST → ...
5. RIDE tiers come last — they have fewer bleed cells in current
   diagnostic, so the filter may have less room to help

## Caveats

- Bleed cells were SELECTED on data that includes OOS — even with
  IS<0 AND OOS<0 sign-stability, applying the skip rule to the same OOS
  is mildly optimistic. Honest test = re-fit bleed cells on IS only,
  apply to true held-out OOS. Likely shrinks the savings 20-40%.
- Trail/stop tables also use full IS+OOS. Same caveat for any trail
  parameter learned from these.
- Day-clustering not applied. Cells with n=89 may be 5-10 days; effective
  sample is much smaller.

## Related memories

- `feedback_5s_inherently_noise.md` — chord level used for diagnostic
  attachment, NOT for prediction at note level
- `project_5level_segmentation_substrate.md` — the chord substrate this
  builds on
- `project_context_filter_vs_tier.md` — the architectural rule this implements
- `feedback_quantile_selection_overfit.md` — the warning we keep ignoring
  about quantile-cell selection bias
