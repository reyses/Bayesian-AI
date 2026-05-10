---
name: Templates / Bayesian table — the engine was right, the KEY was wrong (2026-05-09)
description: Existing hierarchical Bayesian table at training_iso_v2/bayesian_table.py is architecturally correct; the failure was keying by day-aggregate regime label (biased). V0 swaps the key to primitive chord and reuses the engine.
type: project
---

**Discovery 2026-05-09 evening, after architecting V0 from scratch:**

The exploratory work today (event-segmentation, 15m CRM characterization,
oracle-driven failure-mode framing, meta-router architecture, statistical
labels) made me think we were inventing a new system. We weren't.

`training_iso_v2/bayesian_table.py` already implements:

- Per-cell posteriors:
    WR (Beta), EV/$/trade (Normal), peak_$ (Normal), MAE_$ (Normal),
    time-to-peak (Normal), capture-ratio (Beta)
- Hierarchical shrinkage: cell -> tier-only -> universal (Empirical
  Bayes; thin cells borrow from parent).
- Built offline from regret labels (oracle-derived per-trade outcomes).
- Used as the substrate for the adaptive exit-threshold optimizer.

This is the ENGINE the V0 work needed. What blocked it from being
useful before was the KEYING:

| | OLD (existing) | NEW (V0) |
|---|---|---|
| key | `regime_idx` (1 of 6 day-aggregate labels) | primitive chord = (slope_q, curv_q, z_close_q, sigma_rank_q, r2adj_q) |
| cardinality | 6 | up to 5*3*5*5*5 = 1,875 |
| bias | label computed from same metrics it segments | bar-level, no day-aggregate, no lookahead |
| parent | tier-only | axis-marginal (5 parents instead of 1) |

The architecture stays. The shrinkage / posteriors / Empirical Bayes
/ outputs all stay. We swap the index, not the engine.

## What V0 build now consists of (≈80% reuse)

1. Add a `chord_index(state) -> tuple` function that produces the
   primitive chord at-bar from the 5 CRM features computed in
   `tools/event_bucket_15m_crm.py`.
2. Modify `bayesian_table.py` keying:
   - Cell key: chord tuple instead of regime_idx
   - Parent key: per-axis marginal (5 parents); cell shrinks toward
     each axis-marginal weighted by axis informativeness
   - Universal: same as before
3. Add a fit() pass that consumes the 5,340 IS macro events with
   their chord at entry, computes per-cell posteriors with shrinkage.
4. Add live-time lookup function chord_lookup(chord_index) ->
   {P_cascade, P_continuation_60m, expected_max_z, expected_duration,
    per_tier_oracle_$/trade}.
5. (NEW) Compute P_cascade per cell from oracle event resolution
   (e.g., duration > 30min OR max_z >= 4) — this is the meta-router
   signal that didn't exist in the old (regime, tier) table.

## What this means for the session work

- The exploratory work today wasn't wasted; it identified the missing
  primitive substrate (the 5 CRM-derived axes) that the engine needed.
- The architectural locks (filters vs tiers, statistical labels,
  meta-router two-level decision) all map cleanly onto extending
  `bayesian_table.py`.
- We're not building a new system. We're swapping the cell-key in
  an existing system that already has the math right.

## Connections / lineage

- `core/bayesian_brain.py` (legacy) — predecessor probability table
- `training_v2/bayesian_table.py` — V2 variant (regime keying)
- `training_iso_v2/bayesian_table.py` — current canonical implementation
- `training_iso_v2/regret.py` — produces the per-trade oracle outcomes
  that the table is fit on
- `tools/event_bucket_15m_crm.py` — produces the new chord substrate
- `DATA/ATLAS/regime_labels_2d.csv` — DEPRECATED as a key (biased)

## Why the user's pattern brain was right

Pre-2026-05-09 attempts to use templates / probability tables failed
not because the math was wrong but because the cells were too coarse
(6 day-aggregate regimes) AND the cells were biased (regime label
computed from same metrics that determine outcomes). The exploratory
work today produced the unbiased, fine-grained substrate the engine
needed. Templates ARE the right architecture; we just had to do the
substrate work first.
