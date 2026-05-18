---
name: bayesian-archetypes-pending
description: Layer 3 of the regret-oracle arc — Bayesian Trade Archetypes via N-D Trajectory Clustering. Protocol LOCKED 2026-05-16 across 8 topics, build PENDING (5 phases). Spec at research/bayesian_archetypes/project.md.
metadata:
  type: project
---

**Status as of 2026-05-16 evening: protocol LOCKED, build PENDING.**

Full DMAIC spec: `research/bayesian_archetypes/project.md`.

## Locked decisions (8 topics, in discussion order)

1. **Clustering strategy**: peel extremes first (greedy iterative).
2. **Seed criterion**: highest remaining `mfe_velocity` ($ per minute of the
   trade's own outcome). Always positive, lookahead-OK for offline clustering.
3. **Feature pool**: ALL ~190 V2 features at
   `DATA/ATLAS/FEATURES_5s_v2/L{1,2,3}_{5s,15s,1m,5m,15m,1h,4h,1D}/`, joined
   per-bar over each trade's duration. (User flagged my initial ~19-feature
   listing as too narrow.)
4. **Trade signature**: PCA line in 190-D space (centroid + unit direction
   from SVD of the centered trajectory matrix). Per-feature 5%-of-seed-value
   matching was REJECTED because of zero-crossing degeneracy on signed
   features.
5. **Similarity criterion**: perpendicular distance from candidate trajectory
   points to seed's PCA line in z-scored units. Match if all (or quantile of)
   points within radius `r`.
6. **Hierarchical r-ladder**: coarse → fine (e.g., 0.5σ → 0.25σ → 0.125σ).
   Each peeled seed produces a TREE of nested clusters; Bayesian table is
   hierarchical per direction.
7. **Trade decay tracking** (free side-effect of trajectory storage):
   `d(t)` = time series of distance from trade's own trajectory points to
   its matched cluster's PCA line. Three patterns: stable / decaying /
   converging. This UNIFIES: entry classifier + exit signal + duration
   prediction + Bayesian online posterior update for direction.
8. **CUDA build** via PyTorch (batched SVD), matching codebase's CUDA-only
   convention.

**Level-1 stratification: direction (LONG/SHORT)** — clustering runs
independently within each pool.

## Deferred open questions (deliberately not answered)

1. **Decay metric definition**: raw r-distance `d(t)` vs normalized
   `d(t) / r_fine` (where 0=on-line, 1=fine edge, 2=medium edge, 4=coarse
   edge). DECIDE after Phase 4 IS sanity check visualizes actual decay
   curves.
2. **"Decayed" threshold**: hard-cross of `r_coarse` (binary state change)
   vs sustained positive slope of `d(t)` over K bars (gradual). DECIDE at
   Phase 5 when building live evaluator.
3. **Live applicability path**: lead-in-trajectory matching (cluster on past
   N bars too, correlate to forward archetype) vs single-bar entry-match
   with decay-update as bars arrive. DECIDE at Phase 5.

These do NOT block clustering itself; they block live deployment.

## Build phases (pending)

- **P1**: `tools/regret_join_v2_features.py` — daisy-chain CSV × V2 features
   joined per-bar over each trade's duration. Output: per-trade torch tensor.
- **P2**: `tools/regret_trade_signatures.py` (CUDA) — per-trade z-scoring +
   PCA SVD → save (centroid, direction unit-vec, magnitude, per-bar distance
   series). Edge: trades with T_bars < 5 are PCA-unstable; flag/skip.
- **P3**: `tools/regret_bayesian_table.py` (CUDA) — peel + hierarchical
   r-ladder. CLI parameters: `--r-coarse`, `--r-levels`, `--seed-feature`,
   `--direction-stratify`, `--min-cluster-size`, `--min-pool`,
   `--max-iterations`, `--remove-on-peel`. Outputs: hierarchical Bayesian
   table CSV + cluster assignments + decay curves npz.
- **P4**: IS sanity check on first 5 clusters per direction. Are they
   recognizable archetypes? Cluster-size distribution sane? Decay curve
   patterns visible?
- **P5**: `tools/regret_bayesian_live_eval.py` — match live bars to clusters,
   aggregate to Day WR + mode $/day per CLAUDE.md protocol, OOS-validate
   on 2026.

See [[regret-six-layer-architecture]], [[signed-mfe-pivot]],
[[kway-r2-saturation]].
