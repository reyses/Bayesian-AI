# Project: Bayesian Trade Archetypes via N-D Trajectory Clustering

DMAIC frame. Started 2026-05-16.

## Translation table (borrowed concepts → math)
- **Trade archetype** = a cluster of trades that traced similar paths through N-D feature space during their durations. Math: a (centroid, direction) line in 190-D space via PCA.
- **N-D trajectory** = the sequence of N-D feature vectors `X(t)`, one per bar, during the trade's duration. Math: a (T_bars × 190) matrix per trade.
- **PCA line** = first principal axis of the centered trajectory matrix. Math: the top right-singular vector from `SVD(X − X̄)`, paired with the centroid `X̄`.
- **r-distance** = perpendicular distance from a trajectory point to a cluster's PCA line, in z-scored feature space. Math: `||(p − c) − ((p − c) · d̂) · d̂||` where c=cluster centroid, d̂=unit direction.
- **Decay** = the time series `d(t)` of r-distance from the trade's own trajectory points to the cluster's PCA line, as the trade develops.
- **Peeling** = iterative greedy extraction: pick the most-extreme remaining trade, gather its archetype, remove from pool, repeat.
- **Hierarchical clustering** = at each peeled seed, build nested sub-clusters at decreasing r values (coarse → fine).
- **Live applicability**: clustering is offline (full trajectory known). Live use needs a different matching step — covered in §Improve.

## Define

**The problem.** Yesterday's k-way analysis showed direction is predictable at R²≈0.35 from entry-time features when stratified. Direction-callable cells (Wilson CI excludes 30/70%) cover 30-60% of trades depending on k. The gap: cells are static feature-bin combinations — they describe *the bar at entry*, not the *trade as a whole*. Two trades with identical entry features can play out very differently.

**The hypothesis.** A trade is more than its entry bar — it's a *trajectory through N-D feature space over time*. Trades that traced similar N-D trajectories are the same archetype, regardless of nominal entry-feature differences. Clustering on trajectories should reveal richer archetypes than clustering on entry features alone, and enable:
- Direction prediction (which cluster's archetype does a new bar match?)
- Magnitude prediction (cluster's mean MFE)
- Duration prediction (cluster's decay curve)
- Live exit signals (when does a live trade decay out of its archetype?)

**Inputs.** The daisy-chain regret-oracle CSV (7,925 trades, 2025 full IS) joined to the V2 feature stack at every bar during each trade's duration.

**Builds on**:
- `research/regret_oracle/project.md` — the daisy-chain oracle producing the trade set
- `reports/findings/regret_oracle/2026-05-16_direction_signal_kway.md` — direction-target pivot + k-way saturation
- The V2 feature stack at `DATA/ATLAS/FEATURES_5s_v2/L{0,1,2,3}_{5s..1D}/` (~210 columns, ~190 features after dropping timestamps)

## Measure

**Trade pool** (locked):
- Source: `reports/findings/regret_oracle/daisy_chain_IS_full_daisy.csv`
- 7,925 sequential non-overlapping trades, full 2025 IS, 5s base TF, 60-min window cap
- Includes `signed_mfe = mfe_dollars × (+1 if LONG else −1)` column

**Feature pool** (locked):
- All ~190 V2 features at `DATA/ATLAS/FEATURES_5s_v2/L{1,2,3}_{5s,15s,1m,5m,15m,1h,4h,1D}/`
- Joined to each trade's bar-level trajectory by `oracle_ts` and the duration of the trade
- Per trade matrix shape: (T_bars × 190) where T_bars varies per trade (entry → exit)

**Data prep step** (pending build):
- Tool: `tools/regret_join_v2_features.py` (to be written)
- For each trade in daisy-chain: walk from `entry_idx` to `exit_idx`, pull V2 features per bar, stack → per-trade trajectory tensor
- Save: a single torch tensor pickle keyed by `oracle_idx`

## Analyze — locked methodology

### Stratification: direction (Level 1)
- First split: LONG vs SHORT
- The peel runs independently within each direction
- Reason: direction is the prediction target; we want to characterize "what LONG archetypes look like" vs "what SHORT archetypes look like"

### Seed selection: highest `mfe_velocity` remaining
- `mfe_velocity = mfe_dollars / time_to_mfe_min` — the trade's own rate of development
- Always positive (lookahead-OK for clustering; not needed at inference)
- Rationale: peeling extremes first preserves rare archetypes (rubber-band crashes, big rallies) as their own clusters before they get absorbed into dense average clusters

### Trade signature: PCA line in N-D
- For each trade, build matrix X of shape (T_bars × 190), z-score each column (using global feature std), center
- Run SVD → first principal direction → unit vector `d̂` in 190-D
- Store per trade: `centroid c (190-vec)`, `direction d̂ (190-vec unit)`, `magnitude s₁ (singular value)`
- Implementation: PyTorch CUDA, batched SVD via `torch.linalg.svd` where T matches across trades; loop or pad-and-mask where they don't

### Similarity criterion: r-distance to seed's PCA line
- Given seed's `(c_seed, d̂_seed)` and a candidate trade's trajectory `P` (T_cand × 190 z-scored points)
- For each candidate point `p ∈ P`: perpendicular distance to seed's line = `||(p − c_seed) − ((p − c_seed) · d̂_seed) · d̂_seed||`
- Candidate matches if **all (or a configurable quantile of) its points are within distance `r`** of seed's line
- `r` is measured in z-scored units → it's the average per-dimension std-distance across the 190 dims

### Clustering protocol: hierarchical, coarse-to-fine (Option 2)
- Within each direction pool:
  ```
  while pool not exhausted and trades remain meaningful:
      seed = trade with max mfe_velocity in pool
      fit seed's PCA line
      
      Iterate r-levels coarse → fine:
          r₀ = R_COARSE (default e.g., 0.5σ)
          coarse_cluster = trades within r₀ of seed's line
          r₁ = R_COARSE / 2
          medium_cluster = subset of coarse with distance < r₁
          r₂ = R_COARSE / 4
          fine_cluster = subset of medium with distance < r₂
          ...
      
      Store hierarchy as a path: seed_id.coarse, seed_id.medium, seed_id.fine
      Remove coarse_cluster from pool   (default; configurable)
  ```
- Each peeled seed produces a **tree** of nested clusters at multiple `r` levels
- Bayesian table holds the full tree per direction

### Decay tracking (FREE side-effect of trajectory clustering)
- For each clustered trade, store `d(t)` = the time series of r-distances from each of the trade's trajectory points to its matched cluster's PCA line
- Decay curve shapes (post-hoc, descriptive):
  - **stable** : `d(t)` ≈ constant small — archetype is the trade's persistent character
  - **decaying** : `d(t)` grows over time — trade drifts out of archetype
  - **converging** : `d(t)` starts large, shrinks — late-developing archetype member
- The decay metric and the "is decayed" threshold are explicit open questions (see Improve §Open).

### What this unifies
- **Entry classifier**: which cluster does the bar match → predicted direction (cluster's pct_long) + magnitude (cluster's mean_signed)
- **Exit signal**: as live trade develops, distance to cluster line grows → archetype downgrade or exit
- **Duration prediction**: cluster's decay-rate distribution → expected time in archetype
- **Bayesian posterior update**: prior = cluster prior_weight (n/total); likelihood updated each bar from `d(t)` trajectory; posterior = updated direction confidence

### CUDA implementation
- Use PyTorch (matches the codebase's CUDA-only convention)
- Z-score features once (global mean/std) for stable per-dimension distance
- Batch SVD over trades with same T; loop for others or use rolled fixed-T resampling
- Distance computations: matmul-friendly, full GPU

## Improve — implementation phases

### Phase 1: V2 feature join
- Build `tools/regret_join_v2_features.py`
- Input: daisy-chain CSV
- Output: per-trade torch tensor saved to disk (one file per trade or one big sharded file)
- Validation: assert all bar timestamps in [entry_ts, exit_ts] have V2 features available; report skip rate

### Phase 2: Per-trade signatures
- Tool: `tools/regret_trade_signatures.py` (CUDA)
- For each trade: load trajectory tensor → z-score per feature → SVD → save (centroid, direction, magnitude, full distance-from-own-line series)
- Edge case: trades with T_bars < 5 — PCA unstable; flag and either drop or skip clustering of these

### Phase 3: Bayesian table builder
- Tool: `tools/regret_bayesian_table.py` (CUDA)
- Implements the peel + hierarchical r-ladder protocol above
- Inputs (CLI):
  - `--input` — daisy-chain CSV
  - `--trajectories` — per-trade signature file
  - `--r-coarse` — top of r-ladder (default 0.5)
  - `--r-levels` — number of levels (default 3 → ladder [0.5, 0.25, 0.125])
  - `--seed-feature` — column for seed selection (default `mfe_velocity`)
  - `--direction-stratify` — bool (default True)
  - `--min-cluster-size` — drop sub-clusters smaller than this (default 5)
  - `--min-pool` — stop when residual < this (default 20)
  - `--max-iterations` — safety (default 500)
  - `--remove-on-peel` — `coarse` (default), `medium`, `fine`, `none`
- Outputs:
  - `bayesian_table_<name>.csv` — one row per cluster node, hierarchical IDs, prior_weight, centroid, direction, n_members, mean_signed, std_signed, mean_duration, mean_decay_rate
  - `cluster_assignments_<name>.csv` — one row per trade with its assigned cluster path
  - `decay_curves_<name>.npz` — one decay curve per trade (variable length)

### Phase 4: IS sanity check
- Run on full 2025 IS
- Inspect: first 5 clusters per direction — do they look like recognizable archetypes (rubber-band crash, slow grind, etc.)?
- Cluster size distribution — too many singletons or too few clusters?
- Decay curve patterns — does the data support the stable/decaying/converging trichotomy?

### Phase 5: Live evaluator
- Tool: `tools/regret_bayesian_live_eval.py`
- For each bar in a held-out (or 2026 OOS) day: match to closest cluster centroid using the bar's recent past trajectory (e.g., past 60 bars)
- Predict direction = cluster's pct_long > 50 → LONG, else SHORT
- Aggregate: Day WR + mode $/day per CLAUDE.md protocol
- Compare to k-way cell-gate baseline from yesterday

### Open questions to resolve before live deployment

1. **Decay metric definition.**
   - Option A: raw r-distance `d(t)` in z-scored space
   - Option B: normalized to r-level — `d(t) / r_fine` so 0=on-line, 1=fine edge, 2=medium edge, 4=coarse edge
   - Trade-off: B is cleaner for cross-cluster comparison; A is the natural geometric quantity
   - DEFERRED until first IS sanity check shows what the decay curves actually look like

2. **"Decayed" threshold.**
   - Option A: threshold-based — `d(t) > r_coarse` → decayed (binary state crossing)
   - Option B: rate-based — `d(t)` has sustained positive slope over K bars → decaying (gradual)
   - Option C: both — threshold for hard exit signals, rate for early-warning
   - Trade-off: threshold is simple, rate is smoother
   - DEFERRED until decay curves are visualized; decision blocks live exit logic but not cluster building

### Live applicability — explicit acknowledgement

The clustering uses each trade's **forward** trajectory (entry → exit) — i.e., full information including the future of the trade. A live system at the entry bar doesn't have this. Two paths for live use:

- **Lead-in trajectory matching**: also model each trade's *past N bars before entry* trajectory. Cluster those too. At live time, match the current bar's past-N trajectory to the lead-in clusters. Predict direction from the forward cluster correlated with the matched lead-in cluster.
- **Single-bar matching with decay update**: at entry, compute distance from the current bar to all cluster centroids. Pick the closest. As subsequent bars arrive, update the distance trajectory and refine the cluster match. The first few bars are noisy; confidence builds over time.

Phase 5 will explore both. Build resolves this at evaluator time, not clustering time.

## Control

- IS-only findings are descriptive; **2026 OOS validation is mandatory** per MEMORY hard rule before any claim of edge.
- Per-cluster sign-stability test on OOS: do the same clusters still produce the same direction-skew on fresh data?
- Day WR + mode $/day with 95% bootstrap CI per CLAUDE.md, never quoted without it.
- Anti-doom-cascade rule: report deployment risk under multiple gap assumptions (per CLAUDE.md).

## Cycle log

- `cycle_01.md` — pending (will document Phase 1 + Phase 2 build)
