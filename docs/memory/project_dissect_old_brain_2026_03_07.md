---
name: Dissection of pre-snowflake / star-schema Bayesian brain (worktree at 09cd30d8, 2026-03-07)
description: Concrete file-level mapping of the original Bayesian brain + K-means template architecture from the worktree dissection on 2026-05-09 evening. Maps each old file to the V0 chord-keyed equivalent.
type: project
---

**Worktree**: `c:/tmp/dissect-old-bayesian-brain` (commit 09cd30d8, March 7 2026)
**Pre-snowflake spec**: commit 47fa8b3a doc `docs/JULES_SNOWFLAKE_BASELINE.md` (250 lines)
**Pre-snowflake checkpoint commit referenced as `3d0c1b8`** — that commit is on a deleted branch, not in current refs

## Architecture map (file by file)

### `core/fractal_clustering.py` (658 lines) — the K-means template engine

The substrate red flag concentrated here.

**16-D feature vector per pattern** (the obscured multi-D concept):
```
[|z_score|, |velocity|, |momentum|, coherence,
 log2(tf_seconds), depth, parent_is_roche,
 self_adx, self_hurst, self_dmi_diff,
 parent_z, parent_dmi_diff, root_is_roche, tf_alignment,
 self_pid, self_osc_coh]
```

Mixed semantics: physics measures + fractal-tree position (depth, parent context)
+ orientation flags (root_is_roche) + multi-TF coherence. K-means on this blob
produced centroids that no longer corresponded to recognizable patterns.

**`PatternTemplate` dataclass — the star-schema fact row**:
```
template_id            cluster id
centroid               16-D K-means centroid
member_count           # patterns in this cluster
patterns               list of original PatternEvents
physics_variance       cluster tightness
transition_map         {next_template_id: probability}  (Markov)
expected_value         (WR * AvgWin) - (LossR * AvgLoss)
outcome_variance       std(PnL)
avg_drawdown           mean MAE
risk_score, risk_variance
stats_win_rate         fraction with |oracle_marker| >= 1
stats_expectancy       mean(mfe - mae)
stats_mega_rate        fraction with |oracle_marker| == 2
long_bias, short_bias  fraction of positive/negative markers
```

**Recursive K-means**: a parent cluster splits into children if z-variance still
high AND children have >= MIN_SAMPLES_PER_CLUSTER members; depth capped at 5.

**Pre-snowflake spec fix** (commit 47fa8b3a, Feb 21 2026):
1. Raise MIN_PATTERNS_FOR_SPLIT from 20 to 30
2. Replace within-sample silhouette gate with adj-R² gain check
   (split accepted only if weighted_children_adj_R² - parent_adj_R² >= 0.05)
3. Add avg_mfe_bar / p75_mfe_bar fields to template (time-scale of MFE peak)
4. TEMPLATE_MIN_MEMBERS_FOR_STATS raised 5 -> 20

The adj-R² gain test is a BAYESIAN-FLAVOR overfit gate — only split if predictive
power genuinely improves. Today's r2adj_5m primitive uses the same statistical
concept (adjusted R² of a linear fit) but applied at 5min cadence to 5s closes
rather than to the MFE prediction model.

### `core/bayesian_brain.py` (462 lines) — the brain

```
table[StateVector]            -> {wins, losses, total}     # state-keyed
dir_table[(tid, direction)]   -> {wins, losses, total}     # template x direction
dir_bias[tid]                 -> {long_w, long_l, short_w, short_l}
```

`update(TradeOutcome)` writes to all three tables on each completed trade.
TradeOutcome has: state vector, entry/exit prices, pnl, result, exit_reason,
direction, template_id.

The `befdc2df` (March 8 2026) commit added per-template-direction PnL tracking:
`get_expected_pnl(tid, side)` returns running average $/trade, used to classify
upcoming trades as pos-EV / neg-EV / unknown.

### `core/execution_engine.py` (933 lines) — the gate cascade

Phase-1 gates (run on ALL candidates):
- Gate 0       headroom, physics, noise filtering (multiple sub-rules:
                 noise / r3_struct / r3_snap / r4_nightmare / r4_struct /
                 hurst / momentum / tunnel)
- Gate 0.5     extra filter
- Gate 1       distance-to-centroid (must match a known template)
- Gate 2       fractal depth filter

Phase-2 gates (run on the COMPETITION WINNER only):
- Direction cascade
- Gate 3       conviction
- Gate 4       momentum alignment

The "9 gates" the user described maps to gate 0 (with 5+ sub-rules counted as
distinct gates) + gates 0.5 / 1 / 2 / 3 / 4 ≈ 9 gating decisions.

Cluster match (Gate 1) uses scaled-Euclidean to nearest centroid on the 16-D
feature vector; tid -> pattern_library entry holds the brain's posteriors.

### `core/state_vector.py` (80 lines) — the brain keys

Compact representation of market state as a single hashable vector — the
`table[StateVector]` key in bayesian_brain.

### `core/three_body_state.py` — substrate

Where today's "3-body" terminology was established (M_close, M_high, M_low).
The framework lived through the 2026-05-09 work even after the original brain
was deleted.

### `core/timeframe_belief_network.py` — TBN

Multi-TF worker consensus. Maps to the modern `core/tbn` infrastructure.

### `core/quantum_field_engine.py` — physics-paradigm engine

Purged in commit 841e5dc6 ('refactor: metaphor purge, code consolidation,
CPU path removal, compressed replay'). Used physics terms (tunnel probability,
analytical fields) that were removed when CLAUDE.md banned physics metaphors
in production code.

## Mapping to V0 (chord-keyed Bayesian table)

| old file/concept | replaced by |
|------------------|-------------|
| 16-D feature vector (mixed semantics) | 5-axis primitive chord (statistical, single-meaning per axis) |
| K-means recursive splitting | quantile bucketing (canonical cells, no centroid averaging) |
| MIN_SAMPLES_PER_CLUSTER 30 + adj-R² fission gate | hierarchical shrinkage cell -> axis-marginal -> universal |
| PatternTemplate star schema (template_id keyed) | bayesian_table.py cells (chord-tuple keyed) |
| transition_map (Markov next-template) | DEFERRED - not needed for V0 |
| `bayesian_brain.table[StateVector]` | replaced by chord lookup |
| `dir_table[(tid, direction)]` | (chord, tier) cells in `training_iso_v2/bayesian_table.py` |
| `dir_bias[tid]` | per-tier direction bias rolled into per-tier $/trade per cell |
| Gate 0 (headroom/physics) | NORMAL-branch context filter on z, hurst, momentum |
| Gate 1 (cluster match) | OBSOLETE - chord IS the cluster identity |
| Gate 2 (depth) | DEFERRED - fractal depth not in current substrate |
| Gate 3 (conviction) | per-cell P_cascade or per-tier expected $/trade gate |
| Gate 4 (momentum align) | maps to slope/curv axis combination in chord |

## Outstanding questions for the V0 build

1. Do we need the transition_map (Markov next-cell) feature? The original brain
   tracked which template tended to fire next; could be useful for sequence-aware
   gating.
2. Do we restore the 'depth' / fractal-tree-position context? The V0 chord drops
   it, but the original brain considered fractal depth meaningful.
3. The original brain stored a `risk_score` that combined drawdown + variance
   into a single scalar; do we want this as a derived field per chord cell?

## Cleanup

To remove the worktree when dissection is done:
```
git worktree remove c:/tmp/dissect-old-bayesian-brain
```

The worktree adds no risk to main. Reading is read-only; even if we accidentally
modify files in the worktree they're disconnected from any branch ref.
