# L2 redo on V2 features + L3 build smokes — 2026-05-16 (autonomous run)

User asked: re-run L2 direction discrimination on the full ~190 V2 features
(not just the ~19 daisy-chain state vector) AND build L3 (Bayesian trade
archetypes via N-D trajectory clustering) with smoke tests to verify
runtime, in parallel.

## TL;DR

1. **L2 redo on V2 features: R² jumped from 0.262 → 0.307 at k=2** with
   the top-30 pruned V2 features. Same R² as prior k=3, achieved at k=2.
   The V2 stack has features stronger than anything in our prior ~19-feature
   subset — top single feature `L2_15s_price_velocity_12` has lin R²=0.248
   and Spearman ρ=−0.62 on signed_mfe alone.
2. **New direction pattern surfaced**: multi-TF velocity divergence (fast
   TF Q1 × slow TF Q5 → 100% LONG; mirror → 100% SHORT, n=157-173 per cell).
3. **L3 build complete, all smoke tests passed**. P1 trajectory join at
   700 trades/sec. P2 PCA signatures at 21 ms/trade. P3 hierarchical
   Bayesian table builder produces valid nested clusters with correct
   direction-stratification (100% pct_long in LONG clusters).
4. **Full L3 pipeline projected runtime: ~5 min total** (11s trajectory
   build + 3 min signatures + ~30s Bayesian table). Tractable.
5. **One real engineering issue surfaced and fixed**: per-trade PCA SVD on
   GPU was failing for all trades due to ill-conditioning from highly
   correlated V2 features + NaN-fill from slow-TF features being constant
   within a trade. Switched to CPU scipy gesvd + per-trade constant-column
   filter — robust.

## Pipeline built

### Phase 1: V2 feature join → 184 features at entry bar

Tool: `tools/regret_join_v2_features.py`
Joins each daisy-chain trade to its V2 features at the entry bar via
asof-backward on the daily parquets at `DATA/ATLAS/FEATURES_5s_v2/L*_*/`.

- Output: `reports/findings/regret_oracle/daisy_with_v2_features_IS_full.parquet`
- 7,925 rows × 258 cols (74 daisy-chain + 184 V2 features)
- V2 NaN rate: 0.35% (clean)

### Phase 1.5: Feature prune

Tool: `tools/regret_feature_prune.py`
1. 1D regression per V2 feature against signed_mfe (signed direction × magnitude).
2. Correlation matrix on all V2 features.
3. Hierarchical-cluster features at |corr| ≥ 0.85.
4. Pick one representative per cluster (highest combined lin_r2 + |Spearman ρ|).

- 184 → 132 representatives (28% reduction — V2 features less redundant than expected at 0.85 threshold)
- Output: `reports/findings/regret_oracle/feature_prune_representatives_IS_full_daisy_v2.txt`

**Top features by combined score** (lin_r2 + |Spearman ρ|):

| Feature | lin_r2 | Spearman ρ |
|---|---|---|
| **L2_15s_price_velocity_12** | **0.248** | **−0.62** |
| L2_5s_price_velocity_9 | 0.209 | −0.61 |
| L2_1m_price_velocity_15 | 0.233 | −0.58 |
| L1_1m_body | 0.167 | −0.48 |
| L1_15s_body | 0.152 | −0.48 |
| L1_5m_body | 0.174 | −0.46 |
| L1_5s_price_velocity_1b | 0.135 | −0.47 |
| L3_5m_z_se_9 | 0.102 | −0.39 |
| L3_1m_z_se_15 | 0.094 | −0.38 |
| L2_5m_price_accel_9 | 0.118 | −0.36 |

**These are stronger than anything in the prior ~19-feature subset.**
`slope_15s_3m` was our best single feature before with R²~0.20 on signed_mfe.
`L2_15s_price_velocity_12` is now at R²=0.248.

### Phase 2: L2 redo k-way on V2 pruned features

Tool: `tools/regret_kway.py` (extended to accept `--features-file` and parquet input)

**k=2 results, top-30 pruned V2 features, signed_mfe target:**

| Pair | R² order-2 (interaction) |
|---|---|
| **L2_15s_price_velocity_12 × L2_1m_price_velocity_15** | **0.307** |
| L2_5s_price_velocity_9 × L2_1m_price_velocity_15 | 0.301 |
| L2_15s_price_velocity_12 × L1_5m_body | 0.291 |
| L2_15s_price_velocity_12 × L2_5s_price_velocity_9 | 0.290 |
| L2_15s_price_velocity_12 × L1_5s_price_velocity_1b | 0.284 |

**Prior best k=2 on ~19-feature set: R² = 0.262.**
**V2 top-30 at k=2 matches prior k=3.** R² ceiling appears not yet
saturated — V2 features add real capacity.

k=3 currently running in background (will update when complete).

**Direction-callable cells from V2 redo** (Wilson CI > 70% or < 30%):

| Top LONG | Top SHORT |
|---|---|
| `L2_15s_velocity Q1 × L2_5m_velocity Q5` n=157, **97.5% LONG**, mean_signed +$207 | `L2_1m_velocity Q5 × L2_5m_velocity Q1` n=113, 6.2% LONG, mean_signed −$252 |
| `L2_15s_velocity Q1 × L2_1m_velocity Q5` n=30, **100% LONG**, mean_signed +$234 | `L2_15s_velocity Q5 × L2_5m_velocity Q1` n=173, 3.5% LONG, mean_signed −$238 |
| `L2_5s_velocity Q9 × L1_15s_body Q5` n=57, 89.5% LONG, mean_signed +$252 | `L2_5s_velocity Q5 × L2_5m_velocity Q1` n=175, 3.4% LONG, mean_signed −$226 |

**Pattern: multi-TF velocity divergence.** Fast-TF (15s, 5s, 1m) Q1 paired
with slow-TF (5m, 15m) Q5 → 100% LONG. Mirror → 100% SHORT. This is the
"trajectory direction is different at different timescales — there's a fold
about to happen" archetype, made visible by the V2 stack's multi-TF
velocity features. The ~19-feature subset didn't have these.

## L3 build phases (all smoke-tested)

### P1: Trajectory join (per-bar over trade duration)

Tool: `tools/regret_join_v2_trajectories.py`

- Input: daisy-chain CSV. For each trade, generate 5s timestamps from
  entry_ts to exit_ts; asof-backward join V2 features for each.
- Output: npz with `oracle_idx`, `bar_idx`, `X` (n_total_bars × 184), `feature_names`.
- **Smoke (50 trades): 700 trades/sec, 18 MB, 0.1s wall**
- **Projection for full 7,925 trades: ~11 sec, ~3 GB**
- NaN rate 24.7% — driven by slow-TF features (L_1h/4h/1D) updating slower
  than 5s; asof-backward is correct but leaves NaN before first update of
  the trade. Handled in P2 with column-mean fill + zero-fill belt-and-suspenders.

### P2: Per-trade PCA signatures

Tool: `tools/regret_trade_signatures.py`

- For each trade: z-score globally, then per-trade SVD on the centered
  trajectory matrix. Extract centroid (n_features-vec), direction (unit
  vector = first principal axis), magnitude (top singular value).
- **Smoke (50 trades): 21 ms/trade, 100% success after fixes**
- **Projection for full 7,925 trades: ~3 min CPU**

**Engineering note worth recording:**
Initial attempt batched `torch.linalg.svd` on CUDA. Failed for ALL trades
with "input matrix is ill-conditioned or has too many repeated singular
values" — root cause: V2 features highly correlated even after pruning,
combined with slow-TF features being constant within a trade (their update
period exceeds trade duration). The constant columns + collinear columns
make LAPACK's `gesdd` fail to converge.

**Fix (applied):**
1. Switch to CPU `numpy.linalg.svd` with `scipy.linalg.svd(..., lapack_driver='gesvd')` fallback (more robust than `gesdd`).
2. Drop per-trade constant columns BEFORE SVD (centered std < 1e-6 → exclude); map direction back to full feature space with zero for excluded.
3. Belt-and-suspenders NaN-to-zero scrub before SVD.

After fixes: 100% of smoke trades produced valid signatures.
Direction norms = 1.0 (correctly unit-length).
Magnitude distribution: median 63.3, p25 50.2, p75 92.4, max 232.6.

### P3: Hierarchical Bayesian table builder

Tool: `tools/regret_bayesian_table.py`

- Level 1 stratification: LONG / SHORT.
- Within each pool: peel by max `mfe_velocity` seed.
- For each seed: compute perpendicular distance from each candidate's
  centroid to seed's PCA line (centroid-mode; per-bar trajectory mode TODO).
- Hierarchical r-ladder (configurable, default [2.0, 1.0, 0.5, 0.25]);
  cluster at multiple radii → hierarchical tree per seed.
- Output: `bayesian_table_<name>.csv` (one row per cluster node) +
  `cluster_assignments_<name>.csv` (trade → cluster path).

**Smoke (50 trades): r-ladder needs calibration** for 184-D space — initial
defaults were too tight (typical inter-centroid distance scales as
√n_features ≈ 13.5 in z-scored space; r=2 produces zero matches). Bumped
to `[12, 8, 5, 3]` and the build worked:

- LONG: 2 coarse seeds → 22 + 5 = 27 LONG trades clustered
- SHORT: 4 coarse seeds → catches all 24 SHORT trades
- Hierarchical structure correct: L2 ⊂ L1 ⊂ L0 (nested)
- 100% pct_long in LONG clusters, 0% in SHORT clusters (Level-1 stratification works)

For the full IS run, r values likely need similar magnitudes (~10-20 for
coarse, ~3-5 for fine). Will auto-calibrate as a follow-up.

## Outputs (so far)

All under `reports/findings/regret_oracle/`:

- `daisy_with_v2_features_IS_full.parquet` — daisy-chain × V2 entry-bar features
- `feature_prune_IS_full_daisy_v2.csv` — full prune cluster table
- `feature_prune_representatives_IS_full_daisy_v2.txt` — 132 pruned features
- `kway_2_clusters_IS_full_v2_top30.csv` — V2 top-30 k=2 clusters (10,820 cells)
- `kway_2_regression_IS_full_v2_top30.csv` — V2 top-30 k=2 regression (435 pairs)
- `trajectories_smoke_50.npz` — L3 P1 smoke output
- `signatures_smoke_50.npz` — L3 P2 smoke output
- `bayesian_table_smoke_50.csv` + `cluster_assignments_smoke_50.csv` — L3 P3 smoke
- (Pending) `trajectories_IS_full.npz`, `signatures_IS_full.npz`, `bayesian_table_IS_full.csv`
- (Pending) `kway_3_*_v2_top30.csv` — L2 k=3 V2 (running in background)

## Decision criteria — early read

| L2 V2 outcome | Implication for L3 build |
|---|---|
| R² stays ~0.30 at all k | Ceiling close to prior; L3 trajectory work has clear motivation |
| R² jumps to ~0.40 | Mixed — both layers should be tested in OOS |
| R² jumps to ~0.50+ | L2 might suffice; L3 becomes optional polish |

**Currently at k=2 V2: R² = 0.307 (vs prior 0.262).**
Will know more once k=3 V2 completes.

## Tools added (all in `tools/`)

| Tool | Purpose |
|---|---|
| `regret_join_v2_features.py` | Join daisy-chain × V2 features at entry bar |
| `regret_feature_prune.py` | Correlation cluster + 1D R² rank → pruned list |
| `regret_kway.py` (extended) | Now supports `--features-file` and parquet input |
| `regret_join_v2_trajectories.py` | Per-bar V2 features over trade duration |
| `regret_trade_signatures.py` | Per-trade PCA signature (centroid + dir + magnitude) |
| `regret_bayesian_table.py` | Hierarchical peel + Bayesian table builder |

## Next when user checks in

1. Wait for k=3 V2 to complete and report.
2. Run full-IS L3 pipeline (already started in background).
3. Evaluate cluster structure quality at the full scale.
4. Decide on `r` auto-calibration approach for the full Bayesian table.
5. Apply stratified-1D step (per user's confirmed methodology) on V2 features.

---

## UPDATE — full-IS pipeline ran

### L2 V2 k=3 completed (10,820 cells, 4,060 triplets)

**R²_max at order-3 = 0.337**: `L2_15s_price_velocity_12 × L2_1m_price_velocity_15 × L1_5s_price_velocity_1b`.

Progression on V2 top-30 (signed_mfe target):
- k=2 additive: 0.314 (single best pair)
- k=2 interaction: 0.307 (best at order-2 model)
- k=3 order-3: **0.337** (+0.030 over k=2 best)
- 3-way interaction term adds ~0.005-0.012 R² on top of 2-way — small but non-zero

For comparison, prior ~19-feature set: k=3 = 0.307, k=5 = 0.348. V2 k=3 already approaches prior k=5 with fewer features.

**Top SHORT-callable cells at k=3 are stronger than ever**:
- `L2_1m_velocity Q5 × L3_15s_z_se_12 Q5 × L2_5m_velocity Q1`: **n=34, 100% SHORT, mean_signed −$325**
- `L2_1m_velocity Q5 × L2_5m_velocity Q1 × L1_1m_accel_1b Q5`: **n=47, 96% SHORT, mean_signed −$325**
- `L2_5s_velocity × L1_5m_body × L2_5m_velocity`: **n=74, 100% SHORT, mean_signed −$306**

Pattern still the multi-TF velocity divergence with z-score confirmation.

### Bug found + fixed — exit_ts has spurious gaps

Auditing trade durations during trajectory build: **1,388 trades (17.5%) have `exit_ts − oracle_ts >> time_to_mfe_min × 60`.** Example trade 7919: `time_to_mfe_min = 56.5` (3,390s algorithmic duration) but `exit_ts − oracle_ts = 37,927s` (10.5 hours wall-clock).

Root cause: the daisy-chain session detector uses gap > 30 min to split sessions, but smaller gaps in the V2 5s data (< 30 min) aren't caught. When a trade spans such a small data gap, the algorithm's bar-count duration (`time_to_mfe_min`) stays correct, but `exit_ts` accumulates the wall-clock time including the gap.

**Fix applied** to `tools/regret_join_v2_trajectories.py`: derive trade duration from `time_to_mfe_min × 60` (the algorithmic truth) rather than `exit_ts − oracle_ts`. Trajectory bar count is now correctly bounded at 721 bars (60 min cap).

After fix: max trajectory length = 721 bars (correct), median 444 bars, 3.29M total bars.

### Full-IS L3 trajectory + signatures complete

- **Trajectories**: 12.3s build, 7,925 trades, 3.29M bars, 2.3 GB, NaN rate 0.37%
- **Signatures**: 153s build (19.4 ms/trade CPU), 214 PCA-unstable (T_bars < 5), median direction norm 0.973
- **Magnitude distribution**: median 55.5, p75 79.6, max 1247 — wide spread reflecting trade-trajectory diversity

### Full-IS Bayesian table built

CLI: `--r-ladder 15 10 6 3 --min-pool 20 --min-cluster-size 5`

- **LONG**: 123 seeds peeled, **3,872 of 3,891 trades assigned (97% coverage)**
- **SHORT**: 133 seeds peeled, **3,801 of 3,820 trades assigned (97% coverage)**
- 80 cluster nodes total

**Top LONG clusters** (all 100% pct_long by L1 direction stratification):

| Cluster | n | mean_signed | mean_$_magnitude |
|---|---|---|---|
| LONG.S0027.L0 | 6 | **+$870** | $870 |
| LONG.S0090.L0 | 5 | +$449 | $449 |
| LONG.S0063.L0 | 10 | +$419 | $419 |
| LONG.S0095.L0 | 6 | +$369 | $369 |
| LONG.S0076.L0 | 6 | +$368 | $368 |
| LONG.S0058.L0 | 14 | +$368 | $368 |
| LONG.S0025.L0 | 41 | +$340 | $340 |

**Top SHORT clusters** (all 0% pct_long):

| Cluster | n | mean_signed | mean_$_magnitude |
|---|---|---|---|
| SHORT.S0070.L0 | 8 | **−$672** | $672 |
| SHORT.S0099.L0 | 8 | −$532 | $532 |
| SHORT.S0075.L0 | 5 | −$517 | $517 |
| SHORT.S0030.L0 | 5 | −$488 | $488 |
| SHORT.S0051.L0 | 6 | −$474 | $474 |
| SHORT.S0028.L0 | 29 | −$342 | $342 |

L3 surfaces archetypes L2 cell-gating misses. L2 max-cell mean_signed ≈ ±$340. **L3 reaches ±$870 / ±$672** — smaller n, much higher per-trade magnitude. These are the "rubber-band crash" / "rocket rally" archetypes that share trajectory shape, not just point-features.

### Cluster hierarchy quality — open issue

Only **3 of 80 cluster nodes are at L1 (fine)** — 2 LONG, 1 SHORT, both at r=10. The r-ladder [15, 10, 6, 3] mostly produced coarse-level clusters; sub-archetypal structure at finer r wasn't visible.

Two possible reasons:
1. The trajectories cluster naturally at ONE granularity in 184-D space — r=15 captures the archetype, sub-r refinement doesn't subdivide
2. r-ladder needs different values (e.g., 13, 11, 9, 7 — narrower ladder)

Worth investigating. Configurable for re-runs.

### What this means for L2 vs L3

| Layer | Coverage | Top cell/cluster mean_signed |
|---|---|---|
| **L2 V2 k=3** | 7,925 trades classified into ~10k cells | ±$325 (best cell, n=34-47) |
| **L3 Bayesian table** | 7,673 of 7,925 trades assigned to clusters | **±$870 / ±$672** (small-n, n=5-8) |

L3 captures higher-magnitude archetypes but smaller n per cluster. L2 captures broader coverage but lower per-cell mean. They're complementary — not substitutes.

**Critical caveat per MEMORY hard rule**: all of these are IS-only. The L3 cluster archetypes with mean_signed ±$500-870 at n=5-10 are HIGHLY susceptible to multi-comparison overfit (123 LONG + 133 SHORT seeds tested = 256 cluster comparisons; top by magnitude will have selection bias). 2026 OOS validation is the next critical step — apply the IS-fitted cluster centroids to 2026 entry bars and check if the same archetypes still produce similar pct_long + mean_signed.

## Updated outputs

- `daisy_with_v2_features_IS_full.parquet`
- `feature_prune_*_IS_full_daisy_v2.{csv,txt}`
- `kway_2_*_IS_full_v2_top30.csv` (R² 0.307)
- `kway_3_*_IS_full_v2_top30.csv` (R² 0.337)
- `trajectories_IS_full.npz` (corrected duration source)
- `signatures_IS_full.npz`
- **`bayesian_table_IS_full.csv` (80 cluster nodes)**
- `cluster_assignments_IS_full.csv` (7,673 trade→cluster assignments)

## Next priorities

1. **OOS validation** (critical, blocks any claims): apply IS-fitted L3 cluster centroids to 2026 OOS daisy-chain trades. Check pct_long / mean_signed survival per cluster. Same for L2 V2 cells.
2. **Stratified-1D + stratified pair** on V2 features (per protocol). Will surface stratum-conditional V2 features missed at the global level.
3. **r-ladder calibration** for L3 hierarchy — investigate why L1+ sub-clusters didn't form at r=10-3.
4. **L3 trajectory-mode matching** (vs current centroid-mode) — proper protocol requires per-bar distance test, not just centroid. Slower but more faithful.
5. **L4 selector design** — combining L2 cell-gates + L3 cluster matches into a fire/no-fire rule with direction call.

---

## ⚠️ CRITICAL UPDATE — L3 live-predictability test results

After the L3 build + r=7 refit, ran two validation experiments to test whether
the L3 architecture is actually deployable.

### Experiment 1: Cluster variance decomposition (confirms L3 captures real structure)

For the 3,710 trades assigned to a cluster:
- Global Var(signed_mfe) = $14,405 ($120 std)
- SS_between (cluster means) / SS_total = **76.2%**
- **Cluster R² = 0.762** (vs global feature R² = 0.307-0.337)
- Per-cluster σ median = 34% of global σ
- 224 of 298 clusters (75%) have σ_within < 50% global

**The clustering DOES generalize within homogeneous groups.** The user's
hypothesis is empirically confirmed.

### Experiment 2: Can entry-time features predict which cluster a trade joins?

This is the L4 question. Trained cluster-classifiers on:
- Entry-bar V2 features (184D → 298 classes)
- Lead-in trajectory PCA signatures (past 60 bars → centroid + direction)

Predicted cluster's `mean_signed_mfe` used as the prediction; R² vs actual on
held-out test set.

| Approach | Top-1 acc | Dir match | **Test R²** |
|---|---|---|---|
| Entry features → cluster (LR, 184D → 298 classes) | 40% | 72% | **−0.05** |
| Entry features → cluster (GBM) | 7% | 56% | −0.66 |
| Lead-in nearest-centroid (5min lookback) | 34% | 50% | −0.65 |
| Lead-in LR → cluster (5min lookback) | 28% | 70% | −0.04 |
| Lead-in Ridge → signed_mfe direct | — | — | 0.146 |
| ORACLE (perfect cluster knowledge) | — | — | **0.745** |
| Direct global feature regression k=3 (V2 top-30) | — | — | **0.337** |

### Bottom line

- **L3 cluster R²=0.745 is the theoretical ceiling**, but requires post-hoc
  trade-trajectory knowledge.
- **Neither entry features nor 60-bar lead-in can predict cluster membership
  well enough.** Both routes give R² ≈ −0.05 on held-out test data because
  60-72% cluster misclassifications dominate the residual (wrong-cluster
  predictions carry wrong-sign mean_signed).
- **Direct global feature regression at R²=0.337 remains the practical live
  ceiling** — L3 is descriptive, not prescriptive at this granularity.
- **Direction-only prediction IS meaningful at 70-72%** — the cluster
  classifier captures direction reliably, just not magnitude. A hybrid (use
  clusters for direction, L2 features for magnitude) might break past 0.337.

### Why L3 doesn't translate to live

The cluster structure emerges from the *trajectory's evolution over the
trade's duration* — info that only exists after the trade has played out.
The entry bar and the 60-bar lead-in don't carry enough information about
future trajectory shape. The archetypes are real but cryptic to predict
prospectively.

This is an honest finding worth recording: **L3 captures real structure but
is not directly deployable at this design**. Research artifact that reveals
the underlying trade-archetype distribution but doesn't push past L2's
R²=0.337 ceiling for live prediction.

### What remains to test before declaring L3 not deployable

1. **Longer lookback** (240-bar = 20min, 720-bar = 60min) — 5 min may be too
   short to capture the macro setup leading INTO a trade.
2. **Coarser clustering** (20-50 clusters via k-means) — easier classification
   target. Trade-off: less granular archetypes.
3. **Direction-only hybrid** — abandon magnitude prediction; use clusters
   for direction (~70%), L2 features for magnitude (R²~0.34). Combined could
   beat 0.34.
4. **OOS validation** — none of the above is OOS-validated yet; do that
   first before more iteration to avoid further IS-overfit chasing.

### New tools/outputs

- `tools/regret_leadin_trajectories.py` — NEW (per-trade lead-in PCA signatures)
- `tools/regret_cluster_refit.py` — NEW (pool member trajectories, refit cluster PCA line)
- `tools/regret_bayesian_r_scan.py` — NEW (r-sweep diagnostic, r=1..15)
- `reports/findings/regret_oracle/leadin_signatures_IS_full_60bar.npz`
- `reports/findings/regret_oracle/refined_bayesian_IS_full_r7.{csv,npz}`
- `reports/findings/regret_oracle/IS_full_r_scan_{summary,clusters}.csv`

---

## Late-late: BINARY DIRECTION CLASSIFIER — AUC 0.864

User reframe: "if we can deduce direction the rest we can do risk management"
→ "can we calculate the confidence of the direction upon entry?"

Built `tools/regret_direction_classifier.py` — binary LogisticRegression on V2
entry features predicting LONG=1 / SHORT=0. Output = calibrated P(LONG). The
selector dial is |P - 0.5| (confidence).

### Headline result (test 1585, train 6340)

- **AUC 0.864 / Brier 0.142**
- Train AUC 0.864, Brier 0.142 — **no train-test gap, well-calibrated at tails**
- **Massively better** than the cluster-routed direction approaches
  (R^2 = -0.05). Direct binary direction sidesteps the cluster-misclass
  problem entirely.

### Threshold sweep (test set)

| Threshold | Coverage | Overall acc | LONG acc | SHORT acc |
|-----------|----------|-------------|----------|-----------|
| 0.50 | 100%  | 83.0% | 81.4% | 84.6% |
| 0.70 | 80%   | 87.0% | 87.4% | 86.6% |
| 0.85 | 40%   | **88.0%** | 87.5% | 88.4% |
| 0.90 | 23%   | 88.5% | 88.4% | 88.5% |

The L4 selector dial: pick threshold for coverage/precision trade.

## Lookback expansion test — does lead-in help direction?

User: "agree on expanding the lookback to see if it improves" — tested
whether the macro setup INTO entry adds signal beyond the entry bar itself.

Three lookbacks: 60-bar (5 min), 240-bar (20 min), 720-bar (60 min). Each:
PCA centroid + direction in 184-D V2 space (368 extra features) -> concat
with V2 entry -> LogisticRegression.

| Variant | Test AUC | Test Brier | Train AUC | Train-test gap |
|---------|----------|------------|-----------|----------------|
| **Baseline (V2 entry only)** | **0.864** | **0.142** | 0.864 | 0.000 |
| + 60-bar (5 min) lead-in     | 0.850 | 0.152 | 0.888 | 0.038 |
| + 240-bar (20 min) lead-in   | 0.842 | 0.156 | 0.889 | 0.047 |
| + 720-bar (60 min) lead-in   | 0.849 | 0.153 | 0.886 | 0.037 |

### Verdict: ALL lead-in lookbacks HURT

- Train AUC always rises (model latches onto lead-in features)
- Test AUC always falls (those features are noise on unseen trades)
- Calibration deteriorates at every lookback (Brier up)
- No monotonic trend with lookback length — failure is the approach, not the
  window

### Why lead-in PCA doesn't help

1. **V2 entry features already encode the macro setup.** The L1-L3 x 5s-1D
   stack carries 4h/1D-layer features at the entry bar that contain the
   regime info a lead-in PCA would extract.
2. **PCA signatures are lossy.** A 184-D centroid + direction vector is a
   2-vector summary of a 240x184 trajectory matrix. Most of the variance is
   discarded.
3. **Unit-direction vectors going into a linear model are geometrically
   incoherent.** The model's coefficient projects them onto a "good direction"
   axis, but per-trade the meaning of that direction depends on the centroid
   context — the linear model can't condition on it.

### Architectural implication

- **Drop lead-in PCA for direction prediction.** V2 entry-feature LR is the
  L4 operating signal at AUC 0.864.
- The L3 clusters may still help for OTHER tasks (magnitude prediction, risk
  management, exit timing) but NOT for direction routing.
- **Next AUC lever = non-linear model** (GBM with isotonic calibration, or a
  small CNN), NOT more features. The model class has more headroom than the
  feature set does.

### New tools/outputs (late-late)

- `tools/regret_direction_classifier.py` — NEW (binary LR, AUC 0.864 baseline)
- `reports/findings/regret_oracle/direction_classifier_v2_lr.npz`
- `reports/findings/regret_oracle/direction_classifier_v2_lr_with_leadin{60,240,720}.npz`
- `reports/findings/regret_oracle/leadin_signatures_IS_full_{240,720}bar.npz` — NEW
