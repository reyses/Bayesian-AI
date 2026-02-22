# Jules Task: Original Star-Schema Clustering Baseline

## Goal

Restore `training/fractal_clustering.py` on the `old-snowflake-clustering` branch to
the **pre-snowflake original star schema** (the unified KMeans + z-variance recursive
refinement that existed at commit `a4c1b28`, right after PR #192 dynamic exits merge).

Then apply the agreed anti-overfitting fixes on top of that restored baseline, so we
have a clean apples-to-apples comparison against the new shape-first clustering on `main`.

This branch must NOT have the LONG/SHORT snowflake split — we are going back to the
original unified pool approach.

---

## Branch

Work exclusively on: `old-snowflake-clustering`

```bash
git checkout old-snowflake-clustering
```

Do NOT merge or rebase against main.

---

## Step 1 — Restore the pre-snowflake fractal_clustering.py

The file on this branch currently has the snowflake LONG/SHORT split. Restore it to
the original star-schema state using the parent of the snowflake commit (`0d78aee`):

```bash
git checkout 0d78aee^1 -- training/fractal_clustering.py
```

This restores the file to the state just before Jules added the snowflake clusters
(commit `0d78aee` = "Implement Snowflake Clusters, Fractal DNA Tree, Golden Path Oracle…").
The parent commit `0d78aee^1` is `d92f495` — the last state with the original star schema. The restored file uses:
- Unified coarse KMeans (all patterns, no LONG/SHORT split)
- Z-variance recursive refinement (`max_variance=0.5`)
- Silhouette-based `refine_clusters` (we will replace this in Step 3)

The orchestrator on this branch still has fallback logic:
if `pattern_library_long.pkl` not found → falls back to `pattern_library.pkl`.
So NO orchestrator changes are needed.

---

## Step 2 — Raise minimum split thresholds

In the restored `training/fractal_clustering.py`, the stopping conditions are inline
integers (not named constants). Raise ALL occurrences of `<= 20` and `> 20` used as
minimum-member guards to 30:

In `_recursive_split` (around line 169):
```python
# Change:
if z_var <= self.max_variance or len(patterns) <= 20 or depth > 5:

# To:
if z_var <= self.max_variance or len(patterns) <= 30 or depth > 5:
```

In `create_templates` (around line 487):
```python
# Change:
if z_variance > self.max_variance and len(indices) > 20:

# To:
if z_variance > self.max_variance and len(indices) > 30:
```

Also add named constants near the top of the file (after imports):
```python
MIN_PATTERNS_FOR_SPLIT  = 30
MIN_SAMPLES_PER_CLUSTER = 30
R2_FISSION_MIN_GAIN     = 0.05
```

---

## Step 3 — Raise TEMPLATE_MIN_MEMBERS_FOR_STATS

In `config/oracle_config.py`:
```python
# Change:
TEMPLATE_MIN_MEMBERS_FOR_STATS = 5

# To:
TEMPLATE_MIN_MEMBERS_FOR_STATS = 20
```

---

## Step 4 — Replace fission silhouette gate with adj-R² gain

In `refine_clusters`, the current silhouette-on-exit-params approach is pure
within-sample overfitting. Replace it with adj-R² gain (same logic as main).

Add this new method to `FractalClusteringEngine` (before `refine_clusters`):

```python
def _compute_adj_r2(self, patterns: list, scaler) -> float:
    """Adjusted R² of a linear MFE model on the 16D feature vector."""
    from sklearn.linear_model import LinearRegression
    pairs = [
        (self.extract_features(p), p.oracle_meta.get('mfe'))
        for p in patterns
        if getattr(p, 'oracle_meta', None) is not None
    ]
    pairs = [(f, y) for f, y in pairs if y is not None]
    n, k = len(pairs), 16
    if n <= k + 2:
        return -1.0
    X = scaler.transform(np.array([f for f, _ in pairs]))
    y = np.array([m for _, m in pairs])
    if np.std(y) < 1e-9:
        return 1.0
    ols = LinearRegression().fit(X, y)
    ss_res = float(np.sum((y - ols.predict(X)) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot
    return float(1.0 - (1.0 - r2) * (n - 1) / (n - k - 1))
```

Replace the body of `refine_clusters` with adj-R² gain check:

```python
def refine_clusters(self, template_id: int, member_params: List[Dict[str, float]],
                    original_patterns: List[Any]) -> List[PatternTemplate]:
    """
    Try to split a single template into 2 sub-clusters.
    Accept the split only if weighted children adj-R² - parent adj-R² >= R2_FISSION_MIN_GAIN.
    """
    if len(original_patterns) < MIN_SAMPLES_PER_CLUSTER * 2:
        return []

    features = []
    valid = []
    for p in original_patterns:
        try:
            features.append(self.extract_features(p))
            valid.append(p)
        except AttributeError:
            continue

    if len(valid) < MIN_SAMPLES_PER_CLUSTER * 2:
        return []

    X = np.array(features)
    scaler = StandardScaler().fit(X)
    X_scaled = scaler.transform(X)

    parent_r2 = self._compute_adj_r2(valid, scaler)

    kmeans = self._get_kmeans_model(n_clusters=2, n_samples=len(valid), use_cuda=False)
    labels = kmeans.fit_predict(X_scaled)

    children = []
    for lbl in [0, 1]:
        idx = [i for i, l in enumerate(labels) if l == lbl]
        if len(idx) < MIN_SAMPLES_PER_CLUSTER:
            return []
        children.append([valid[i] for i in idx])

    weighted_r2 = sum(
        len(c) / len(valid) * self._compute_adj_r2(c, scaler)
        for c in children
    )

    if weighted_r2 - parent_r2 < R2_FISSION_MIN_GAIN:
        return []  # split doesn't improve MFE predictability

    result = []
    for i, child_patterns in enumerate(children):
        child_feats = np.array([self.extract_features(p) for p in child_patterns])
        child_scaled = scaler.transform(child_feats)
        centroid_raw = scaler.inverse_transform([np.mean(child_scaled, axis=0)])[0]
        t = PatternTemplate(
            template_id=template_id * 100 + i,
            centroid=centroid_raw,
            member_count=len(child_patterns),
            patterns=child_patterns,
            physics_variance=0.0,
        )
        self._aggregate_oracle_intelligence(t, child_patterns)
        result.append(t)
    return result
```

---

## Step 5 — Add avg_mfe_bar / p75_mfe_bar fields

In `PatternTemplate` dataclass (after `regression_sigma_ticks`):
```python
avg_mfe_bar: float = 0.0
p75_mfe_bar: float = 0.0
```

In `_aggregate_oracle_intelligence`, after the Direction Bias section:
```python
# Time-scale: bar index where MFE peaked
mfe_bars = [
    p.oracle_meta.get('mfe_bar')
    for p in patterns
    if getattr(p, 'oracle_meta', None) is not None
    and p.oracle_meta.get('mfe_bar', -1) >= 0
]
if len(mfe_bars) >= TEMPLATE_MIN_MEMBERS_FOR_STATS:
    template.avg_mfe_bar = float(np.mean(mfe_bars))
    template.p75_mfe_bar = float(np.percentile(mfe_bars, 75))
```

---

## What NOT to Change

- The unified coarse KMeans in `create_templates` — keep it (no LONG/SHORT split)
- The z-variance recursive splitting criterion — keep it (this IS the original baseline)
- `MAX_RECURSION_DEPTH = 5` / `depth > 5` — keep it
- `orchestrator.py` — no changes (fallback to `pattern_library.pkl` already exists)

---

## Verification

After changes, run:
```
python training/orchestrator.py --fresh --forward-start 20250101 --forward-end 20250131
```

Check that:
1. Phase 2.5 does NOT print "Snowflake Clustering: Splitting" — it should print the
   original coarse KMeans output (no LONG/SHORT split)
2. Only ONE library is saved: `pattern_library.pkl` (not long/short pair)
3. Phase 4 completes without errors
4. Oracle report prints

---

## Purpose

This branch produces the original pre-snowflake baseline. After both runs complete:

```python
import pandas as pd
main_df = pd.read_csv('run_logs/oracle_trade_log_main.csv')        # shape-first
snow_df = pd.read_csv('run_logs/oracle_trade_log_snowflake.csv')   # original star schema

print("Shape-first PnL:", main_df.actual_pnl.sum())
print("Original schema: ", snow_df.actual_pnl.sum())
print("Shape-first WR: ", (main_df.result=='WIN').mean())
print("Original WR:    ", (snow_df.result=='WIN').mean())
```

The comparison tells us whether the shape-first redesign improves generalization over
the original star schema, and whether the snowflake was actually helpful or harmful.
