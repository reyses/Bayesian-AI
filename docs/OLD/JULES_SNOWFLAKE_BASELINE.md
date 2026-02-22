# Jules Task: Original Star-Schema Clustering Baseline

## Goal

Apply anti-overfitting fixes to `training/fractal_clustering.py` on the `pre-snowflake`
branch — the original unified KMeans + z-variance recursive refinement that existed
before the snowflake LONG/SHORT split was introduced.

This branch is already at the correct pre-snowflake checkpoint (commit `3d0c1b8`).
No file restoration needed — just apply the fixes below.

This gives us a fair apples-to-apples comparison against the new shape-first clustering
on `main`: same data, same forward pass, different Phase 2.5.

---

## Branch

Work exclusively on: `pre-snowflake`

```bash
git checkout pre-snowflake
```

Do NOT merge or rebase against main.

---

## File to Modify

**`training/fractal_clustering.py`** — apply the fixes below.
**`config/oracle_config.py`** — one constant change.

---

## Changes to Apply

### 1. Raise minimum split thresholds

The stopping conditions are inline integers in the current file. Raise ALL minimum-member
guards from 20 to 30:

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

Also add named constants near the top of the file (after imports, before the dataclass):
```python
MIN_PATTERNS_FOR_SPLIT  = 30
MIN_SAMPLES_PER_CLUSTER = 30
R2_FISSION_MIN_GAIN     = 0.05
```

---

### 2. Raise TEMPLATE_MIN_MEMBERS_FOR_STATS

In `config/oracle_config.py`:
```python
# Change:
TEMPLATE_MIN_MEMBERS_FOR_STATS = 5

# To:
TEMPLATE_MIN_MEMBERS_FOR_STATS = 20
```

This prevents MFE/MAE/avg_mfe_bar statistics from being computed on templates with
fewer than 20 members — too small to estimate percentiles reliably.

---

### 3. Replace fission silhouette gate with adj-R² gain

The current `refine_clusters` uses silhouette on {TP, SL, trail} — pure within-sample
overfitting. Replace it with an adj-R² gain test.

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

Replace the body of `refine_clusters` with the adj-R² gain check:

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

### 4. Add avg_mfe_bar / p75_mfe_bar fields

In `PatternTemplate` dataclass (after `regression_sigma_ticks`):
```python
avg_mfe_bar: float = 0.0
p75_mfe_bar: float = 0.0
```

In `_aggregate_oracle_intelligence`, after the Direction Bias section (section 5 in the
method body), add:
```python
# 6. Time-scale: bar index where MFE peaked
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
- The z-variance recursive splitting criterion — keep it (this IS the baseline)
- `depth > 5` max recursion guard — keep it
- `orchestrator.py` — no changes needed

---

## Verification

After changes, run:
```
python training/orchestrator.py --fresh --forward-start 20250101 --forward-end 20250131
```

Check that:
1. Phase 2.5 prints coarse KMeans output — NO "Snowflake Clustering: Splitting N patterns
   by oracle direction..." message
2. A single `pattern_library.pkl` is saved (not a long/short pair)
3. Phase 4 completes without errors
4. Oracle report prints

---

## Purpose

This branch produces the original pre-snowflake baseline. After both runs complete:

```python
import pandas as pd
main_df = pd.read_csv('run_logs/oracle_trade_log_main.csv')           # shape-first
base_df = pd.read_csv('run_logs/oracle_trade_log_pre_snowflake.csv')  # original star schema

print("Shape-first PnL:", main_df.actual_pnl.sum())
print("Original schema: ", base_df.actual_pnl.sum())
print("Shape-first WR: ", (main_df.result=='WIN').mean())
print("Original WR:    ", (base_df.result=='WIN').mean())
```

The comparison tells us whether the shape-first redesign improves generalization over
the original star schema.
