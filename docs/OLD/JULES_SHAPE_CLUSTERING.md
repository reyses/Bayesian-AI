# Jules Task: Shape-First Clustering with Adj-R² Splitting

## Goal

Replace the current K-means-first clustering approach with a two-stage pipeline:

1. **Shape taxonomy grouping first** — group patterns by their fractal context
   (`depth × pattern_type × lagrange_zone × hurst_category`) before any K-means.
   This ensures a depth-2 (1h) ROCHE_SNAP at L3_ROCHE is never in the same
   cluster as a depth-5 (1m) STRUCTURAL_DRIVE at L1_STABLE. They have different
   physics, different resolution windows, and different MFE distributions.

2. **Adj-R² as the split criterion** — within each shape group, only split further
   when the adjusted R² of `oracle_mfe ~ feature_vector` is below a threshold.
   Adjusted R² naturally penalises splitting small clusters (the adjustment term
   `(n-1)/(n-k-1)` dominates when n is small relative to k=16), preventing
   the overfitting that silhouette/z-variance allow.

## File to Modify

**`training/fractal_clustering.py`** only.

---

## Changes Required

### 1. New Constants (top of file, after existing constants)

```python
R2_STOP_THRESHOLD  = 0.15   # stop splitting when adj-R²(mfe ~ features) >= this
R2_FISSION_MIN_GAIN = 0.05  # minimum adj-R² gain required to allow fission
```

Remove `MAX_RECURSION_DEPTH = 5` — replaced by adj-R² criterion.
Keep `MIN_PATTERNS_FOR_SPLIT = 30` (raise from 20 — hard floor below which we never split).
Keep `MIN_SAMPLES_PER_CLUSTER = 30` (raise from 20 — same floor for _fit_branch).

---

### 2. New Static Method: `_shape_label`

Add as a `@staticmethod` on `FractalClusteringEngine`, after `extract_features`:

```python
@staticmethod
def _shape_label(p) -> str:
    """
    Discrete shape taxonomy used for initial grouping before K-means.

    Encodes the fractal hierarchy position + physical regime of a pattern.
    Patterns with the same label share the same physics context and should
    be analysed together before any quantitative split.

    Returns a string key like: "d5|ROCHE_SNAP|L3_ROCHE|trend"
    """
    depth = int(getattr(p, 'depth', 0))
    ptype = getattr(p, 'pattern_type', 'UNKNOWN')
    state = getattr(p, 'state', None)
    lzone = getattr(state, 'lagrange_zone', 'UNKNOWN') if state else 'UNKNOWN'
    hurst = getattr(state, 'hurst_exponent', 0.5) if state else 0.5
    hcat  = 'trend' if hurst > 0.6 else ('revert' if hurst < 0.4 else 'random')
    return f"d{depth}|{ptype}|{lzone}|{hcat}"
```

---

### 3. New Method: `_compute_adj_r2`

Add as an instance method on `FractalClusteringEngine`, after `_shape_label`:

```python
def _compute_adj_r2(self, patterns: list, scaler) -> float:
    """
    Adjusted R² of oracle_mfe ~ 16D feature vector for a group of patterns.

    Returns -1.0 when there aren't enough patterns to fit reliably (n <= k+2).
    The adjusted penalty is large when n is small relative to k=16,
    so small clusters naturally score low and won't qualify for further splitting.
    Returns 1.0 when MFE variance is near-zero (perfectly coherent cluster).
    """
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
        return 1.0  # zero MFE variance — cluster is perfectly coherent
    ols = LinearRegression().fit(X, y)
    ss_res = float(np.sum((y - ols.predict(X)) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot
    adj_r2 = 1.0 - (1.0 - r2) * (n - 1) / (n - k - 1)
    return float(adj_r2)
```

---

### 4. Replace `_recursive_split`

Replace the entire method with this version that uses adj-R² as the stopping
criterion instead of z-score variance:

```python
def _recursive_split(self, X: np.ndarray, patterns: list,
                     start_id: int, scaler, depth: int = 0) -> list:
    """
    Recursively split a cluster until adj-R²(mfe ~ features) is sufficient
    or the cluster is too small to split reliably.

    Stopping conditions (in priority order):
      1. Hard floor: fewer than MIN_PATTERNS_FOR_SPLIT patterns
      2. Coherence met: adj-R² >= R2_STOP_THRESHOLD (features explain MFE well)
      3. Cannot split: only 1 unique point in feature space

    K-means split uses k=2 or 3 based on cluster size.
    """
    def _make_template(tid, X_sub, pats):
        centroid = np.mean(X_sub, axis=0)
        raw_centroid = scaler.inverse_transform([centroid])[0]
        return PatternTemplate(
            template_id=tid,
            centroid=raw_centroid,
            member_count=len(pats),
            patterns=pats,
            physics_variance=float(np.std(X_sub[:, 0]))
        )

    # 1. Hard floor
    if len(patterns) <= MIN_PATTERNS_FOR_SPLIT:
        return [_make_template(start_id, X, patterns)]

    # 2. Coherence check — stop if adj-R² is already good enough
    adj_r2 = self._compute_adj_r2(patterns, scaler)
    if adj_r2 >= R2_STOP_THRESHOLD:
        return [_make_template(start_id, X, patterns)]

    # 3. Unique-point check
    n_unique = len(np.unique(X, axis=0))
    k = min(3, max(2, len(patterns) // MIN_PATTERNS_FOR_SPLIT))
    k = min(k, n_unique)
    if k <= 1:
        return [_make_template(start_id, X, patterns)]

    # 4. K-means split
    km = self._get_kmeans_model(n_clusters=k, n_samples=len(X))
    labels = km.fit_predict(X)

    result = []
    nid = start_id
    for lbl in range(k):
        mask = labels == lbl
        if mask.sum() == 0:
            continue
        sub_X = X[mask]
        sub_p = [patterns[i] for i in np.where(mask)[0]]
        children = self._recursive_split(sub_X, sub_p, nid, scaler, depth + 1)
        result.extend(children)
        nid += len(children)
    return result
```

---

### 5. Replace `_fit_branch`

Replace the coarse K-means + z-variance loop section with shape-taxonomy grouping.
Keep: feature extraction, scaler fitting, aggregation, and transition matrix setup.
Remove: `target_k` coarse K-means, `z_variance > max_variance` loop.

Full replacement of `_fit_branch`:

```python
def _fit_branch(self, patterns: List[Any], direction: str):
    """
    Shape-first clustering for one directional branch (LONG or SHORT).

    Stage 1: Group patterns by shape taxonomy (depth × type × zone × hurst).
             Each shape group represents a geometrically distinct market situation.
    Stage 2: Within each shape group, recursively split using adj-R² criterion.
             Only split when features do not yet explain MFE variance well enough.

    Returns (scaler, list[PatternTemplate]).
    """
    import time as _time
    from collections import defaultdict
    from sklearn.preprocessing import StandardScaler

    if not patterns:
        return StandardScaler(), []

    print(f"\n--- Fitting {direction} Branch ({len(patterns)} patterns) ---")
    t0 = _time.perf_counter()

    # 1. Extract features and fit scaler on the whole branch
    features, valid_patterns = [], []
    for p in patterns:
        try:
            features.append(self.extract_features(p))
            valid_patterns.append(p)
        except AttributeError:
            continue

    if not features:
        return StandardScaler(), []

    X_all = np.array(features)
    scaler = StandardScaler()
    scaler.fit(X_all)
    print(f"  Feature matrix: {X_all.shape[0]} patterns × {X_all.shape[1]} features")

    # 2. Group by shape taxonomy
    shape_groups = defaultdict(list)
    for p in valid_patterns:
        shape_groups[self._shape_label(p)].append(p)

    print(f"  Shape groups: {len(shape_groups)} distinct shapes")
    for key in sorted(shape_groups):
        print(f"    {key}: {len(shape_groups[key])} patterns")

    # 3. For each shape group: adj-R² recursive split
    start_id_offset = 0 if direction == 'LONG' else 50000
    next_id = start_id_offset
    final_templates = []
    t2 = _time.perf_counter()

    for shape_key in sorted(shape_groups.keys()):
        shape_patterns = shape_groups[shape_key]
        sub_feats, ok_patterns = [], []
        for p in shape_patterns:
            try:
                sub_feats.append(self.extract_features(p))
                ok_patterns.append(p)
            except AttributeError:
                continue
        if not sub_feats:
            continue

        sub_X = scaler.transform(np.array(sub_feats))

        if len(ok_patterns) < MIN_PATTERNS_FOR_SPLIT:
            # Too small to split — one template directly
            centroid = np.mean(sub_X, axis=0)
            raw_centroid = scaler.inverse_transform([centroid])[0]
            final_templates.append(PatternTemplate(
                template_id=next_id,
                centroid=raw_centroid,
                member_count=len(ok_patterns),
                patterns=ok_patterns,
                physics_variance=float(np.std(sub_X[:, 0]))
            ))
            next_id += 1
        else:
            refined = self._recursive_split(sub_X, ok_patterns, next_id, scaler)
            final_templates.extend(refined)
            next_id += len(refined)

    print(f"  Recursive split done ({_time.perf_counter() - t2:.2f}s) → {len(final_templates)} templates")

    # 4. Sort by size, aggregate oracle intelligence
    final_templates.sort(key=lambda x: x.member_count, reverse=True)
    print(f"  Aggregating Oracle Intelligence...", end="", flush=True)
    for template in final_templates:
        self._aggregate_oracle_intelligence(template, template.patterns, scaler)
    print(" done.")

    print(f"  Branch total: {_time.perf_counter() - t0:.1f}s")
    return scaler, final_templates
```

---

### 6. Replace `refine_clusters` (Fission)

Replace the silhouette-on-exit-params fission with adj-R²-gain fission.
A split is only allowed if the weighted adj-R² of children exceeds the parent
by at least `R2_FISSION_MIN_GAIN = 0.05`.

```python
def refine_clusters(self, template_id: int, member_params: List[Dict[str, float]],
                    original_patterns: List[Any]) -> List[PatternTemplate]:
    """
    CLUSTER FISSION (Adj-R² Gain):
    Splits a template only when doing so genuinely improves the explanatory power
    of oracle_mfe ~ feature_vector (measured by weighted adj-R² gain across children).

    Replaces the previous silhouette-on-exit-params approach, which was prone to
    within-sample overfitting — a split could improve silhouette without improving
    out-of-sample predictive coherence.
    """
    if len(original_patterns) < 2 * MIN_PATTERNS_FOR_SPLIT:
        return []

    # Parent adj-R² (using global scaler fitted on all data)
    parent_r2 = self._compute_adj_r2(original_patterns, self.scaler)

    # Extract features once
    feats, ok_pats = [], []
    for p in original_patterns:
        try:
            feats.append(self.extract_features(p))
            ok_pats.append(p)
        except AttributeError:
            continue

    if len(ok_pats) < 2 * MIN_PATTERNS_FOR_SPLIT:
        return []

    X_scaled = self.scaler.transform(np.array(feats))

    best_gain, best_n, best_labels = -np.inf, 1, None

    for n in range(2, MAX_FISSION_CLUSTERS):
        if len(ok_pats) < n * MIN_PATTERNS_FOR_SPLIT:
            break

        km = self._get_kmeans_model(n_clusters=n, n_samples=len(X_scaled), use_cuda=False)
        labels = km.fit(X_scaled).labels_

        # Weighted adj-R² across children
        weighted_r2, total, valid = 0.0, len(ok_pats), True
        for lbl in range(n):
            sub = [ok_pats[i] for i in np.where(labels == lbl)[0]]
            if len(sub) < MIN_PATTERNS_FOR_SPLIT:
                valid = False
                break
            sub_r2 = self._compute_adj_r2(sub, self.scaler)
            weighted_r2 += sub_r2 * len(sub) / total

        if not valid:
            continue
        gain = weighted_r2 - parent_r2
        if gain > best_gain:
            best_gain, best_n, best_labels = gain, n, labels

    if best_gain < R2_FISSION_MIN_GAIN or best_labels is None:
        return []

    print(f"Template {template_id}: FISSION! adj-R² gain={best_gain:+.3f} → {best_n} sub-templates")

    new_templates = []
    for lbl in range(best_n):
        sub_pats = [ok_pats[i] for i in np.where(best_labels == lbl)[0]]
        if not sub_pats:
            continue
        sub_feats = [self.extract_features(p) for p in sub_pats]
        raw_centroid = np.mean(sub_feats, axis=0)
        new_tmpl = PatternTemplate(
            template_id=int(f"{template_id}{lbl}"),
            centroid=raw_centroid,
            member_count=len(sub_pats),
            patterns=sub_pats,
            physics_variance=float(np.std([f[0] for f in sub_feats]))
        )
        self._aggregate_oracle_intelligence(new_tmpl, sub_pats, self.scaler)
        new_templates.append(new_tmpl)

    return new_templates
```

---

## What Changes and Why

| Before | After |
|---|---|
| Coarse K-means on all patterns → z-variance recursive split | Shape taxonomy grouping → adj-R² recursive split |
| `MAX_RECURSION_DEPTH = 5` (arbitrary) | No depth limit — adj-R² stops naturally |
| `z_var <= 0.5` stopping criterion | `adj_r2 >= 0.15` stopping criterion |
| `MIN_PATTERNS_FOR_SPLIT = 20` | `MIN_PATTERNS_FOR_SPLIT = 30` (harder floor) |
| Fission: silhouette on {TP, SL, trail} params | Fission: adj-R² gain on oracle_mfe ~ features |
| Fission threshold: silhouette >= 0.45 | Fission threshold: gain >= 0.05 |

## Remove the Snowflake LONG/SHORT Split

The snowflake split (`create_templates` → `_fit_branch(long_patterns, 'LONG')` +
`_fit_branch(short_patterns, 'SHORT')`) should be **removed**.

Reason: the shape taxonomy already implicitly separates directions.
`lagrange_zone` and `pattern_type` encode the physics of a setup; the
resulting templates will be overwhelmingly one direction which
`_aggregate_oracle_intelligence` captures via `long_bias`/`short_bias`.
Splitting LONG/SHORT upfront halves each branch's sample size, doubling
overfitting risk, and is redundant when the shape groups are already
geometrically coherent.

New `create_templates` outer shell:
```python
def create_templates(self, manifest: List[Any]) -> List[PatternTemplate]:
    """Shape-first clustering — no LONG/SHORT pre-split."""
    print(f"Shape Clustering: {len(manifest)} patterns")
    scaler, templates = self._fit_branch(manifest, 'ALL')
    self.scaler = scaler  # keep fallback scaler

    # Build Transition Matrix
    valid = [p for p in manifest if p is not None]
    if valid:
        print("  Building Transition Matrix...", end="", flush=True)
        self._build_transition_matrix(templates, valid)
        print(" done.")

    self.templates = templates
    return templates
```

In `_fit_branch`, remove the `start_id_offset` branching — just use `next_id = 0`
since there's no LONG/SHORT ID namespace collision anymore.

In `register_template_logic` (orchestrator.py), the `direction` field on each
template will now be '' (empty) — the forward pass should use `long_bias`/`short_bias`
to gate direction rather than checking `template['direction'] == 'LONG'`.
Check how `direction` is used in orchestrator.py Gate 1 and update accordingly.

## What Stays Unchanged

- Direction gating via `long_bias`/`short_bias` in orchestrator Gate 1 (already there)
- `extract_features` 16D vector (unchanged)
- `_aggregate_oracle_intelligence` (unchanged)
- `_build_transition_matrix` (unchanged)
- `create_templates` outer shell (unchanged — still calls `_fit_branch` per direction)
- `PatternTemplate` dataclass (unchanged)
- All orchestrator code (unchanged)

## Expected Template Count

With shape-first grouping and adj-R² stopping:
- Initial shape buckets: ~30-80 distinct shapes per direction (fewer than current 500+ coarse K clusters)
- After adj-R² splitting: only heterogeneous buckets are split; homogeneous ones stay whole
- Net result: fewer total templates, but each template covers a coherent geometric regime
- Templates with < 30 members should be rare or absent (hard floor prevents it)

## Notes

- Requires `--fresh` (shape grouping will produce different template IDs)
- `oracle_meta['mfe']` is available at Phase 2.5 — computed during Phase 2 discovery
- adj-R² will return -1.0 for groups with < 18 patterns (n ≤ k+2) — they become single templates without splitting, correctly
- `self.scaler` in `refine_clusters` uses the global fallback scaler fitted on all patterns — acceptable since direction is already separated upstream
