# Jules Task: Replace KMeans with Hypervolume Tree Clustering

## Problem Statement

The current KMeans clustering on 16D feature vectors is fundamentally broken:
1. **Centroid space mismatch** — templates store raw centroids, forward pass uses scaled
   features. Result: Gate 1 rejects 100% of candidates. Zero template matches.
2. **Flat compression** — the 16D vector discards the fractal cascade structure. A pattern
   at depth 4 and depth 12 cannot be in the same cluster even if their physics are identical.
3. **Depth filter required** — Gate 0.5 blocks depths 6-12 (91.6% of all FN) because
   deep patterns never match shallow-trained centroids.
4. **Scaler fragility** — any checkpoint staleness breaks the entire matching pipeline.

Pre-snowflake (no depth filter, simpler matching): 50.8% WR, $9,334 PnL, 1,903 trades.
Main (with KMeans + depth filter): 44.0% WR, $1,704 PnL, 141 trades.

## The Fix: Hypervolume Tree with Per-Depth Navigation

### Core Idea

Each pattern produces a **16F × depth matrix** — 16 features at each depth level of the
fractal cascade. This matrix defines a **hypervolume** in 16D space that evolves across
depth. Instead of compressing this to a point and using KMeans, we build a **hierarchical
tree** where each level groups patterns by their 16D geometry at that depth.

### Architecture: Regression-Validated Recursive Grouping

**The regression is the grouping criterion, not the matching criterion.**

At each depth level, for each group of patterns:
1. Fit a **16D multivariate regression** across the group → produces the expected centroid
2. Compute **residuals** — how each pattern deviates from the regression expectation
3. **Group by residual similarity** — patterns that deviate the same way belong together
4. If residuals are tight (low variance) → valid branch (the centroid is meaningful)
5. If residuals are spread → split into sub-branches by clustering the residual vectors

The regression validates that a centroid represents a real geometric structure (not a
KMeans artifact). The residuals reveal the true sub-structure: patterns with the same
deviation pattern share the same hypervolume family, even if their absolute 16D values
differ.

**Training (build the tree):**

```
ALL patterns
  │
  ├─ Depth 0 (1D): Fit 16D regression → compute residuals → group by residuals
  │     │
  │     ├─ Cluster A  (residuals tight → coherent macro regime)
  │     │   ├─ Depth 1 (4h): Fit regression on A's patterns → residuals → group
  │     │   │     ├─ A1  (similar deviations → same sub-structure)
  │     │   │     │   ├─ Depth 2 (1h): regression → residuals → group
  │     │   │     │   │     └─ ... continues to detection depth
  │     │   │     │   ...
  │     │   │     ├─ A2  (different deviation pattern → different branch)
  │     │   │     ...
  │     │   ...
  │     ├─ Cluster B  (different residual family → different macro regime)
  │     │   └─ ...
  │     ...
```

At each level there may be N unique 16D signatures, but the regression + residual
clustering determines which ones genuinely belong together vs which are structurally
different. A branch forms when patterns share the same deviation pattern from the
regression expectation — not just when they're close in Euclidean space.

A template is a **path through the tree**: `A → A3 → A3.7 → A3.7.2`
This path encodes the full hypervolume shape — each node validated by regression.

### Navigation: Workers Walk the Tree (16F × 1D per level)

**Live/forward pass — each worker navigates one level:**

```
Depth 0 worker sees 16F×1D (16 features at macro level)
  → Classify: "I'm in Group A"   (compare 16D vector to depth-0 centroids)

Depth 1 worker sees 16F×1D (16 features at 4h level)
  → Within Group A, classify: "I'm in A3"  (compare to A's depth-1 sub-centroids)

Depth 2 worker sees 16F×1D (16 features at 1h level)
  → Within A3, classify: "I'm in A3.7"  (compare to A3's depth-2 sub-centroids)

  ... each worker navigates one level with just 16 features ...

Detection depth worker:
  → "I'm in A3.7.2" → this IS my template → load template stats
```

Each worker classifies in 16D only — not 168D, not 192D. The tree structure encodes
which groups at level N contain which sub-groups at level N+1. The worker only compares
against the sub-centroids within its parent group (small search space).

### Why This Is Better Than Flat Regression

The previous version (hypervolume regression → 168D signature) still flattens the
structure to a single vector. This tree approach:
1. **Preserves the hierarchy** — macro context constrains micro classification
2. **Workers navigate naturally** — each operates at its own TF level with 16D
3. **Depth-invariant** — a pattern detected at depth 4 walks 5 tree levels;
   depth 12 walks 13 levels. Same tree, different path lengths.
4. **Interpretable** — "Group A = macro uptrend, A3 = 4h pullback, A3.7 = 1h breakout"
5. **No scaler mismatch possible** — each level has its own local grouping
6. **Maps to existing belief network** — workers already sit at each TF level

## Existing Infrastructure on Main (Already Built — Wire In)

These features already exist on main. The hypervolume tree must integrate them, not replace them.

### Feature Extraction & Coordinate System

| Item | File | How Hypervolume Uses It |
|------|------|------------------------|
| `build_16d_vector()` static method | `quantum_field_engine.py` | **Canonical 16D coordinate definition** — every cell boundary, membership test, and live match uses this vector. This IS the hypervolume coordinate system. |
| `extract_features()` (speed-optimized) | `fractal_clustering.py` | `math.log1p` + direct attr access. Used at every depth level during tree build + navigation. |
| `states_map` passed to oracle at all depths | `fractal_discovery_agent.py` | CST structural integrity computed at every depth — hypervolume needs this for all 12 levels. |
| Spectral data enrichment (`z_score`, `velocity`) | `fractal_discovery_agent.py` | Populated before extraction. Available as features in the 16D vector. |

### Oracle & Template Stats

| Item | File | How Hypervolume Uses It |
|------|------|------------------------|
| `mfe_bar` tracking in oracle | `fractal_discovery_agent.py` | Feeds `avg_mfe_bar` / `p75_mfe_bar` on leaf templates. |
| `avg_mfe_bar`, `p75_mfe_bar` on PatternTemplate | `fractal_clustering.py` | Time-exhaustion exits. Computed per hypervolume leaf cell from oracle_meta. |
| CST structural integrity lists in oracle | `fractal_discovery_agent.py` | Per-bar L2 drift from entry. Stored in oracle_meta, used by WaveRider CST. |

### Adj-R² Split Criterion (ADAPT into Tree)

Main's `fractal_clustering.py` uses adj-R²(oracle_mfe ~ 16D features) as the stopping criterion
for recursive splits. **The hypervolume tree absorbs this directly:**

- **Stop splitting** a node when adj-R²(mfe ~ 16D) ≥ 0.15 — features already predict outcomes well
- **Only keep a split** if weighted child R² > parent R² + 0.05 — prevents overfitting
- The adjusted penalty `1 - (1-R²)(n-1)/(n-k-1)` naturally stops when n is small relative to k=16

```python
def _compute_adj_r2(features_16d: np.ndarray, oracle_mfe: np.ndarray) -> float:
    """Already on main. Measures how well 16D features predict MFE for a group."""
    from sklearn.linear_model import LinearRegression
    reg = LinearRegression().fit(features_16d, oracle_mfe)
    r2 = reg.score(features_16d, oracle_mfe)
    n, k = features_16d.shape
    if n <= k + 1:
        return 0.0
    return 1 - (1 - r2) * (n - 1) / (n - k - 1)
```

### Direction & Exit Infrastructure (Post-Matching — Keep As-Is)

| Item | File | Notes |
|------|------|-------|
| Gate 4: direction confidence (`abs(p_long - 0.5) < 0.15`) | `orchestrator.py` | Fires AFTER tree match. No change needed. |
| Gate 5: multi-TF direction consensus (≥0.60) | `orchestrator.py` | Fires AFTER tree match. No change needed. |
| `compute_p_profitable()` | `belief_network.py` | Aggregates DMI/momentum across workers. |
| `get_direction_consensus()` | `belief_network.py` | 4-signal composite for Gate 5. |
| Continuous net pressure model | `belief_network.py` | hold/exit pressure → trail widen/tighten/urgent. |
| Decay cascade (z-score drift) | `belief_network.py` | **EXTEND**: add "live vector left cell" as new decay signal (see Phase 7). |
| Pre-converted coefficients (list→np.array) | `belief_network.py` | ~40x per-tick speedup. Already done. |
| Scaler mean/scale pre-extraction | `belief_network.py` | Avoids sklearn overhead. Already done. |
| Price-aware conviction modulation | `belief_network.py` | Workers modulate by trade_side + profit_ticks. |
| DNA tree integration | `orchestrator.py` | Complementary to hypervolume (identity vs geometry). Keep. |

### Basin Geometry → Cell Bounding Box (ADAPT)

Main stores `basin_mean` and `basin_std` on PatternTemplate for CST integrity checks
(L2 distance < basin_mean + 3*basin_std). **Hypervolume replaces this with explicit bounds:**

```python
# On HypervolumeNode (replaces basin_mean/basin_std):
cell_min_16d: np.ndarray   # per-axis minimum of member features
cell_max_16d: np.ndarray   # per-axis maximum of member features

# CST integrity check becomes:
def is_inside_cell(live_16d: np.ndarray, node: HypervolumeNode) -> bool:
    """Binary membership test — is the live vector inside the hyperrectangle?"""
    return np.all(live_16d >= node.cell_min_16d) and np.all(live_16d <= node.cell_max_16d)
```

This is strictly better than L2 + basin: axis-aligned bounds catch drift on ANY axis,
not just overall distance from centroid.

---

## Implementation

### Phase 1: Enrich Parent Chain with Full 16D Features

**File**: `training/fractal_discovery_agent.py`

The current `_build_parent_chain()` (line 751) stores only 7 values:
```python
{'tf': p.timeframe, 'type': p.pattern_type, 'z': p.z_score,
 'mom': p.momentum, 'coh': p.coherence, 'timestamp': p.timestamp,
 'oracle_marker': getattr(p, 'oracle_marker', 0)}
```

**Change**: Store the full 16D feature vector at each ancestor:
```python
def _build_parent_chain(self, p: PatternEvent) -> List[Dict]:
    """Builds the full parent chain with 16D features at each depth."""
    chain_entry = {
        'tf': p.timeframe,
        'depth': p.depth,
        'type': p.pattern_type,
        'features_16d': FractalClusteringEngine.extract_features(p),
        'timestamp': p.timestamp,
        'oracle_marker': getattr(p, 'oracle_marker', 0),
    }
    return [chain_entry] + (p.parent_chain or [])
```

**IMPORTANT**: The enriched chain must be built during scan, NOT reconstructed later.
Each level's ThreeBodyQuantumState is only available during the cascade scan.

### Phase 2: Hypervolume Tree Data Structure

**File**: `training/fractal_clustering.py` — new class

```python
@dataclass
class HypervolumeNode:
    """One node in the hypervolume tree. Represents a group at a specific depth."""
    depth: int                             # which depth level this node covers
    centroid_16d: np.ndarray               # 16D centroid of this group at this depth
    cell_min_16d: np.ndarray               # per-axis minimum bounds (for CST membership test)
    cell_max_16d: np.ndarray               # per-axis maximum bounds (for CST membership test)
    member_count: int                      # patterns that pass through this node
    children: Dict[int, 'HypervolumeNode'] # child_id -> child node (next depth)
    template: Optional[PatternTemplate]    # leaf nodes get a template with oracle stats
    node_id: str                           # path string e.g. "A3.7.2"

    # Per-depth scaler (fitted on residuals at this depth within this group)
    scaler: Optional[StandardScaler] = None
    regression_r2: float = 0.0             # how well parent predicts this level
    branch_tightness: float = 0.0          # residual variance (lower = tighter group)
    adj_r2_mfe: float = 0.0               # adj-R²(oracle_mfe ~ 16D) — split stopping criterion


@dataclass
class HypervolumeTree:
    """The full hypervolume tree. Root contains depth-0 groups."""
    roots: Dict[int, HypervolumeNode]      # root_id -> depth-0 node
    max_depth: int                         # deepest level in the tree (12 for 1s)
    n_templates: int                       # total leaf templates
```

### Phase 3: Build the Tree (Training)

**File**: `training/fractal_clustering.py` — new method

```python
def fit_hypervolume_tree(self, patterns: List[PatternEvent],
                         min_group_size: int = 30) -> HypervolumeTree:
    """Build hierarchical hypervolume tree by recursive 16D grouping per depth.

    Algorithm:
    1. Extract 16F x depth matrix for each pattern (from enriched parent chain)
    2. At depth 0: cluster ALL patterns by their depth-0 16D slice → root groups
    3. For each root group, at depth 1: cluster by depth-1 16D slice → sub-groups
    4. Recurse until detection depth or group too small to split
    5. Leaf nodes become templates with oracle stats

    At each depth level d, for each group G:
      - Extract the 16D feature vector at depth d for all patterns in G
      - Fit StandardScaler on these vectors (local to this node)
      - Run KMeans (or adj-R² recursive split) to partition G
      - Create child nodes for each sub-cluster
      - Recurse into children at depth d+1
    """
    # 1. Build 16F x depth matrices for all patterns
    matrices = {}  # pattern_id -> np.ndarray (depth+1, 16)
    for i, p in enumerate(patterns):
        mat = self.build_hypervolume_matrix(p)
        if mat is not None and mat.shape[0] >= 2:
            matrices[i] = mat

    # 2. Recursive grouping
    root_nodes = self._split_at_depth(
        pattern_indices=list(matrices.keys()),
        matrices=matrices,
        patterns=patterns,
        depth=0,
        parent_id="",
        min_group_size=min_group_size
    )

    max_depth = max(m.shape[0] for m in matrices.values()) - 1
    tree = HypervolumeTree(roots=root_nodes, max_depth=max_depth,
                           n_templates=self._count_leaves(root_nodes))
    return tree

def _split_at_depth(self, pattern_indices, matrices, patterns,
                    depth, parent_id, min_group_size) -> Dict[int, HypervolumeNode]:
    """Group patterns by regression residuals at given depth. Recurse into children.

    Algorithm:
    1. Extract 16D vectors at this depth for all patterns in the group
    2. Fit multivariate regression: features ~ f(group_context)
       This produces the "expected" centroid for this group at this depth
    3. Compute residuals: how each pattern deviates from the regression expectation
    4. Cluster the RESIDUALS — patterns that deviate the same way = same branch
    5. Validate: only form a branch if residual variance is low (tight group)
    6. Recurse into each branch at depth+1
    """

    # Extract 16D vectors at this depth for all patterns that HAVE this depth
    feat_at_depth = []
    valid_indices = []
    for idx in pattern_indices:
        mat = matrices[idx]
        if depth < mat.shape[0]:  # pattern reaches this depth
            feat_at_depth.append(mat[depth])
            valid_indices.append(idx)

    if len(valid_indices) < min_group_size:
        return {}  # too small — becomes leaf in parent

    feat_array = np.array(feat_at_depth)  # (N, 16)

    # ── Step 0: Check adj-R² stopping criterion ──────────────────────────
    # If 16D features already predict MFE well enough, don't split further.
    # This is adapted from main's _compute_adj_r2 — prevents overfitting.
    oracle_mfe = np.array([getattr(patterns[i], 'oracle_mfe', 0.0)
                           for i in valid_indices])
    if np.std(oracle_mfe) > 0:
        adj_r2 = _compute_adj_r2(feat_array, oracle_mfe)
    else:
        adj_r2 = 0.0

    if adj_r2 >= 0.15 and len(valid_indices) < 200:
        # Features explain MFE well and group is small — stop splitting,
        # this becomes a leaf. Return empty dict so parent creates template.
        return {}

    # ── Step 1: Fit 16D regression to find expected centroid ──────────────
    if depth > 0:
        parent_feat = []
        for idx in valid_indices:
            mat = matrices[idx]
            parent_feat.append(mat[depth - 1])
        parent_array = np.array(parent_feat)  # (N, 16)

        # Multivariate OLS: feat_current = parent_feat @ B + intercept
        # Use torch.linalg.lstsq on CUDA for large groups (>5000 patterns)
        from sklearn.linear_model import LinearRegression
        reg = LinearRegression()
        reg.fit(parent_array, feat_array)
        predicted = reg.predict(parent_array)
        residuals = feat_array - predicted      # (N, 16)
        regression_centroid = feat_array.mean(axis=0)
        regression_r2 = reg.score(parent_array, feat_array)
    else:
        regression_centroid = feat_array.mean(axis=0)
        residuals = feat_array - regression_centroid
        regression_r2 = 0.0

    # ── Step 2: Cluster the residuals ────────────────────────────────────
    scaler = StandardScaler()
    residuals_scaled = scaler.fit_transform(residuals)

    residual_var = np.mean(np.var(residuals_scaled, axis=0))
    if residual_var < 0.3:
        n_clusters = 1
    else:
        n_clusters = self._choose_k(residuals_scaled, valid_indices,
                                     matrices, patterns)

    if n_clusters <= 1:
        labels = np.zeros(len(valid_indices), dtype=int)
    else:
        kmeans = self._get_kmeans_model(n_clusters, len(valid_indices))
        labels = kmeans.fit_predict(residuals_scaled)

    # ── Step 3: Build nodes from residual clusters ───────────────────────
    nodes = {}
    for cluster_id in range(max(1, n_clusters)):
        member_mask = labels == cluster_id
        member_indices = [valid_indices[i] for i, m in enumerate(member_mask) if m]

        if len(member_indices) < min_group_size:
            continue

        node_id = f"{parent_id}{cluster_id}" if parent_id else str(cluster_id)

        # Centroid + cell bounding box in feature space (for matching + CST)
        cluster_feat = feat_array[member_mask]
        centroid = cluster_feat.mean(axis=0)
        cell_min = cluster_feat.min(axis=0)    # per-axis lower bound
        cell_max = cluster_feat.max(axis=0)    # per-axis upper bound

        # Residual tightness = branch quality score
        cluster_residuals = residuals_scaled[member_mask]
        branch_tightness = np.mean(np.var(cluster_residuals, axis=0))

        # Adj-R² for this cluster (how well features predict MFE here)
        cluster_mfe = oracle_mfe[member_mask]
        if len(cluster_feat) > 20 and np.std(cluster_mfe) > 0:
            cluster_adj_r2 = _compute_adj_r2(cluster_feat, cluster_mfe)
        else:
            cluster_adj_r2 = 0.0

        # ── Adj-R² gain check: only split if children improve on parent ──
        # Attempt recursive split into children at next depth
        children = self._split_at_depth(
            member_indices, matrices, patterns,
            depth + 1, node_id + ".", min_group_size
        )

        # If children exist, verify R² gain ≥ 0.05 (from main's fission logic)
        if children:
            child_r2_weighted = sum(
                c.adj_r2_mfe * c.member_count for c in children.values()
            ) / max(1, sum(c.member_count for c in children.values()))
            if child_r2_weighted < cluster_adj_r2 + 0.05:
                children = {}  # split doesn't help — collapse to leaf

        # If no children (leaf), create template with oracle stats
        template = None
        if not children:
            member_patterns = [patterns[i] for i in member_indices]
            template = PatternTemplate(
                template_id=hash(node_id) % 100000,
                centroid=centroid,
                member_count=len(member_patterns),
                patterns=member_patterns,
                physics_variance=branch_tightness,
            )
            self._aggregate_oracle_intelligence(template)
            # Time-scale calibration (from main): avg/p75 MFE bar
            mfe_bars = [getattr(p, 'oracle_meta', {}).get('mfe_bar', 0)
                        for p in member_patterns]
            if mfe_bars:
                template.avg_mfe_bar = np.mean(mfe_bars)
                template.p75_mfe_bar = np.percentile(mfe_bars, 75)

        nodes[cluster_id] = HypervolumeNode(
            depth=depth,
            centroid_16d=centroid,
            cell_min_16d=cell_min,
            cell_max_16d=cell_max,
            member_count=len(member_indices),
            children=children,
            template=template,
            node_id=node_id,
            scaler=scaler,
            regression_r2=regression_r2,
            branch_tightness=branch_tightness,
            adj_r2_mfe=cluster_adj_r2,
        )

    return nodes
```

### Phase 4: Navigate the Tree (Forward Pass)

**File**: `training/orchestrator.py` — replaces Gate 1 L2 matching + Gate 0.5 depth filter

Navigation uses cell membership (bounding box) as primary test, L2 to centroid as tiebreaker.
No global distance threshold needed — the cell bounds define what "matches" at each level.

```python
def _navigate_hypervolume_tree(self, tree: HypervolumeTree,
                                candidate: PatternEvent) -> Optional[Tuple[PatternTemplate, HypervolumeNode]]:
    """Walk the hypervolume tree to find the best matching template.

    At each depth level:
    1. Extract candidate's 16D features at this depth (from enriched parent chain)
       Uses build_16d_vector() — the canonical coordinate definition from QFE.
    2. Check which child cells CONTAIN the live vector (bounding box test)
    3. If multiple cells contain it, pick nearest centroid (tiebreaker)
    4. If NO cell contains it, try nearest centroid within 2x radius (soft match)
    5. Repeat until leaf (template) or no match

    Returns (matched_template, leaf_node) or None.
    The leaf_node carries cell_min_16d/cell_max_16d for CST integrity checks.
    """
    matrix = FractalClusteringEngine.build_hypervolume_matrix(candidate)
    if matrix is None or matrix.shape[0] < 2:
        return None

    current_nodes = tree.roots
    matched_path = []  # for logging: "0.2.1.4"

    for depth in range(matrix.shape[0]):
        if not current_nodes:
            return None

        feat_16d = matrix[depth]  # (16,)

        # ── Primary: cell membership test (is vector inside bounding box?) ──
        containing_nodes = []
        for node_id, node in current_nodes.items():
            if (np.all(feat_16d >= node.cell_min_16d) and
                    np.all(feat_16d <= node.cell_max_16d)):
                dist = np.linalg.norm(feat_16d - node.centroid_16d)
                containing_nodes.append((dist, node))

        if containing_nodes:
            containing_nodes.sort(key=lambda x: x[0])
            best_node = containing_nodes[0][1]
        else:
            # ── Fallback: nearest centroid within soft margin ──
            best_node = None
            best_dist = float('inf')
            for node_id, node in current_nodes.items():
                dist = np.linalg.norm(feat_16d - node.centroid_16d)
                # Soft margin: accept if within mean cell radius
                cell_radius = np.mean(node.cell_max_16d - node.cell_min_16d) / 2
                if dist < cell_radius * 2.0 and dist < best_dist:
                    best_dist = dist
                    best_node = node
            if best_node is None:
                return None  # no match at this depth

        matched_path.append(best_node.node_id)

        if best_node.template is not None:
            return (best_node.template, best_node)

        current_nodes = best_node.children

    return None
```

**CST at entry — pass cell bounds to WaveRider** (replaces centroid + basin_mean/std from main):

```python
# In orchestrator forward pass, after tree match:
matched = self._navigate_hypervolume_tree(self.hypervolume_tree, best_candidate)
if matched:
    template, leaf_node = matched
    # Pass cell bounds for live structural integrity checks
    pos = wave_rider.open_position(
        ...,
        cst_cell_min=leaf_node.cell_min_16d,    # replaces cst_centroid
        cst_cell_max=leaf_node.cell_max_16d,    # replaces cst_basin_mean/std
        cst_ancestry=ancestry_context,
    )
```

In `wave_rider.py`, `check_structural_integrity()` becomes:
```python
# OLD (main): L2 distance from centroid, compare to basin_mean + 3*basin_std
# NEW: binary membership test — is live 16D vector inside the cell?
live_16d = QuantumFieldEngine.build_16d_vector(current_state, ancestry)
inside = np.all(live_16d >= self.cst_cell_min) and np.all(live_16d <= self.cst_cell_max)
if not inside:
    return 'structural_break'  # live vector exited the hypervolume cell
```

### Phase 5: Remove Gate 0.5 Depth Filter

**File**: `training/orchestrator.py` — line ~1375

The depth filter exists because KMeans can't handle cross-depth patterns. With the
hypervolume tree, depth is a dimension of the tree, not a gate. **Delete entirely:**

```python
# DELETE this block:
_cand_depth = getattr(p, 'depth', 6)
if _cand_depth >= 6 or _cand_depth in _DEPTH_FILTER_OUT:
    skip_headroom += 1
    _candidate_gate[id(p)] = 'gate0_5'
    ...
    continue
```

### Phase 6: Integrate with Belief Network Workers

**File**: `training/orchestrator.py` — forward pass wiring

The belief network workers already operate at individual TF levels. The tree navigation
naturally maps: each worker's TF corresponds to a depth level, and the worker's conviction
can inform the tree traversal (e.g., only follow a branch if the worker at that depth
agrees with the group's direction bias).

```python
# In forward pass, replace Gate 1 block with tree navigation:
matched_template = self._navigate_hypervolume_tree(
    self.hypervolume_tree, best_candidate)

if matched_template is None:
    # No tree path found — fall through to WORKER_BYPASS (existing logic)
    ...
else:
    # Template matched via tree — use template stats for TP/SL/direction
    active_template_id = matched_template.template_id
    ...
```

### Phase 7: Extend Decay Cascade with Cell Exit Signal

**File**: `training/timeframe_belief_network.py` — extend `get_decay_cascade()`

The decay cascade already tracks z-score drift from expected trajectory. Hypervolume adds
a new, stronger signal: **the live 16D vector left its training cell**.

```python
# In get_decay_cascade(), add cell exit as a decay factor:
def get_decay_cascade(self, cell_bounds=None, live_16d=None):
    """Extended: original z-score drift + hypervolume cell exit."""
    cascade = self._original_decay_cascade()

    # NEW: cell membership decay signal
    if cell_bounds and live_16d is not None:
        cell_min, cell_max = cell_bounds
        inside = np.all(live_16d >= cell_min) and np.all(live_16d <= cell_max)
        if not inside:
            # How far outside? Compute per-axis breach magnitude
            breach = np.maximum(cell_min - live_16d, 0) + np.maximum(live_16d - cell_max, 0)
            breach_magnitude = np.linalg.norm(breach)
            cascade['cell_exit'] = True
            cascade['cell_breach'] = breach_magnitude
            cascade['cascade_score'] += breach_magnitude * 0.5  # weight cell exit signal

    return cascade
```

This gives physics-informed exits that respond to BOTH z-score drift (existing) AND
geometric cell departure (new). The two signals are complementary:
- Z-score drift catches gradual mean-reversion failure
- Cell exit catches sudden regime change (feature vector jumps to different region)

---

## Migration: What Gets Removed vs Connected

### REMOVED (hypervolume replaces entirely)

| Component | File | What Happens |
|-----------|------|--------------|
| KMeans clustering on 16D | `fractal_clustering.py` | Replaced by `fit_hypervolume_tree()` |
| Shape taxonomy pre-grouping (`_shape_label`) | `fractal_clustering.py` | Tree structure replaces categorical pre-partitioning |
| Snowflake LONG/SHORT pre-split | `fractal_clustering.py` | Direction emerges from tree splits naturally |
| `clustering_scaler.pkl` checkpoint | `fractal_clustering.py` | Per-node scalers inside tree — no separate file |
| `templates.pkl` / `pattern_library.pkl` | `fractal_clustering.py` | Replaced by `hypervolume_tree.pkl` |
| Gate 0.5 depth filter (`depth >= 6`) | `orchestrator.py` ~line 1375 | **DELETE** — tree's adj-R² decides which depths are viable |
| Gate 1 L2 nearest-centroid matching | `orchestrator.py` | Replaced by `_navigate_hypervolume_tree()` cell membership |
| `MAX_CLUSTER_DISTANCE = 4.5` constant | `orchestrator.py` | No global distance threshold — cell bounds define match |
| `basin_mean` / `basin_std` on templates | `fractal_clustering.py` | Replaced by `cell_min_16d` / `cell_max_16d` per node |
| Snowflake z-score → LONG/SHORT routing | `orchestrator.py` | No branch routing — single tree, direction from leaf stats |
| `_long_scaler` / `_short_scaler` | `fractal_clustering.py` | Per-node scalers in tree |
| `pattern_library_long.pkl` / `_short.pkl` | checkpoints | Single `hypervolume_tree.pkl` |
| DNA tree (`FractalDNATree`) | `fractal_dna_tree.py` + `orchestrator.py` | Redundant — the hypervolume matrix path IS the identity signature |

### CONNECTED (wire existing features into hypervolume)

| Component | File | Connection Point |
|-----------|------|-----------------|
| `build_16d_vector()` | `quantum_field_engine.py` | Canonical coordinate definition for all cell operations |
| `extract_features()` | `fractal_clustering.py` | Called at every depth during tree build (keep, speed-optimized) |
| Adj-R²(mfe ~ 16D) | `fractal_clustering.py` | Node split stopping criterion + fission gain check |
| `avg_mfe_bar` / `p75_mfe_bar` | `fractal_clustering.py` | Computed on leaf template from oracle_meta `mfe_bar` |
| CST structural integrity | `fractal_discovery_agent.py` | `states_map` at all depths feeds CST; WaveRider uses cell bounds |
| `mfe_bar` oracle tracking | `fractal_discovery_agent.py` | Feeds time-scale calibration on leaf templates |
| Spectral enrichment | `fractal_discovery_agent.py` | z_score/velocity available in 16D vector |
| Gate 4 (direction confidence) | `orchestrator.py` | Fires AFTER tree match — no change, just reordered in gate chain |
| Gate 5 (direction consensus) | `orchestrator.py` | Fires AFTER tree match — no change |
| `compute_p_profitable()` | `belief_network.py` | Post-match direction quality — unchanged |
| `get_direction_consensus()` | `belief_network.py` | Post-match consensus — unchanged |
| Net pressure model | `belief_network.py` | Trail widen/tighten/urgent — unchanged |
| Decay cascade | `belief_network.py` | **Extended**: original z-score drift + new cell exit signal |
| Price-aware conviction | `belief_network.py` | Workers modulate by trade_side + profit_ticks — unchanged |
| ~~DNA tree~~ | ~~`orchestrator.py`~~ | **SKIP** — hypervolume matrix path replaces DNA identity |
| Pre-converted coefficients | `belief_network.py` | Performance optimization — unchanged |
| Scaler mean/scale extraction | `belief_network.py` | Performance optimization — unchanged |

### Gate Chain After Hypervolume

```
Gate 0:  Structural rules (headroom, momentum, etc.)        — UNCHANGED
Gate 0.5: Depth filter                                       — REMOVED
Gate 1:  Template matching (L2 → hypervolume tree walk)      — REPLACED
Gate 2:  Belief conviction                                   — UNCHANGED
Gate 3:  Entry conviction threshold                          — UNCHANGED
Gate 3.5: Risk:Reward                                        — UNCHANGED
Gate 4:  Direction confidence (logistic prob)                 — UNCHANGED (post-match)
Gate 5:  Direction consensus (multi-TF)                      — UNCHANGED (post-match)
```

---

## Data Flow Summary

```
TRAINING:
  fractal_discovery_agent scans all TFs
    → _build_parent_chain() stores features_16d at each ancestor    [Phase 1]
    → _consult_oracle() computes mfe_bar + structural integrity     [existing on main]
    → patterns (16F × depth matrices)
      → fit_hypervolume_tree()                                      [Phase 3]
        → depth 0: regression → residuals → cluster → root groups
          → depth 1: regression within each root → sub-groups
            → ... adj-R² stops when features predict MFE well
              → leaf templates with oracle stats + avg_mfe_bar/p75

  Checkpoint: hypervolume_tree.pkl (contains all nodes, scalers, cell bounds)

FORWARD PASS:
  candidate pattern arrives at Gate 0 (structural rules)            [unchanged]
    ↓ passes Gate 0
  build_hypervolume_matrix() → 16F × depth matrix                   [Phase 4]
    → _navigate_hypervolume_tree()
      → depth 0: cell membership test → "Group A"
        → depth 1: within A, membership → "A3"
          → depth 2: within A3, membership → "A3.7"
            → ... → leaf node → template + cell_min/cell_max
    ↓ template matched (or WORKER_BYPASS if no path)
  Gate 2: belief conviction                                         [unchanged]
  Gate 3: entry conviction                                          [unchanged]
  Gate 3.5: R:R check                                               [unchanged]
  Gate 4: direction confidence (logistic prob)                       [existing on main]
  Gate 5: direction consensus (multi-TF)                             [existing on main]
    ↓ all gates pass
  WaveRider.open_position(cell_min, cell_max, ...)                  [Phase 4]

LIVE POSITION:
  Each bar:
    → build_16d_vector(current_state, ancestry)                      [existing on main]
    → CST: is live vector inside cell bounds?                        [Phase 4]
    → Decay cascade: z-score drift + cell exit signal                [Phase 7]
    → Net pressure: widen/tighten/urgent                             [existing on main]
    → Price-aware conviction modulation                              [existing on main]
    → Time-exhaustion: bar_count vs avg_mfe_bar/p75_mfe_bar          [existing on main]
```

## What Changes

| Component | Current (KMeans on main) | New (Hypervolume Tree) |
|-----------|--------------------------|------------------------|
| Feature input | 16D point per pattern | 16D × depth matrix per pattern |
| Template definition | 16D centroid + basin_mean/std | Path through tree + cell bounding box per node |
| Matching (Gate 1) | L2 distance < 4.5 in flat 16D | Tree walk: cell membership (bounding box) per depth |
| Depth handling | Gate 0.5 blocks depth≥6 | Depth = tree level (adj-R² decides viability) |
| Split criterion | KMeans + shape taxonomy | Regression residuals + adj-R²(mfe) stopping |
| Scaler | Global (mismatch bug) | Per-node local (impossible to mismatch) |
| CST integrity | L2 from centroid vs basin | Binary: is live 16D inside cell bounds? |
| Decay cascade | z-score drift only | z-score drift + cell exit signal |
| Shape info | Lost (compressed to point) | Preserved (tree path = 16F×12D hypergeometry) |
| Identity | DNA tree (separate structure) | Matrix path IS the identity (DNA tree removed) |
| Checkpoints | templates.pkl + scaler.pkl + library_long/short.pkl | Single hypervolume_tree.pkl |
| Worker integration | None | Each worker navigates one tree level |
| Interpretability | Centroid ID number | Path label: "0.2.1.4" = regime chain |

## Files to Modify

| File | Action | Details |
|------|--------|---------|
| `training/fractal_discovery_agent.py` | MODIFY | Enrich `_build_parent_chain()` with `features_16d` at each ancestor |
| `training/fractal_clustering.py` | MAJOR REWRITE | New classes `HypervolumeNode`, `HypervolumeTree`; new methods `build_hypervolume_matrix()`, `fit_hypervolume_tree()`, `_split_at_depth()`; absorb adj-R² from main; keep `extract_features()` unchanged; REMOVE: KMeans clustering, shape taxonomy, snowflake split, `_long_scaler`/`_short_scaler` |
| `training/orchestrator.py` | MODIFY | New `_navigate_hypervolume_tree()` with cell membership; REMOVE: Gate 0.5, Gate 1 L2 matching, `MAX_CLUSTER_DISTANCE`, snowflake branch routing, DNA tree wiring; KEEP: Gates 0/2/3/3.5/4/5 unchanged |
| `training/wave_rider.py` | MODIFY | CST check: replace `basin_mean/std` with `cell_min/cell_max` bounding box test |
| `training/timeframe_belief_network.py` | MODIFY | Extend `get_decay_cascade()` with cell exit signal |
| `training/fractal_dna_tree.py` | REMOVE USAGE | Stop loading/saving DNA tree in orchestrator (hypervolume matrix path replaces it) |

## What NOT to Change

- The 16D feature vector definition (`extract_features`, `build_16d_vector`) — this IS the
  hypervolume coordinate system. Used at every depth level.
- Oracle attribution logic (unchanged — receives template from tree instead of KMeans)
- Gate 0 structural rules (headroom, momentum, etc.)
- Gate 2/3/3.5 (belief, conviction, R:R)
- Gate 4/5 (direction confidence/consensus) — post-match gates, algorithm-independent
- Belief network worker architecture (10 TF workers, conviction, etc.)
- Net pressure model (trail widen/tighten/urgent)
- Price-aware conviction modulation
- Report generation (adapt to show tree path instead of centroid ID)
- The data pipeline, ATLAS structure, or 1s inner loop
- `tools/run_benchmark.py` and analysis tooling

## CUDA Acceleration

The regression + residual clustering at each tree node is inherently GPU-friendly:

- **Multivariate OLS**: `(X'X)⁻¹X'Y` on N×16 matrices → `torch.linalg.lstsq` on CUDA
- **Residual computation**: `Y - X@B` → simple CUDA matrix subtract
- **KMeans on residuals**: already implemented in `training/cuda_kmeans.py`

At upper tree levels (depth 0-2), groups can contain 100K+ patterns — this is where
GPU acceleration matters most. Use `torch.cuda` for the regression, feed residuals
into existing `CUDAKMeans`.

```python
# CUDA regression at each node (pseudocode):
import torch

feat_gpu = torch.tensor(feat_array, device='cuda', dtype=torch.float32)
parent_gpu = torch.tensor(parent_array, device='cuda', dtype=torch.float32)

# Multivariate OLS on GPU
B = torch.linalg.lstsq(parent_gpu, feat_gpu).solution  # (16, 16)
predicted = parent_gpu @ B
residuals = feat_gpu - predicted  # stays on GPU for KMeans
residuals_cpu = residuals.cpu().numpy()  # only move to CPU when needed
```

The tree build is a one-time training cost. Forward pass navigation is CPU-only
(single 16D comparison per depth level — trivial).

## Dependencies

- numpy (existing) — `np.linalg.norm`, standard operations
- sklearn (existing) — `StandardScaler`
- torch with CUDA (existing) — `torch.linalg.lstsq` for GPU regression
- `training/cuda_kmeans.py` (existing) — KMeans on residuals
- No new dependencies

## Checkpoint Compatibility

The tree replaces `templates.pkl` + `clustering_scaler.pkl` with a single
`hypervolume_tree.pkl`. **First run MUST use `--fresh`** to build the tree from scratch.
The tree checkpoint contains all per-node scalers, so there is no separate scaler file
that can go stale.

## Verification

### Test Dataset

`DATA/ATLAS_1MONTH/` — January 2025, all 14 TFs, 39MB (1/10th of full ATLAS).
Use this for all development testing. It has enough patterns to build a meaningful tree
but runs ~10x faster than full ATLAS.

### Validation Steps

1. Syntax check: `python -c "import training.orchestrator"`
2. **Smoke test** (1 day, ~3 seconds): `python -m training.orchestrator --fresh --data DATA/ATLAS_1DAY`
   - Should build hypervolume tree (even if small) and navigate it
   - Gate 0.5 should NOT exist — all depths should produce candidates
3. **Dev test** (1 month, ~2-3 min): `python -m training.orchestrator --fresh --data DATA/ATLAS_1MONTH`
   - Tree should have meaningful depth structure (multiple root groups, branches)
   - Template matching should work (some trades matched via tree, not all WORKER_BYPASS)
   - Check `reports/is/oracle_trade_log.csv` for tree path labels
   - Verify adj-R² stopping produced leaf cells with varying depth
4. Full IS (10 months): `python -m training.orchestrator --fresh`
   - Compare against pre-snowflake baseline: 50.8% WR, 1,903 trades, $9,334
   - Expect more trades than current main (141) since depth filter is gone
   - Expect template match rate > 50% (current: 0% — all WORKER_BYPASS)
   - Tree path labels should appear in oracle_trade_log.csv

## Success Criteria

- Tree builds successfully with meaningful depth structure (not all patterns in one leaf)
- Template matching works across all depths (no centroid space mismatch)
- Gate 0.5 depth filter removed entirely
- Trade count > 500 (current: 141 due to depth blocking)
- Template match rate > 50% (current: 0% — all WORKER_BYPASS)
- Phase 5 strategy report shows templates with actual trades (current: all 0)
- Tree path labels are interpretable (e.g., "0.2.1.4" maps to a recognizable regime)

## Priority

**CRITICAL** — The current clustering is completely non-functional (0% template matches).
This is the most impactful architectural fix available.
