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
    member_count: int                      # patterns that pass through this node
    children: Dict[int, 'HypervolumeNode'] # child_id -> child node (next depth)
    template: Optional[PatternTemplate]    # leaf nodes get a template with oracle stats
    node_id: str                           # path string e.g. "A3.7.2"
    radius: float                          # max distance of members from centroid (for matching)

    # Per-depth scaler (fitted on residuals at this depth within this group)
    scaler: Optional[StandardScaler] = None
    regression_r2: float = 0.0             # how well parent predicts this level
    branch_tightness: float = 0.0          # residual variance (lower = tighter group)


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

    # ── Step 1: Fit 16D regression to find expected centroid ──────────────
    # If this is a sub-group, regress current depth features against
    # parent depth features to capture the expected evolution.
    # For root level (depth 0), the regression centroid is simply the mean.
    if depth > 0:
        # Use parent-depth features as predictors for current-depth features
        parent_feat = []
        for idx in valid_indices:
            mat = matrices[idx]
            parent_feat.append(mat[depth - 1])  # parent level features
        parent_array = np.array(parent_feat)  # (N, 16)

        # Multivariate OLS: feat_current = parent_feat @ B + intercept
        from sklearn.linear_model import LinearRegression
        reg = LinearRegression()
        reg.fit(parent_array, feat_array)
        predicted = reg.predict(parent_array)  # expected 16D at this depth
        residuals = feat_array - predicted      # (N, 16) deviation from expected
        regression_centroid = feat_array.mean(axis=0)
        regression_r2 = reg.score(parent_array, feat_array)
    else:
        # Root level: centroid is mean, residuals are deviations from mean
        regression_centroid = feat_array.mean(axis=0)
        residuals = feat_array - regression_centroid  # (N, 16)
        regression_r2 = 0.0

    # ── Step 2: Cluster the residuals ────────────────────────────────────
    # Patterns with similar residuals deviate from the regression in the
    # same way — they share the same hypervolume sub-structure.
    scaler = StandardScaler()
    residuals_scaled = scaler.fit_transform(residuals)

    # Check if group is already tight (low residual variance → single branch)
    residual_var = np.mean(np.var(residuals_scaled, axis=0))
    if residual_var < 0.3:  # tight group, no need to split
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

        # Centroid in feature space (not residual space) for matching
        cluster_feat = feat_array[member_mask]
        centroid = cluster_feat.mean(axis=0)

        # Residual tightness = branch quality score
        cluster_residuals = residuals_scaled[member_mask]
        branch_tightness = np.mean(np.var(cluster_residuals, axis=0))

        # Radius for matching threshold
        radius = np.percentile(
            np.linalg.norm(cluster_residuals -
                           cluster_residuals.mean(axis=0), axis=1), 95)

        # Recurse to next depth
        children = self._split_at_depth(
            member_indices, matrices, patterns,
            depth + 1, node_id + ".", min_group_size
        )

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

        nodes[cluster_id] = HypervolumeNode(
            depth=depth,
            centroid_16d=centroid,
            member_count=len(member_indices),
            children=children,
            template=template,
            node_id=node_id,
            radius=radius,
            scaler=scaler,          # residual scaler for this node
            regression_r2=regression_r2,
            branch_tightness=branch_tightness,
        )

    return nodes
```

### Phase 4: Navigate the Tree (Forward Pass)

**File**: `training/orchestrator.py` — replace Gate 1 matching

```python
def _navigate_hypervolume_tree(self, tree: HypervolumeTree,
                                candidate: PatternEvent,
                                max_dist: float = 4.5) -> Optional[PatternTemplate]:
    """Walk the hypervolume tree to find the best matching template.

    At each depth level:
    1. Extract candidate's 16D features at this depth (from parent chain)
    2. Compare to child centroids within current node
    3. Follow the nearest child (if within max_dist)
    4. Repeat until leaf (template) or no match

    Returns the matched PatternTemplate or None if no path found.
    """
    matrix = FractalClusteringEngine.build_hypervolume_matrix(candidate)
    if matrix is None or matrix.shape[0] < 2:
        return None

    # Start at root level
    current_nodes = tree.roots

    for depth in range(matrix.shape[0]):
        if not current_nodes:
            return None  # no branches at this depth

        # Candidate's 16D features at this depth
        feat_16d = matrix[depth].reshape(1, -1)

        # Find nearest node at this depth
        best_node = None
        best_dist = float('inf')
        for node_id, node in current_nodes.items():
            if node.scaler is not None:
                feat_scaled = node.scaler.transform(feat_16d)
                centroid_scaled = node.scaler.transform(
                    node.centroid_16d.reshape(1, -1))
                dist = np.linalg.norm(feat_scaled - centroid_scaled)
            else:
                dist = np.linalg.norm(feat_16d - node.centroid_16d.reshape(1, -1))
            if dist < best_dist:
                best_dist = dist
                best_node = node

        if best_dist > max_dist or best_node is None:
            return None  # no match at this depth

        # If leaf, return its template
        if best_node.template is not None:
            return best_node.template

        # Otherwise, descend into children
        current_nodes = best_node.children

    return None
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

## Data Flow Summary

```
TRAINING:
  patterns (16F x depth matrices)
    → fit_hypervolume_tree()
      → depth 0: KMeans on 16D → root groups
        → depth 1: KMeans on 16D within each root → sub-groups
          → depth 2: ...
            → ... → leaf templates with oracle stats

FORWARD PASS:
  candidate pattern
    → build_hypervolume_matrix() → 16F x depth matrix
    → _navigate_hypervolume_tree()
      → depth 0: match 16D → "Group A"
        → depth 1: match 16D within A → "A3"
          → depth 2: match 16D within A3 → "A3.7"
            → ... → leaf template → load stats → trade
```

## What Changes

| Component | Current (KMeans) | New (Hypervolume Tree) |
|-----------|-----------------|------------------------|
| Feature input | 16D point per pattern | 16D × depth matrix per pattern |
| Template definition | 16D centroid | Path through tree of 16D centroids |
| Matching | L2 in flat 16D | Tree walk: L2 in 16D per depth level |
| Depth handling | Gate 0.5 blocks depth≥6 | Depth = tree level (no blocking) |
| Scaler | Global, must match exactly | Per-node local (no mismatch possible) |
| Shape info | Lost (compressed to point) | Preserved (tree path = volumetric structure) |
| Worker integration | None | Each worker navigates one tree level |
| Interpretability | Centroid ID number | Path label: "A3.7.2" = meaningful |

## Files to Modify

1. `training/fractal_discovery_agent.py` — Enrich `_build_parent_chain()` with `features_16d`
2. `training/fractal_clustering.py` — New classes `HypervolumeNode`, `HypervolumeTree`;
   new methods `build_hypervolume_matrix()`, `fit_hypervolume_tree()`, `_split_at_depth()`;
   keep `extract_features()` unchanged
3. `training/orchestrator.py` — New method `_navigate_hypervolume_tree()`;
   replace Gate 1 matching; remove Gate 0.5 depth filter

## What NOT to Change

- The 16D feature extraction function (`extract_features`) — still the basis, now used
  at every depth level instead of just the detection level
- Oracle attribution logic (unchanged — just receives template from tree instead of KMeans)
- Belief network / wave rider / exit mechanics
- Report generation (adapt to show tree path instead of centroid ID)
- Gate 0 structural rules (still valid — headroom, momentum, etc.)

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

1. Syntax check: `python -c "import training.orchestrator"`
2. Quick validation: `python -m training.orchestrator --fresh --data DATA/ATLAS_1DAY`
   - Should build hypervolume tree and navigate it during forward pass
   - Gate 0.5 should NOT exist — all depths should produce candidates
3. Full IS: `python -m training.orchestrator --fresh`
   - Compare against pre-snowflake baseline: 50.8% WR, 1,903 trades, $9,334
   - Expect more trades than current main (141) since depth filter is gone
   - Expect template matching to work (no more 100% WORKER_BYPASS)
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
