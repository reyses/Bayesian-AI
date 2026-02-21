# JULES TASK: Snowflake Hierarchical DNA Tree — Gate 1 Integration

## Architecture Vision

The system implements a **Snowflake Schema** for Hierarchical Bayesian inference:

```
L3 MACRO (1h/4h/1D)   — sets the prior:    "High vol NY session, baseline WR=60%"
    ↓
L2 MESO  (1m-5m)      — updates prior:     "Structural Drive ADX>30, WR shifts to 75%"
    ↓
L1 MICRO (1s-15s)     — pulls the trigger: "Roche Snap -3σ, posterior WR=92%"
```

The **FractalDNATree** already implements this hierarchy — it builds a tree where each level
clusters patterns by their features at that TF scale. The DNA path `1h:2|30m:0|15m:3|5m:1`
IS the hierarchical state path through L3→L2→L1.

**The current gap**: Gate 1 in `orchestrator.py` still uses FLAT 14D centroid matching — all
TF features in one vector → dimensional dilution (the exact curse of dimensionality the
snowflake is designed to prevent). The DNA tree is built but only used for reporting.

**The fix**: Make Gate 1 USE the DNA tree for matching (traverse L3→L2→L1), not flat centroids.

---

## Mathematical Foundation

Instead of: `P(Win | flat_14D_vector)` → flat centroid distance

Use: `P(Win | Micro, Meso, Macro) = P(Micro | Meso) × P(Meso | Macro) × P(Win) / Evidence`

At each DNA tree level, the node already stores:
- `win_rate` — conditional WR at this branch
- `mfe_coeff` / `mfe_intercept` — regression model for MFE at this level
- `member_count` — how many patterns reached this node

The leaf node's `win_rate` IS the posterior probability after traversing all levels.

Direction: from `dir_coeff` logistic regression at the matched template cluster (Gate 1 already
has this via `lib_entry.get('dir_coeff')`). The DNA tree does NOT encode direction — the
fractal arrangement itself carries directional geometry, read by the regression.

---

## Part 1: Unified DNA Tree (Remove Direction Split)

### `training/fractal_dna_tree.py`

**Remove the LONG/SHORT split — build ONE unified tree:**

**Change `FractalDNATree.__init__`:**
```python
def __init__(self, n_clusters_per_level: int = 5):
    self.n_clusters_per_level = n_clusters_per_level
    self.root: Optional[TreeNode] = None          # Single unified root
    self._dna_index: Dict[str, TreeNode] = {}
```

**Change `FractalDNATree.fit`:**
```python
def fit(self, patterns: list) -> None:
    valid = [p for p in patterns if getattr(p, 'oracle_marker', 0) != 0]
    self.root = self._build_branch(valid)
    self._build_dna_index()
```

**Change `_build_branch`** — remove `direction` parameter:
```python
def _build_branch(self, patterns: list) -> TreeNode:
    root = TreeNode(node_id='root', timeframe='root',
                    centroid=np.zeros(1), member_count=len(patterns))
    self._split_by_tf_level(root, patterns, tf_level_idx=0)
    self._aggregate_oracle_stats(root, patterns)
    return root
```

**Change `_split_by_tf_level`** — remove `direction` parameter throughout.
Node IDs become: `"1h:3"`, `"30m:0"` (no `L|`/`S|` prefix).

**Change `_build_dna_index`:**
```python
def _build_dna_index(self) -> None:
    def _traverse(node, path_so_far):
        if not node.children:
            dna_key = '|'.join(path_so_far)
            self._dna_index[dna_key] = node
            return
        for child_id, child in node.children.items():
            _traverse(child, path_so_far + [child_id])
    if self.root:
        _traverse(self.root, [])
```

**Change `PatternDNA`** — remove `direction` field:
```python
@dataclass
class PatternDNA:
    path: List[str]   # ['1h:3', '30m:0', '15m:2', '5m:1']

    @property
    def key(self) -> str:
        return '|'.join(self.path)

    @property
    def depth(self) -> int:
        return len(self.path)

    def parent_dna(self) -> Optional['PatternDNA']:
        if len(self.path) <= 1:
            return None
        return PatternDNA(path=self.path[:-1])
```

**Change `match`** — remove direction routing, start from `self.root`:
```python
def match(self, p) -> tuple:
    if self.root is None:
        return None, None, 0.0
    path = []
    current = self.root
    total_dist = 0.0
    for tf in self.TF_ORDER:
        if not current.children:
            break
        is_leaf = (tf == getattr(p, 'timeframe', '15s'))
        feat = np.array(self._extract_tf_features(p, tf, is_leaf=is_leaf))
        best_child, best_dist, best_child_id = None, float('inf'), None
        for child_id, child in current.children.items():
            if child.timeframe != tf:
                continue
            dist = float(np.linalg.norm(feat - child.centroid))
            if dist < best_dist:
                best_dist, best_child, best_child_id = dist, child, child_id
        if best_child is None:
            break
        path.append(best_child_id)
        total_dist += best_dist
        current = best_child
        if is_leaf:
            break
    if not path:
        return None, None, 0.0
    dna = PatternDNA(path=path)
    avg_dist = total_dist / len(path)
    confidence = 1.0 / (1.0 + avg_dist)
    return dna, current, confidence
```

**Change `TreeNode`** — remove `direction` field:
```python
@dataclass
class TreeNode:
    node_id: str
    timeframe: str
    centroid: np.ndarray
    member_count: int
    children: Dict[str, 'TreeNode'] = field(default_factory=dict)
    leaf_pattern_ids: List[int] = field(default_factory=list)
    win_rate: float = 0.0
    mean_mfe_ticks: float = 0.0
    mean_mae_ticks: float = 0.0
    p75_mfe_ticks: float = 0.0
    p25_mae_ticks: float = 0.0
    expectancy: float = 0.0
    mfe_coeff: Optional[List[float]] = None
    mfe_intercept: float = 0.0
```

Keep `_extract_tf_features`, `_aggregate_oracle_stats`, `_cluster_level`, `get_stats_for_dna`,
`get_top_paths` — same logic, just remove direction parameters.

---

## Part 2: Gate 1 — Use DNA Tree Traversal Instead of Flat Centroids

### `training/orchestrator.py` — forward pass Gate 1

**Currently (flat centroid match):**
```python
# Extract 14D features
features = np.array([FractalClusteringEngine.extract_features(p)])
feat_scaled = self.scaler.transform(features)
dists = np.linalg.norm(centroids_scaled - feat_scaled, axis=1)
nearest_idx = np.argmin(dists)
dist = dists[nearest_idx]
tid = valid_template_ids[nearest_idx]
```

**Replace with DNA tree traversal** (when DNA tree is available):
```python
if self.dna_tree is not None:
    # Hierarchical Bayesian match: L3(1h) → L2(5m) → L1(15s)
    pattern_dna, dna_node, dna_conf = self.dna_tree.match(p)
    if pattern_dna is not None and dna_conf > 0.1:
        # Use DNA node stats as the hierarchical posterior
        _dna_win_rate  = dna_node.win_rate
        _dna_mfe_ticks = dna_node.mean_mfe_ticks
        # Still need a template for params (TP/SL/trail) — find nearest flat centroid
        # but now use dna_conf as a quality gate instead of raw euclidean distance
        features = np.array([FractalClusteringEngine.extract_features(p)])
        feat_scaled = self.scaler.transform(features)
        dists = np.linalg.norm(centroids_scaled - feat_scaled, axis=1)
        nearest_idx = np.argmin(dists)
        dist = dists[nearest_idx]
        tid = valid_template_ids[nearest_idx]
        # Gate: reject if flat dist too large (no template match) OR dna_conf too low
        _gate_dist = dist  # keep existing dist threshold logic
    else:
        # Fallback to flat centroid match if DNA tree not yet available
        features = np.array([FractalClusteringEngine.extract_features(p)])
        feat_scaled = self.scaler.transform(features)
        dists = np.linalg.norm(centroids_scaled - feat_scaled, axis=1)
        nearest_idx = np.argmin(dists)
        dist = dists[nearest_idx]
        tid = valid_template_ids[nearest_idx]
        pattern_dna = dna_node = None
        dna_conf = 0.0
else:
    # No DNA tree — flat match only
    features = np.array([FractalClusteringEngine.extract_features(p)])
    feat_scaled = self.scaler.transform(features)
    dists = np.linalg.norm(centroids_scaled - feat_scaled, axis=1)
    nearest_idx = np.argmin(dists)
    dist = dists[nearest_idx]
    tid = valid_template_ids[nearest_idx]
    pattern_dna = dna_node = None
    dna_conf = 0.0
```

**Log `dna_conf` and DNA node stats** alongside existing DNA path logging.

**Direction** — keep existing `dir_coeff` logistic regression (already implemented, unchanged):
```python
_dir_coeff = lib_entry.get('dir_coeff')
if _dir_coeff is not None:
    _logit = np.dot(_live_scaled, np.array(_dir_coeff)) + lib_entry.get('dir_intercept', 0.0)
    _p_long = 1.0 / (1.0 + np.exp(-_logit))
    side = 'long' if _p_long >= 0.5 else 'short'
else:
    _long_bias = lib_entry.get('long_bias', 0.5)
    side = 'long' if _long_bias >= 0.5 else 'short'
```

---

## Part 3: DNA Logging Cleanup

### `training/orchestrator.py` — any reference to `pattern_dna.direction`

Search for `pattern_dna.direction` and `dna.direction` — remove or replace.
The DNA path key is now `"1h:3|30m:0|15m:2"` (no `L|`/`S|` prefix).

### Report section "TOP 10 DNA PATHS"
DNA keys now show: `1h:1|30m:0|15m:0|5m:0` instead of `S|1h:1|30m:0|...`

---

## What NOT to Change
- `_extract_tf_features` — unchanged
- `_aggregate_oracle_stats` — unchanged
- `_cluster_level` — unchanged
- `register_template_logic` — unchanged (still saves `dir_coeff`, `long_bias`)
- Phase 3 template optimization — unchanged
- Gate 1 distance threshold (dist > 4.5) — unchanged
- `dir_coeff` direction routing — keep as is, already correct

---

## Current Baseline (run 2026-02-20 20:33:05)
- 954 trades, 61.3% WR, -$31,011
- Wrong direction: 44.7% (426 trades) — dir_coeff not yet fitted (needs --fresh unified clustering)
- `long_bias` fallback currently routing direction
- Gate 0 STRUCTURAL_DRIVE headroom: 56.1% of FN — main bottleneck
- DNA paths still show L|/S| prefix (old tree from last run)

## Expected Outcome After This Task
- DNA tree: single root, paths like `1h:2|30m:0|15m:3`
- Gate 1 uses DNA traversal confidence alongside flat dist
- Direction from `dir_coeff` → wrong direction rate drops from 44.7% → ~7%
- DNA node win_rate feeds conviction scoring (replaces flat template WR)
- `fractal_dna_tree.pkl` single file (no long/short split)
