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

Direction: from the **worker belief network** — workers read live momentum context (dir_prob,
wave_maturity, conviction per TF) and decide which way price is moving from this geometry.
The DNA tree is purely structural — same `1h:2|30m:0` node can be LONG or SHORT depending
on current worker state. `dir_coeff` stays in pattern_library for diagnostics only.

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

**Change `TreeNode`** — remove `direction` field, add energy/volume stats for decay traversal:
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
    mean_adx: float = 0.0         # avg structural energy at this node (for decay traversal)
    mean_rel_volume: float = 1.0  # avg relative volume at this node (participation proxy)
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

**Direction** — from the worker belief network, NOT from the DNA tree and NOT from dir_coeff.

The DNA tree encodes **where** in the fractal geometry the market is (shape/structure).
The workers encode **what the market is doing right now** in that geometry (momentum/context).
The same geometric pattern `1h:2|30m:0` can be LONG at 9:30am (fresh wave) or SHORT at 2pm
(exhausted wave) — identical geometry, different worker belief.

Worker-based direction:
```python
# Use belief network direction (aggregated across TF workers)
side = _belief.direction  # 'long' or 'short' from worker conviction-weighted vote
# dir_coeff stays in pattern_library as a shape-based prior for diagnostics only
# long_bias fallback is removed — if no worker belief, skip the trade
if side is None:
    continue  # no worker consensus — do not force a direction
```

`dir_coeff` and `long_bias` remain stored in `pattern_library` for offline analysis but
are no longer used to route the live direction decision.

---

## Part 3: DNA Logging Cleanup

### `training/orchestrator.py` — any reference to `pattern_dna.direction`

Search for `pattern_dna.direction` and `dna.direction` — remove or replace.
The DNA path key is now `"1h:3|30m:0|15m:2"` (no `L|`/`S|` prefix).

### Report section "TOP 10 DNA PATHS"
DNA keys now show: `1h:1|30m:0|15m:0|5m:0` instead of `S|1h:1|30m:0|...`

---

## Part 4: Full Geometric Feature Encoding (parent_chain fix + volume)

### The Problem

`_build_parent_chain` in `fractal_discovery_agent.py` only stores `z`, `mom`, `coh` per
ancestor entry. But `_extract_tf_features` in `fractal_dna_tree.py` tries to read
`adx`, `hurst`, `dmi_plus`, `dmi_minus`, `pid`, `osc_coh` from ancestors — they silently
fall back to defaults (0.0) for every non-leaf level.

Result: the tree's higher-level nodes (`1h`, `30m`, `15m`) cluster on nearly just `z` + `mom`.
The rich geometry only works at the leaf level. This defeats the hierarchical structure.

### Fix `_build_parent_chain` in `training/fractal_discovery_agent.py`

```python
def _build_parent_chain(self, p: PatternEvent) -> List[Dict]:
    s = getattr(p, 'state', None)
    chain_entry = {
        'tf':           p.timeframe,
        'type':         p.pattern_type,
        'z':            p.z_score,
        'velocity':     getattr(p, 'velocity', 0.0),
        'mom':          p.momentum,
        'coh':          p.coherence,
        'adx':          getattr(s, 'adx_strength', 0.0)          if s else 0.0,
        'hurst':        getattr(s, 'hurst_exponent', 0.5)        if s else 0.5,
        'dmi_plus':     getattr(s, 'dmi_plus', 0.0)              if s else 0.0,
        'dmi_minus':    getattr(s, 'dmi_minus', 0.0)             if s else 0.0,
        'pid':          getattr(s, 'term_pid', 0.0)              if s else 0.0,
        'osc_coh':      getattr(s, 'oscillation_coherence', 0.0) if s else 0.0,
        'rel_volume':   getattr(s, 'rel_volume', 1.0)            if s else 1.0,
        'timestamp':    p.timestamp,
        'oracle_marker': getattr(p, 'oracle_marker', 0),
    }
    return [chain_entry] + (p.parent_chain or [])
```

### Add `rel_volume` to `ThreeBodyQuantumState` in `core/three_body_state.py`

Relative volume = current bar volume / rolling N-bar mean volume.
Scale-invariant (1.0 = average, 2.0 = double volume, 0.5 = half volume).

```python
@dataclass(frozen=True)
class ThreeBodyQuantumState:
    # ... existing fields ...
    rel_volume: float = 1.0   # current vol / rolling mean vol (N=20 bars)
```

### Compute `rel_volume` in `core/quantum_field_engine.py`

In `_batch_compute_cpu` (and GPU kernel), after computing momentum:
```python
# Rolling mean volume (20-bar window)
vol_mean = np.convolve(volumes, np.ones(20)/20, mode='same')
vol_mean = np.maximum(vol_mean, 1e-9)  # avoid div/0
rel_vol  = volumes / vol_mean           # shape (N,)
```
Store `rel_volume=float(rel_vol[bar_idx])` when constructing `ThreeBodyQuantumState`.

### Update `_extract_tf_features` in `training/fractal_dna_tree.py`

Expand feature vector from 8D to 10D by splitting `dmi_diff` into separate components
and adding `rel_volume`:

**Leaf (pattern's own state):**
```python
s = getattr(p, 'state', None)
if s:
    return [
        abs(getattr(p, 'z_score', 0.0)),
        np.log1p(abs(getattr(p, 'velocity', 0.0))),
        np.log1p(abs(getattr(p, 'momentum', 0.0))),
        getattr(s, 'adx_strength', 0.0)          / 100.0,
        getattr(s, 'dmi_plus', 0.0)               / 100.0,   # ← split
        getattr(s, 'dmi_minus', 0.0)              / 100.0,   # ← split
        getattr(s, 'hurst_exponent', 0.5),
        getattr(s, 'term_pid', 0.0),
        getattr(s, 'oscillation_coherence', 0.0),
        np.log1p(max(getattr(s, 'rel_volume', 1.0), 0.0)),   # ← new
    ]
```

**Ancestor (from parent_chain):**
```python
return [
    abs(ancestor.get('z', 0.0)),
    np.log1p(abs(ancestor.get('velocity', 0.0))),
    np.log1p(abs(ancestor.get('mom', 0.0))),
    ancestor.get('adx', 0.0)       / 100.0,
    ancestor.get('dmi_plus', 0.0)  / 100.0,   # ← split
    ancestor.get('dmi_minus', 0.0) / 100.0,   # ← split
    ancestor.get('hurst', 0.5),
    ancestor.get('pid', 0.0),
    ancestor.get('osc_coh', 0.0),
    np.log1p(max(ancestor.get('rel_volume', 1.0), 0.0)),  # ← new
]
```

Zero-feature fallback changes from `[0.0] * 8` to `[0.0] * 10`.

### Add `mean_adx` and `mean_rel_volume` to `TreeNode`

For the probability decay traversal (energy cascade analysis):

```python
@dataclass
class TreeNode:
    # ... existing fields ...
    mean_adx:        float = 0.0   # avg ADX of patterns at this node
    mean_rel_volume: float = 1.0   # avg relative volume at this node
```

Compute in `_aggregate_oracle_stats`:
```python
adx_vals = [getattr(getattr(p,'state',None),'adx_strength',0.0) for p in patterns]
vol_vals  = [getattr(getattr(p,'state',None),'rel_volume', 1.0) for p in patterns]
node.mean_adx        = float(np.mean(adx_vals)) if adx_vals else 0.0
node.mean_rel_volume = float(np.mean(vol_vals))  if vol_vals else 1.0
```

These fields enable the traversal-time energy decay calculation:
```
energy_ratio[i] = node_i.mean_adx / parent_node.mean_adx
volume_ratio[i] = node_i.mean_rel_volume / parent_node.mean_rel_volume
reversal_proximity = 1.0 - product(energy_ratio[i] for i in path)
```

---

## What NOT to Change
- `_aggregate_oracle_stats` core logic — unchanged (just add adx/vol aggregation)
- `_cluster_level` — unchanged
- `register_template_logic` — unchanged (still saves `dir_coeff`, `long_bias` for diagnostics)
- Phase 3 template optimization — unchanged
- Gate 1 distance threshold (dist > 4.5) — unchanged
- Worker belief network direction logic — unchanged (this IS the direction source now)

## What to Remove
- `dir_coeff` / `long_bias` direction routing at Gate 1 — workers decide direction
- `long_bias` fallback — if no worker belief, skip rather than guess

---

## Current Baseline (run 2026-02-20 20:33:05)
- 954 trades, 61.3% WR, -$31,011
- Wrong direction: 44.7% (426 trades) — dir_coeff not yet fitted (needs --fresh unified clustering)
- `long_bias` fallback currently routing direction
- Gate 0 STRUCTURAL_DRIVE headroom: 56.1% of FN — main bottleneck
- DNA paths still show L|/S| prefix (old tree from last run)

## Expected Outcome After This Task
- DNA tree: single unified root, paths like `1h:2|30m:0|15m:3` (no L|/S| prefix)
- Tree is purely geometric — encodes where in the fractal structure, not what direction
- Gate 1 uses DNA traversal confidence alongside flat dist
- Direction from worker belief network — workers read current momentum context
- DNA node win_rate feeds conviction scoring (replaces flat template WR)
- `fractal_dna_tree.pkl` single file (no long/short split)
- Wrong direction rate drops once workers have full context (not shape-baked direction)
