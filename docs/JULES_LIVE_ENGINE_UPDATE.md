# Jules Spec: Live Engine Full Parity Update (Phase B)

## Context

The `live/live_engine.py` was updated with Phase A minimum-viable fixes (scaler
fallback, trail params, direction confidence gate, screening gates, client kwarg).
It now boots and loads checkpoints. **Phase B** brings it to full parity with the
orchestrator forward pass (`training/orchestrator.py` lines 1500-2000).

**Scope**: ~300 lines across `live/live_engine.py` (primary) + minor touchups in
`live/config.py` and `live/launcher.py`.

---

## 1. Replace L2 Centroid Matching with HypervolumeTree Navigation

**Current** (live_engine.py ~300-330): L2 distance to `_centroids_scaled` matrix.
**Target**: Navigate `HypervolumeTree` checkpoint (bounding box + centroid hierarchy).

### Implementation

```python
# In _load_checkpoints(), after pattern_library:
tree_path = os.path.join(cpdir, 'hypervolume_tree.pkl')
if os.path.exists(tree_path):
    with open(tree_path, 'rb') as f:
        self._hypervolume_tree = pickle.load(f)
    logger.info(f"  Loaded hypervolume tree")
else:
    self._hypervolume_tree = None
    logger.warning("  No hypervolume_tree.pkl — falling back to L2 centroids")
```

```python
# In _evaluate_candidates(), replace L2 block with:
if self._hypervolume_tree is not None:
    matched = self._navigate_hypervolume_tree(self._hypervolume_tree, p)
    if matched is None:
        _candidate_gate[id(p)] = 'gate1'
        continue
    tmpl, leaf = matched
    tid = tmpl.template_id
    dist = 0.0  # distance no longer meaningful
else:
    # Legacy L2 fallback (existing code)
    ...
```

Port `_navigate_hypervolume_tree()` from orchestrator.py lines 460-497:
- At each depth, test if 16D vector is inside node's `cell_min_16d`/`cell_max_16d`
- If inside: pick node with smallest L2 to centroid
- If outside: fallback to nearest centroid within 2x cell radius
- Recurse to children or return template at leaf

**Reference**: `training/orchestrator.py` lines 450-498, `training/fractal_clustering.py` lines 139-674

---

## 2. Per-Node Scalers

**Current**: Single `self._scaler` (global StandardScaler) for all features.
**Target**: Use leaf node's fitted scaler when available.

### Implementation

In the direction/exit computation, after tree match:
```python
# Use leaf node scaler if available, else global
_node_scaler = getattr(leaf, 'scaler', None) if leaf else None
_scaler = _node_scaler if _node_scaler is not None else self._scaler
_live_scaled = _scaler.transform([_live_feat])[0]
```

**Reference**: `training/fractal_clustering.py` — each `HypervolumeNode` stores a
fitted `StandardScaler` on `node.scaler`.

---

## 3. Feature Importance Weighting

**Current**: Raw scaled features used for direction/MFE prediction.
**Target**: Apply `_imp_w` importance weights before regression.

### Implementation

Add constant at module level:
```python
_IMP_W = np.ones(16)
_IMP_W[8]  = 1.50  # hurst
_IMP_W[15] = 1.45  # osc_coh
_IMP_W[7]  = 1.35  # adx
_IMP_W[9]  = 1.30  # dmi_diff
_IMP_W[0]  = 1.25  # z_score
```

Apply everywhere features are used for prediction:
```python
_live_weighted = _live_scaled * _IMP_W
_pred = np.dot(_live_weighted, np.array(coeff)) + intercept
```

**Reference**: `training/fractal_clustering.py` lines 1189-1195,
`training/orchestrator.py` lines 1649-1668

---

## 4. Shape Classification Gate

**Current**: Not implemented.
**Target**: Match incoming 15m price segment to seed library templates.

### Implementation

```python
# Module-level seed library (same as orchestrator.py lines 113-136)
def _build_seed_library(n=16):
    t = np.linspace(0, 1, n)
    seeds = {}
    seeds['LINEAR_UP']        = t
    seeds['LINEAR_DOWN']      = 1 - t
    seeds['EXPONENTIAL_UP']   = (np.exp(t) - 1) / (np.e - 1)
    seeds['EXPONENTIAL_DOWN'] = 1 - (np.exp(t) - 1) / (np.e - 1)
    seeds['STEP_UP']          = (t > 0.5).astype(float)
    seeds['STEP_DOWN']        = (t <= 0.5).astype(float)
    # ... (copy full set from orchestrator)
    for k in seeds:
        s = seeds[k]
        seeds[k] = (s - s.min()) / (s.max() - s.min() + 1e-12)
    return seeds

_SEED_LIBRARY = _build_seed_library()
_DIRECTIONAL_SHAPES = {'LINEAR_UP', 'LINEAR_DOWN', 'EXPONENTIAL_UP', ...}
```

In the gate cascade, after cluster match:
```python
# Collect last 16 close prices from bar buffer
_segment = self._bar_buffer[-16:]  # normalized to [0,1]
_best_corr, _best_shape = -1, 'UNKNOWN'
for name, template in _SEED_LIBRARY.items():
    corr = np.corrcoef(_segment, template)[0, 1]
    if corr > _best_corr:
        _best_corr = corr
        _best_shape = name
# Gate: reject if best correlation < 0.5 (noise / no recognizable shape)
if _best_corr < 0.5:
    continue  # gate_shape
```

**Note**: This is currently defined but NOT applied in the orchestrator forward
pass either. Include as optional gate (`--enable-shape-gate` flag) for testing.

**Reference**: `training/orchestrator.py` lines 113-136,
`docs/JULES_WAVEFORM_INTEGRATION.md`

---

## 5. Direction Consensus (Multi-TF Voting)

**Current**: `_determine_direction()` uses single-template regression + fallbacks.
**Target**: Add belief network consensus as override when available.

### Implementation

The belief network's `get_direction_consensus()` already exists. Wire it in:

```python
# After _determine_direction() returns (side, p_long, source):
_consensus = self._belief_network.get_direction_consensus(side)
if _consensus is not None and _consensus['confidence'] >= 0.60:
    if _consensus['direction'] != side:
        side = _consensus['direction']
        logger.debug(f"Direction override by consensus: {side} "
                     f"(conf={_consensus['confidence']:.2f})")
elif _consensus is not None and _consensus['confidence'] < 0.60:
    logger.debug(f"Gate 5 reject: consensus conf={_consensus['confidence']:.2f} < 0.60")
    return  # skip trade
```

**Reference**: `training/timeframe_belief_network.py` lines 1036-1123,
`training/orchestrator.py` line 1845 (`CONSENSUS_CONFIDENCE_THRESHOLD = 0.60`)

---

## 6. P(profitable) Gate

**Current**: Not implemented.
**Target**: Reject trades where P(profitable) < 0.70.

### Implementation

```python
# After direction consensus, before exit sizing:
_template_wr = lib_entry.get('win_rate', 0.5)
_p_prof = self._belief_network.compute_p_profitable(side, _template_wr)
if _p_prof < 0.70:
    logger.debug(f"Gate P(prof) reject: {_p_prof:.3f} < 0.70 (tid={best_tid})")
    return
```

**Reference**: `training/timeframe_belief_network.py` lines 798-853,
`training/orchestrator.py` line 1827

---

## 7. Conviction SL Widening

**Current**: Not implemented (SL computed but not widened by conviction).
**Target**: Widen SL for high-conviction trades.

### Implementation

In `_compute_exit_params()` or after it returns, apply:
```python
if belief is not None:
    _conv = belief.conviction
    _sl_mult = 1.0 + 0.5 * max(0, _conv - 0.5)
    sl_ticks = int(sl_ticks * _sl_mult)
```

**Reference**: `training/orchestrator.py` lines 83-86, 1892-1896
Constants: `CONVICTION_SL_MULTIPLIER = 0.5`, `CONVICTION_SL_THRESHOLD = 0.5`

---

## 8. Three-Body Physics Override (Sub-Minute)

**Current**: Not implemented.
**Target**: For depth >= 5 candidates, use quantum state for exits.

### Implementation

After exit sizing, before execute entry:
```python
_cand_depth = getattr(best_candidate, 'depth', 5)
if _cand_depth >= 5:
    _s = best_candidate.state
    _tb_sigma = _s.sigma_fractal
    _tb_z = _s.z_score
    _tb_coh = _s.coherence
    _tb_lz = _s.lagrange_zone
    _tb_hurst = _s.hurst_exponent

    _is_pure = (abs(_tb_z) >= 1.0 and _tb_sigma > 0.0
                and _tb_coh >= 0.3 and _tb_lz != 'CHAOS'
                and abs(_tb_hurst - 0.5) >= 0.08)

    if _is_pure:
        _sigma_t = _tb_sigma / 0.25  # tick_size for MNQ
        tp_ticks  = max(4, int(abs(_tb_z) * _sigma_t))
        sl_ticks  = max(4, int(0.5 * _sigma_t))
        trail_ticks = max(4, int(1.5 * _sigma_t))
        logger.debug(f"Three-body override: TP={tp_ticks} SL={sl_ticks} "
                     f"trail={trail_ticks} (z={_tb_z:.2f}, σ={_tb_sigma:.4f})")
```

**Reference**: `training/orchestrator.py` lines 1937-1967

---

## 9. Remove Gate 0.5 (Depth Filter)

**Current** (live_engine.py ~310-315): Explicit depth filter gate.
**Target**: Remove it — tree structure handles depth filtering implicitly.

### Implementation

Delete or comment out any `if p.depth > X: continue` block in the candidate loop.
Add comment: `# Gate 0.5 REMOVED — depth filtering handled by tree structure`

**Reference**: `training/orchestrator.py` line 1552

---

## 10. Update Scoring (Remove Distance)

**Current**: `score = p_depth + dist + tier_adj + depth_adj` (includes L2 distance).
**Target**: `score = p_depth + tier_adj + depth_adj` (distance removed).

### Implementation

```python
# When using tree navigation:
dist = 0.0  # distance no longer meaningful with tree
score = p_depth + tier_adj + depth_adj
```

**Reference**: `training/orchestrator.py` lines 1557-1569

---

## Gate Cascade Order (Final)

```
Gate 0   — Headroom (no open position, not in cooldown)
Gate 1   — Tree match (HypervolumeTree navigate → template + leaf node)
Gate 2   — Brain (should_fire on template ID)
           Score competition (depth + tier + depth adj, no distance)
Gate 3   — Belief conviction (is_confident)
Gate 4   — Direction confidence (|p_long - 0.5| >= 0.05)  ← Phase A done
Gate 3.5 — Screening fission + hour filter                ← Phase A done
Gate 5   — Direction consensus (confidence >= 0.60)
Gate 6   — P(profitable) >= 0.70
           Shape gate (optional, --enable-shape-gate)
           Exit sizing → conviction SL widen → three-body override
           Execute entry
```

---

## Files to Modify

| File | Changes | ~Lines |
|------|---------|--------|
| `live/live_engine.py` | Items 1-10 above | ~250 |
| `live/config.py` | Add `enable_shape_gate: bool = False` | ~2 |
| `live/launcher.py` | Add `--enable-shape-gate` CLI flag | ~5 |

---

## Testing

```bash
# 1. Compile check
python -m py_compile live/live_engine.py

# 2. Checkpoint loading (no NT8)
python -c "
from live.config import LiveConfig
from live.live_engine import LiveEngine
cfg = LiveConfig(checkpoint_dir='checkpoints')
engine = LiveEngine(cfg, dry_run=True)
engine._load_checkpoints()
print(f'Tree loaded: {engine._hypervolume_tree is not None}')
print(f'Templates: {len(engine._valid_tids)}')
print('OK')
"

# 3. With NT8 Sim101:
python -m live.launcher --account Sim101 --dry-run --log-level DEBUG
```

---

## Dependencies

- Phase A fixes must be in place (already done)
- `hypervolume_tree.pkl` checkpoint must exist (saved during Phase 2.5)
- Belief network workers must be initialized from checkpoint
- Seed library is code-only (no checkpoint needed)
