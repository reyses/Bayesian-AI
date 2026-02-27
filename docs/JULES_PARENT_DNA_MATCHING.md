# Jules Spec: Parent DNA Matching

## Summary

Replace flat nearest-centroid template matching with a two-step **Parent Match → DNA Verification** architecture using the existing TF workers.

**Current**: Every worker independently does nearest-centroid against ALL leaf template centroids. Workers don't coordinate. Matching is ambiguous (always returns something, even for garbage input).

**New**: The 15m worker matches the parent signature first (gate). Then upper workers (30m, 1h) and lower workers (5m, 1m, 15s) verify their respective rows of the template's DNA matrix. Conviction = how well the full multi-TF DNA agrees.

## Architecture

### Step 1: Parent Match (15m Worker = Anchor)

The 15m worker (`tf_seconds=900`) is the PRIMARY matcher. When it fires:

1. Extract 16D features from current 15m quantum state
2. Check which template's **15m-depth DNA row** matches (cell containment on that row's bounds, NOT nearest-centroid)
3. If inside a cell → template candidate identified
4. If outside all cells → no match, no signal (definitive rejection)
5. Broadcast the matched `template_id` to all other workers

**Cell containment** is binary (in/out), replacing the fuzzy nearest-centroid approach. This is the "definitive" matching the user wants.

### Step 2: DNA Verification (All Other Workers)

Once a template candidate is identified, each worker verifies its depth row:

**Upper workers** (30m=1800s, 1h=3600s):
- Check if their current quantum state matches the template's expected DNA at their TF depth
- These are context-only TFs — they confirm the macro structure supports the trade
- Compute `dna_agreement_score` = similarity between live features and template's expected features at that depth

**Lower workers** (5m=300s, 3m=180s, 1m=60s, 30s, 15s):
- Check if their current quantum state matches the template's expected DNA at their TF depth
- These confirm the micro structure is consistent
- Lower TFs update more frequently — they provide real-time micro confirmation

### Step 3: Conviction Aggregation

```
P(win) = f(parent_match_quality, upper_dna_agreement, lower_dna_agreement)
```

- If 15m matches but 1h contradicts → low conviction
- If all depths align → high conviction → high P(win)
- Weighted by TF_WEIGHTS (existing: [4.0, 3.5, 3.0, ...])
- MIN_ACTIVE_LEVELS still required (existing threshold)

## DNA Matrix ↔ TF Mapping

The hypervolume matrix (`build_hypervolume_matrix`) has rows ordered [root, ..., parent, self]. Each row came from a specific TF in the fractal discovery chain. The parent chain entries already store `tf` labels:

```python
chain_entry = {
    'tf': p.timeframe,       # e.g. '1h', '30m', '15m'
    'features_16d': [...],   # 16D feature vector at this depth
    'depth': p.depth,
    ...
}
```

**Per-template TF-depth map** (precomputed during Phase 2.5):
```python
# Example: template whose chain is 1h → 15m → 5m
template.tf_depth_map = {
    '1h':  0,   # depth 0 = 1h features
    '15m': 1,   # depth 1 = 15m features (parent match row)
    '5m':  2,   # depth 2 = 5m features
}
template.dna_centroids = {
    '1h':  np.array([...]),   # expected 16D at 1h depth
    '15m': np.array([...]),   # expected 16D at 15m depth
    '5m':  np.array([...]),   # expected 16D at 5m depth
}
template.dna_bounds = {
    '15m': (cell_min_16d, cell_max_16d),   # for containment check
    ...
}
```

Not every template has every TF. A template from a shallow cascade (e.g., 15m → 5m only) won't have 1h depth data. Workers whose TF isn't in the template's map simply abstain (neutral, not negative).

## Files Changed

### `training/fractal_clustering.py`

1. **PatternTemplate** dataclass — add fields:
   - `tf_depth_map: Dict[str, int]` — maps TF label → depth index
   - `dna_centroids: Dict[str, np.ndarray]` — expected 16D per TF depth
   - `dna_bounds: Dict[str, Tuple[np.ndarray, np.ndarray]]` — (min, max) per TF depth for containment

2. **`_split_at_depth()`** — when creating leaf templates (line ~637):
   - Extract TF labels from member patterns' parent chains
   - Compute per-TF-depth centroids and bounds from member features
   - Store on PatternTemplate

3. **`build_hypervolume_matrix()`** — also return TF labels per row:
   - Already has `chain_entry['tf']` — just need to propagate this

### `training/orchestrator.py`

1. **`run_forward_pass()`** — replace flat centroid init:
   - Instead of: `centroids_scaled = np.array([lib[tid]['centroid'] for tid in tids])`
   - Pass: per-template DNA maps to the belief network
   - Remove: `centroids_scaled` construction entirely

2. **`register_template_logic()`** — store new fields:
   - `tf_depth_map`, `dna_centroids`, `dna_bounds` in pattern_library entry

### `training/timeframe_belief_network.py`

1. **`__init__()`** — accept DNA maps instead of flat centroids:
   - Remove: `centroids_scaled` parameter
   - Add: `dna_templates: Dict[int, DnaTemplate]` (precomputed per template)
   - Build per-TF index: for each TF, which templates have DNA data at that TF?

2. **15m Worker (`tf_seconds=900`)** — becomes the anchor:
   - New matching logic: cell containment on 15m-depth bounds
   - On match: set `self._active_template_id` on the network
   - On no match: clear active template (no signal possible)

3. **Other Workers** — become DNA verifiers:
   - If `self._active_template_id` is set:
     - Look up template's expected DNA at this worker's TF
     - Compute similarity between live features and expected DNA
     - Store `dna_agreement_score` in WorkerBelief
   - If no active template: skip analysis (wait for 15m to fire)

4. **`get_belief()`** — aggregate DNA agreement:
   - Path conviction now includes DNA agreement weighting
   - Workers without DNA data for the active template → neutral (weight=0)
   - Core formula unchanged: weighted geometric mean of P(direction)

5. **Remove**: `LEAF_TOP_K` nearest-centroid logic (no longer needed)
6. **Remove**: `centroids_scaled` attribute and all distance computations

### DNA Agreement Score

For each worker verifying its DNA row:

```python
def _dna_agreement(self, live_features_16d, expected_16d, bounds_min, bounds_max):
    """
    Binary containment + distance-based confidence.
    Returns score in [0, 1].
    """
    # Hard gate: is the live feature inside the cell?
    inside = np.all(live_features_16d >= bounds_min) and np.all(live_features_16d <= bounds_max)
    if not inside:
        return 0.0  # Hard rejection

    # Soft score: how close to centroid? (1.0 = at centroid, 0.5 = at boundary)
    cell_range = bounds_max - bounds_min
    cell_range[cell_range < 1e-9] = 1.0  # avoid div/0
    normalized_pos = (live_features_16d - bounds_min) / cell_range  # [0,1] per axis
    center_dist = np.linalg.norm(normalized_pos - 0.5)  # distance from center
    max_dist = np.sqrt(16) * 0.5  # max possible distance in 16D unit cube
    score = 1.0 - (center_dist / max_dist)
    return float(np.clip(score, 0.0, 1.0))
```

## Worker Timing

No change to update cadence:
- 1h worker: fires every ~240 bars (checks 1h DNA row)
- 30m worker: fires every ~120 bars
- **15m worker: fires every ~60 bars (ANCHOR — triggers template search)**
- 5m worker: fires every ~20 bars
- 1m worker: fires every ~4 bars
- 15s worker: fires every bar (fastest micro confirmation)

The 15m worker sets the template. Lower workers verify on their faster cadences. Upper workers confirm on their slower cadences (they may lag — that's OK, they represent stable context).

## Edge Cases

1. **No 15m match**: No template candidate → no signal. Workers idle. This is correct — if the 15m pattern isn't recognized, don't trade.

2. **Template has no upper TF data**: Some templates only span 15m → 5m (shallow cascade). Upper workers abstain (neutral). Conviction computed from available depths only. Require `MIN_ACTIVE_LEVELS >= 3` still.

3. **15m matches but lower TFs disagree**: Low conviction. The micro structure contradicts the macro setup. This is a PASS (don't trade).

4. **Multiple 15m cell overlaps**: Rare but possible. Take the template whose 15m centroid is closest (tiebreaker). Or report both and let DNA verification disambiguate.

5. **Template switch mid-trade**: If 15m worker fires again and matches a DIFFERENT template, the active template updates. Existing trade management continues with wave_rider — this only affects NEW signal generation.

## Verification

1. `python -m py_compile training/fractal_clustering.py`
2. `python -m py_compile training/timeframe_belief_network.py`
3. `python -m py_compile training/orchestrator.py`
4. `python -m pytest tests/test_clustering_integration.py -v`
5. `python training/orchestrator.py --forward-pass --data DATA/ATLAS_1DAY` (quick smoke test)

## Migration

- Old: `centroids_scaled` (flat N×16 array) + nearest-distance
- New: per-template DNA maps + cell containment + worker coordination
- Backward compatible: if `dna_centroids` missing from a template, fall back to old centroid matching (allows incremental rollout)

## Expected Impact

- **More definitive matching**: Cell containment is binary — either you match or you don't. No more "every bar matches something"
- **Higher signal quality**: Multi-TF DNA verification filters out false positives where the 15m looks right but the micro structure is wrong
- **Lower trade count, higher win rate**: Stricter matching → fewer signals → each signal has higher conviction
- **Eliminates centroid distance as a signal source**: Replaces fuzzy distance with structured DNA agreement
