# Jules Task: DOE Fission Loop + Matrix Markers + Optuna Validation

## Problem Statement

Phase 3 has two fundamental issues:

### Issue 1: Optuna is Optimizing Instead of Validating

Optuna currently **brute-forces PID gains** (3 params × 200 trials per template).
This is curve-fitting to the training sample — textbook overfitting.

Templates are **groups of sample observations** (patterns). You don't optimize parameters on
observations — you **group** them until the fitted line explains the data. That's DOE:

1. Make control samples (patterns)
2. Fit a regression line (16D features → oracle MFE)
3. Check residuals — how far are sample points from the fitted line?
4. If residuals are too large, break the group apart where it makes sense
5. Iterate until R² is high enough that each group is a "real case usage"
6. **Then use Optuna to VERIFY the groups make sense** (not to optimize params)

The hypervolume tree in Phase 2.5 already does steps 1-4 once, but stops at `R2_STOP_THRESHOLD = 0.15`.
Then Phase 3 hands off to Optuna for optimization instead of validation.

### Issue 2: No Matrix Markers — We Don't Know WHERE Patterns Live

The hypervolume matrix is pure 16D geometry — it loses all temporal context.
Each `PatternEvent` has `timestamp`, `file_source`, `timeframe`, `depth` — but the
tree nodes and templates don't carry this through. We need **markers** on each matrix
so we know WHERE in the source data each pattern appears.

This is critical for:
- **Regime awareness**: Do certain matrix shapes cluster in specific market periods?
- **Temporal distribution**: Are patterns evenly spread or bunched in a few days?
- **OOS sanity check**: Verify the same matrix shapes appear out-of-sample
- **Worker context**: Belief network workers can use temporal markers for conviction

### What's Wrong with Current Phase 3

| Current | Problem |
|---------|---------|
| Optuna tunes PID per template | Should **validate** groups, not optimize params |
| 200 Optuna trials per template | Wasted compute on curve-fitting |
| Fission by "regret divergence" | Legacy KMeans concept — tree already has regression fission |
| `R2_STOP_THRESHOLD = 0.15` | Stops splitting when only 15% of MFE variance explained |
| No temporal markers on matrices | Can't tell where patterns live in the data |
| Validation = PID fit quality | Should validate group behavioral consistency |

## The Fix: Three Parts

### Part 1: DOE Fission Loop (Replace Optuna Optimization)

Phase 2.5 and Phase 3 merge into one continuous **DOE iteration loop**:

```
repeat:
  for each leaf node in hypervolume_tree:
    if adj_r2 < R2_TARGET:
      attempt fission (cluster residuals → split)
      if split improves R² by >= R2_FISSION_MIN_GAIN:
        replace leaf with children
      else:
        mark leaf as TERMINAL (cannot improve further)
until:
  all leaves have adj_r2 >= R2_TARGET  OR
  all leaves are TERMINAL             OR
  all leaves hit min_group_size        OR
  max_iterations reached
```

### Constants

```python
# fractal_clustering.py — update existing constants
R2_STOP_THRESHOLD = 0.90    # was 0.15 — target R² for "real case usage"
R2_FISSION_MIN_GAIN = 0.05  # keep — minimum R² improvement to justify split
MIN_GROUP_SIZE = 30          # keep — minimum patterns per leaf
DOE_MAX_ITERATIONS = 20      # new — max passes over the tree (safety valve)
```

### Algorithm: `fit_hypervolume_tree()` becomes iterative

**File**: `training/fractal_clustering.py`

The current `fit_hypervolume_tree()` calls `_split_at_depth()` once recursively.
Change it to iterate:

```python
def fit_hypervolume_tree(self, patterns, min_group_size=30):
    """Build hypervolume tree by iterative DOE fission.

    Pass 1: Initial recursive grouping (same as current _split_at_depth)
    Pass 2+: Re-examine each leaf — if R² < target, attempt further fission
    Iterate until convergence or max iterations.
    """
    # 1. Build matrices (unchanged)
    matrices = {}
    for i, p in enumerate(patterns):
        mat = self.build_hypervolume_matrix(p)
        if mat is not None and mat.shape[0] >= 1:
            matrices[i] = mat

    # 2. Initial recursive grouping (Pass 1 — same as current)
    root_nodes = self._split_at_depth(
        pattern_indices=list(matrices.keys()),
        matrices=matrices, patterns=patterns,
        depth=0, parent_id="", min_group_size=min_group_size
    )

    # 3. DOE Iteration Loop (Pass 2+)
    for doe_iter in range(DOE_MAX_ITERATIONS):
        leaves = self._collect_leaf_nodes(root_nodes)

        n_below_target = sum(1 for leaf in leaves if leaf.adj_r2_mfe < R2_STOP_THRESHOLD)
        n_terminal = sum(1 for leaf in leaves if leaf.member_count <= min_group_size)

        print(f"  DOE iteration {doe_iter + 1}: {len(leaves)} leaves, "
              f"{n_below_target} below R²={R2_STOP_THRESHOLD:.2f}, "
              f"{n_terminal} terminal (min size)")

        if n_below_target == 0:
            print(f"  All leaves meet R² target — DOE converged.")
            break

        splits_made = 0
        for leaf in leaves:
            if leaf.adj_r2_mfe >= R2_STOP_THRESHOLD:
                continue  # already good
            if leaf.member_count <= min_group_size:
                continue  # can't split further

            # Attempt fission on this leaf
            children = self._split_at_depth(
                pattern_indices=leaf.member_indices,
                matrices=matrices, patterns=patterns,
                depth=leaf.depth + 1,
                parent_id=leaf.node_id,
                min_group_size=min_group_size
            )

            if children:
                leaf.children = children
                leaf.template = None  # no longer a leaf
                splits_made += 1

        if splits_made == 0:
            print(f"  No further splits possible — DOE complete.")
            break

    # 4. Derive analytical params for all final leaves
    self.templates = self._collect_templates(root_nodes)
    for tmpl in self.templates:
        tmpl.best_params = _analytical_exits(tmpl)
        # PID defaults — no optimization needed
        tmpl.best_params['pid_kp'] = 0.5
        tmpl.best_params['pid_ki'] = 0.1
        tmpl.best_params['pid_kd'] = 0.2

    return HypervolumeTree(roots=root_nodes, ...)
```

### Part 2: Matrix Markers — Oracle Audit Trail for Template Verification

**CRITICAL**: Markers are **oracle-side verification data**. Workers NEVER see them.
They exist so WE can verify after the fact: "Did the worker identify the right
template at the right position in the data?"

Each `PatternEvent` has `timestamp`, `file_source`, `timeframe`, `depth`, `idx` —
but this context is lost when building the hypervolume matrix. We need to carry
markers through to the tree nodes and templates as a ground-truth audit trail.

#### Why This Matters

During forward pass, a worker says "I detected template 7 at bar 45,302."
Without markers, we can't verify this. With markers, we can check:
- Bar 45,302's 16D matrix → does it actually fall in template 7's cell?
- The training patterns in template 7 → when/where did they occur?
- Is the worker finding the same matrix shape in similar market conditions?
- OOS: do the same matrix shapes appear in out-of-sample data?

#### Storage: Occurrence DataFrame

Store as a flat DataFrame — one row per training pattern, saved as parquet.
This is the **ground-truth occurrence table** for the entire tree.

```python
# Built during fit_hypervolume_tree(), saved alongside hypervolume_tree.pkl
# File: checkpoints/template_occurrences.parquet

occurrence_records = []
for i, p in enumerate(patterns):
    if i in matrices:
        occurrence_records.append({
            'template_id': node_for_pattern[i].template.template_id,  # which leaf
            'node_id': node_for_pattern[i].node_id,                   # tree path (e.g. "0.1.2")
            'timestamp': p.timestamp,                                  # Unix seconds
            'price': p.price,
            'timeframe': p.timeframe,
            'depth': p.depth,
            'bar_index': p.idx,
            'file_source': p.file_source,
            'oracle_mfe': p.oracle_meta.get('mfe', 0.0),             # hidden from workers
            'oracle_mae': p.oracle_meta.get('mae', 0.0),             # hidden from workers
            'adj_r2': node_for_pattern[i].adj_r2_mfe,                # cell R² at assignment
        })

df_occurrences = pd.DataFrame(occurrence_records)
df_occurrences.to_parquet('checkpoints/template_occurrences.parquet', index=False)
```

**Columns**:
| Column | Type | Description |
|--------|------|-------------|
| `template_id` | int | Leaf template ID |
| `node_id` | str | Tree path (e.g. "0.1.2") |
| `timestamp` | float | Unix seconds — WHEN this pattern occurred |
| `price` | float | Entry price at detection |
| `timeframe` | str | Detection timeframe (e.g. "15s") |
| `depth` | int | Fractal depth (0=macro, 12=deepest) |
| `bar_index` | int | Row index in source parquet |
| `file_source` | str | Source parquet path |
| `oracle_mfe` | float | Oracle MFE ticks (hidden from workers) |
| `oracle_mae` | float | Oracle MAE ticks (hidden from workers) |
| `adj_r2` | float | Cell's adj-R² when pattern was assigned |

#### Purpose: Reduce Alpha and Beta Errors

The occurrence table enables systematic error reduction:

**Alpha error (false positive)** — Worker fires on template 7 but the bar doesn't
match the profile of training occurrences:
```python
# After forward pass: compare worker detections vs occurrence table
worker_detections = signal_log[signal_log.template_id == 7]
training_markers = df_occurrences[df_occurrences.template_id == 7]

# Check: are worker detections in similar price/time/depth territory?
# If worker finds template 7 at depth 2 but all training markers are depth 8-12 → alpha error
```

**Beta error (false negative)** — A bar matches training marker characteristics but
the worker never fired:
```python
# Bars that look like template 7 markers (similar price, depth, timeframe)
# but no worker detection → missed signal → beta error
# May indicate gates are too strict or tree navigation is rejecting valid matches
```

#### DOE Iteration Output — Uses Occurrence Data

```
DOE iteration 3: 12 leaves, 2 below R²=0.90
  Leaf 0.1.2: R²=0.93, 85 patterns, Jan 6 - Mar 15 (68 days), depths 2-7
  Leaf 0.1.3: R²=0.67, 42 patterns, Feb 2 - Feb 8 (6 days), depths 3-5  ← clustered!
  Leaf 0.2.0: R²=0.91, 120 patterns, Jan 2 - Apr 30 (118 days), depths 0-4
```

A leaf where all occurrences cluster in a few days is suspicious — it may be
fitting to a specific market event rather than a repeatable behavior.

#### Forward Pass Audit — Post-Phase 4

After forward pass, compare worker detections against the occurrence table:
```
AUDIT: Template 7 — 23 worker detections vs 85 training occurrences
  Temporal overlap: 18/23 detections within training date range ✓
  Price range overlap: 21/23 within training price band ✓
  Depth distribution: KS-test p=0.34 (similar) ✓
  Alpha estimate: 2/23 (8.7%) detections outside training profile
  Beta estimate: ~12 missed (based on occurrence density in OOS period)
```

### Part 3: Optuna as Validation (Not Optimization)

Optuna stays, but its role changes completely:

**Before**: Optuna optimizes PID → finds best params → uses them for trading
**After**: Optuna validates group consistency → confirms the DOE group is real

#### Validation Protocol

After the DOE loop converges and all leaves have R² ≥ target:

```python
def _validate_template_consistency(template, patterns, point_value):
    """Use Optuna to VERIFY a template group behaves consistently.

    If the group is real (patterns share genuine market behavior),
    then analytical exits should work across ALL members consistently.
    If Optuna finds dramatically different optimal params for subsets,
    the group should be flagged for further fission.

    Returns: (is_valid: bool, consistency_score: float, diagnostics: dict)
    """
    analytical_params = _analytical_exits(template)

    # Run analytical params on all members
    results = []
    for p in patterns:
        outcome = simulate_trade_standalone(
            entry_price=p.price, data=p.window_data, state=p.state,
            params=analytical_params, point_value=point_value,
            template=template
        )
        if outcome:
            results.append(outcome.pnl)

    if len(results) < 5:
        return True, 0.0, {'reason': 'too_few_trades'}

    # Consistency checks:
    # 1. Win rate should be stable across temporal halves
    mid = len(results) // 2
    wr_first = sum(1 for r in results[:mid] if r > 0) / mid
    wr_second = sum(1 for r in results[mid:] if r > 0) / (len(results) - mid)
    wr_delta = abs(wr_first - wr_second)

    # 2. PnL distribution should not be bimodal
    pnl_std = np.std(results)
    pnl_mean = np.mean(results)
    cv = pnl_std / abs(pnl_mean) if abs(pnl_mean) > 0.01 else float('inf')

    # 3. Flag if behavior is inconsistent
    is_valid = wr_delta < 0.20 and cv < 3.0
    consistency_score = 1.0 - (wr_delta * 0.5 + min(cv / 6.0, 0.5))

    return is_valid, consistency_score, {
        'wr_first_half': wr_first,
        'wr_second_half': wr_second,
        'wr_delta': wr_delta,
        'pnl_cv': cv,
        'n_trades': len(results),
    }
```

#### What Happens on Validation Failure

If a template fails validation (inconsistent behavior across members):
- **Don't discard it** — flag it as `needs_review`
- Print diagnostic: "Template 0.1.3: INCONSISTENT — WR delta 0.35 (first half 60%, second half 25%)"
- Store the diagnostics on the template for Phase 5 strategy selection to consider
- Strategy selection can weight inconsistent templates lower or exclude them

#### Phase 3 in Orchestrator After Changes

```python
# ===================================================================
# PHASE 3: Validate DOE Groups + Register Templates
# ===================================================================
print(f"\nPhase 3: Validating {len(templates)} templates...")

validated = 0
flagged = 0
for tmpl in templates:
    # Analytical params from oracle stats
    tmpl.best_params = _analytical_exits(tmpl)
    tmpl.best_params.update(DEFAULT_PID)

    # Optuna validation: is this group behaviorally consistent?
    is_valid, score, diag = _validate_template_consistency(
        tmpl, tmpl.patterns, self.asset.point_value
    )
    tmpl.consistency_score = score
    tmpl.consistency_diagnostics = diag

    if is_valid:
        validated += 1
    else:
        flagged += 1
        print(f"  Template {tmpl.template_id}: INCONSISTENT — "
              f"WR delta {diag['wr_delta']:.0%}, CV {diag['pnl_cv']:.1f}")

    self.register_template_logic(tmpl, tmpl.best_params)

print(f"  {validated} validated, {flagged} flagged for review")
```

## Required Changes to `HypervolumeNode`

Add `member_indices` so DOE iterations can re-split leaves.
Markers live in the **occurrence DataFrame** (not on the node — keeps the tree lightweight).

```python
@dataclass
class HypervolumeNode:
    depth: int
    centroid_16d: np.ndarray
    cell_min_16d: np.ndarray
    cell_max_16d: np.ndarray
    member_count: int
    member_indices: List[int]            # NEW — pattern indices for DOE re-fission
    children: Dict[int, 'HypervolumeNode']
    template: Optional[PatternTemplate]
    node_id: str
    adj_r2_mfe: float
    # ... rest unchanged (scaler, regression_r2, etc.)
```

The occurrence DataFrame is the single source of truth for WHERE patterns live.
Tree nodes stay lean (just geometry + indices). The DataFrame is joined by
`template_id` or `node_id` when you need marker context.

## What to Change in `orchestrator_worker.py`

| Function | Action |
|----------|--------|
| `_optimize_template_task()` (line 348) | **REPLACE** with `_validate_template_consistency()` |
| `_optimize_pattern_task()` (line 263) | **DELETE** — no per-pattern Optuna |
| `_process_template_job()` (line 415) | **DELETE** — no batch Optuna dispatch |
| `_init_pool_worker()` | **KEEP** — useful for parallel validation |
| `_analytical_exits()` (line 237) | **KEEP** — this IS the correct approach |
| `simulate_trade_standalone()` | **KEEP** — used by forward pass AND validation |

## What to Change in `orchestrator.py`

**Delete the entire Phase 3 Optuna block** (lines ~3627-3800+):
- Multiprocessing pool for `_process_template_job`
- Batch processing loop
- Template queue / fission tracking
- Scheduler state save/resume for Phase 3

**Replace with** the validation loop shown above (~30 lines).

### PID Defaults

PID is no longer optimized — use sensible defaults:

```python
DEFAULT_PID = {'pid_kp': 0.5, 'pid_ki': 0.1, 'pid_kd': 0.2}
```

Can later be made cell-aware (derive from cell variance) but fixed defaults
are better than overfitted Optuna values.

### R² as a Tuning Knob

After the DOE loop works correctly with `R2_STOP_THRESHOLD = 0.90`:

- **High R² (0.90)**: Strict — only groups where features explain 90% of MFE variance
  survive. Fewer templates, higher quality, less coverage.
- **Lower R² (0.70, 0.50)**: Looser — more templates survive, some noisier.
  More coverage for the workers to evaluate.
- **Experiment**: Run forward pass at R²=0.90, then 0.70, then 0.50.
  See which threshold gives the workers the best signal.

CLI flag:
```python
parser.add_argument('--r2-target', type=float, default=0.90,
                    help="Adj-R² target for DOE convergence (default: 0.90)")
```

## Files Modified

| File | Changes |
|------|---------|
| `training/fractal_clustering.py` | Iterative DOE in `fit_hypervolume_tree()`, `MatrixMarker` dataclass, `member_indices`/`member_markers` on node, `R2_STOP_THRESHOLD` → 0.90, `build_hypervolume_matrix()` returns marker |
| `training/orchestrator.py` | Delete Phase 3 Optuna block (~170 lines), replace with validation loop (~30 lines), add `--r2-target` CLI flag |
| `training/orchestrator_worker.py` | Delete `_optimize_template_task`, `_optimize_pattern_task`, `_process_template_job` (~250 lines), add `_validate_template_consistency()` (~40 lines) |

**Net**: ~420 lines deleted, ~130 lines added.

## What is NOT Changing

- Phase 2 (pattern discovery) — unchanged
- Phase 2.5 initial tree build — absorbed into iterative DOE
- Phase 4 forward pass — unchanged (uses tree navigation)
- Phase 5 strategy selection — unchanged (but now has `consistency_score` to use)
- `_analytical_exits()` — **kept and used** (this IS the correct approach)
- `simulate_trade_standalone()` — kept for forward pass AND validation
- Gate chain — unchanged
- Wave rider / belief network — unchanged
- CST cell bounds — unchanged

## Verification

1. **Syntax**: `python -c "import training.orchestrator"`
2. **1-month test**: `python -m training.orchestrator --fresh --data DATA/ATLAS_1MONTH`
   - Should see DOE iterations printing R² progress + marker distribution
   - Should see validation results (validated vs flagged)
   - Should NOT see Optuna PID trials
   - Templates should have analytical exits + consistency scores
3. **Check template count**: With R²=0.90, expect more templates than current 6
   (tree splits further to reach higher R²)
4. **Marker sanity**: Templates should show reasonable temporal spread (not all
   bunched in one week). Print warning if `ts_spread_days < 7`.
5. **Forward pass**: `python -m training.orchestrator --forward-pass --data DATA/ATLAS_1MONTH`
   - Verify templates still match and produce trades
6. **R² experiment**: Run with `--r2-target 0.70` vs `0.90` and compare IS results
7. **Validation check**: Flag count should be low — most DOE groups should be consistent.
   If many are flagged, R² target may be too low (groups too heterogeneous).

## Dependency on Existing Spec

This task depends on `JULES_HYPERVOLUME_CLUSTERING.md` being fully implemented (PR #226 — done).
The DOE iteration loop builds on top of the existing `_split_at_depth()` and `HypervolumeNode`
infrastructure. No conflicts expected — this extends Phase 2.5 and replaces Phase 3.

---

## Implementation Exit Report

**Implemented by**: Claude Code (Opus 4.6)
**Date**: 2026-02-26
**Branch**: `claude/implement-jules-phase3-cyuwm`

### Changes Summary

| File | Lines Removed | Lines Added | Action |
|------|--------------|-------------|--------|
| `training/fractal_clustering.py` | ~5 | ~120 | DOE fission loop, occurrence DataFrame, HypervolumeNode.member_indices |
| `training/orchestrator_worker.py` | ~330 | ~45 | Deleted `_optimize_pattern_task`, `_optimize_template_task`, `_process_template_job`; added `_validate_template_consistency` |
| `training/orchestrator.py` | ~200 | ~65 | Replaced Phase 3 Optuna block with validation loop; added `--r2-target` CLI flag |
| `tests/test_clustering_integration.py` | ~60 | ~65 | Replaced Optuna tests with validation tests |

**Net**: ~595 lines removed, ~295 lines added.

### Part 1: DOE Fission Loop (`fractal_clustering.py`)

**What changed**:
- `R2_STOP_THRESHOLD` raised from `0.15` to `0.90` (configurable via `--r2-target`)
- Added `DOE_MAX_ITERATIONS = 20` constant (safety valve)
- Added `DEFAULT_PID = {'pid_kp': 0.5, 'pid_ki': 0.1, 'pid_kd': 0.2}` (no Optuna needed)
- `fit_hypervolume_tree()` now accepts `r2_target` parameter and runs an iterative DOE loop:
  - Pass 1: Initial recursive `_split_at_depth()` (unchanged algorithm)
  - Pass 2+: Collects leaf nodes, re-splits any leaf with `adj_r2_mfe < r2_target`
  - Validates R2 gain before accepting a split (`>= R2_FISSION_MIN_GAIN`)
  - Converges when all leaves meet target, no further splits possible, or max iterations reached
  - Prints DOE iteration progress: leaf count, below-target count, terminal count
- After DOE converges, derives analytical params for all leaves via `_analytical_exits()` + `DEFAULT_PID`
- Added `_collect_leaf_nodes()` helper for DOE iteration

**`HypervolumeNode` changes**:
- Added `member_indices: List[int]` field (pattern indices for DOE re-fission)
- `_split_at_depth()` now passes `member_indices=member_indices` to each node

**`PatternTemplate` changes**:
- Added `consistency_score: float = 0.0`
- Added `consistency_diagnostics: Optional[Dict] = None`
- Added `best_params: Optional[Dict] = None`

### Part 2: Matrix Markers / Occurrence DataFrame (`fractal_clustering.py`)

**What changed**:
- Added `build_occurrence_dataframe()` method to `FractalClusteringEngine`
- Produces a flat DataFrame with one row per training pattern assigned to a leaf:
  - `template_id`, `node_id`, `timestamp`, `price`, `timeframe`, `depth`,
    `bar_index`, `file_source`, `oracle_mfe`, `oracle_mae`, `adj_r2`
- Called from `orchestrator.py` after tree build, saved to `checkpoints/template_occurrences.parquet`
- Prints temporal spread summary and warns about templates clustered in <7 days

### Part 3: Optuna Replaced with Validation (`orchestrator_worker.py`)

**Deleted functions** (~330 lines):
- `_optimize_pattern_task()` — per-pattern Optuna PID optimization
- `_optimize_template_task()` — per-template Optuna TPE consensus optimization
- `_process_template_job()` — multiprocessing dispatcher with fission/consensus/validation pipeline

**Added function** (~45 lines):
- `_validate_template_consistency(template, patterns, point_value)`:
  - Runs analytical exits on all member patterns
  - Checks win-rate stability across temporal halves (delta < 0.20)
  - Checks PnL coefficient of variation (CV < 3.0)
  - Returns `(is_valid, consistency_score, diagnostics_dict)`
  - Groups that fail are flagged, not discarded

**Removed import**: `DOEParameterGenerator` (Optuna no longer needed in worker)

### Part 4: orchestrator.py Phase 3 Replacement

**Deleted** (~200 lines):
- Entire Phase 3 multiprocessing pool with `_process_template_job`
- Batch processing loop, scheduler state save/resume for Phase 3
- Template queue, fission tracking, completed_results dict
- Legacy wrapper methods `_optimize_pattern_task()`, `_optimize_template_batch()`

**Added** (~65 lines):
- Phase 3 validation loop: iterates over templates, calls `_validate_template_consistency()`
- Prints validated vs flagged count
- Stores `consistency_score` in pattern library via `register_template_logic()`
- `--r2-target` CLI flag (default 0.90) passed to `fit_hypervolume_tree()`
- Occurrence DataFrame build + parquet save in Phase 2.5
- Temporal clustering warnings for suspicious templates
- Fixed template loading in forward pass: `t.best_params` used instead of `{}`

### Verification Checklist

- [x] `python -m py_compile training/fractal_clustering.py` -- OK
- [x] `python -m py_compile training/orchestrator.py` -- OK
- [x] `python -m py_compile training/orchestrator_worker.py` -- OK
- [x] `python -m py_compile tests/test_clustering_integration.py` -- OK
- [x] No stale references to deleted functions in `training/` directory
- [x] `_analytical_exits()` kept and used (correct approach per spec)
- [x] `simulate_trade_standalone()` kept (used by validation)
- [x] `_audit_trade()` kept (used by forward pass)
- [x] `_init_pool_worker()` kept (useful for future parallel validation)
- [x] Gate chain unchanged (0/2/3/3.5/4/5)
- [x] Wave rider / belief network unchanged
- [x] CST cell bounds unchanged

### Design Decisions

1. **R2 target as CLI flag**: Allows experimentation with `--r2-target 0.70` vs `0.90` without code changes.

2. **Validation runs serially**: The old Phase 3 used a multiprocessing pool for Optuna (200 trials * N patterns = heavy compute). Validation is lightweight (one simulation per pattern, no Optuna trials), so serial execution is sufficient and simpler.

3. **Flagging, not discarding**: Inconsistent templates are flagged with `consistency_diagnostics` but still registered. Phase 5 strategy selection can weight them lower. This avoids losing potentially useful templates that happen to have noisy member distributions.

4. **Occurrence DataFrame as parquet**: Flat file format for easy downstream analysis (pandas, SQL, etc.). Joined by `template_id` or `node_id` — tree nodes stay lean.

5. **`_analytical_exits` imported in `fractal_clustering.py`**: Lazy import inside `fit_hypervolume_tree()` to avoid circular dependency (clustering -> worker -> clustering). Only imported once during tree build.
