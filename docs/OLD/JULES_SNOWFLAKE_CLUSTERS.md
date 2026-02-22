# JULES TASK: Snowflake Cluster Schema — LONG and SHORT Mode Split

## The Problem

`FractalClusteringEngine.extract_features()` collapses direction at the first line:
```python
return [abs(z), v_feat, m_feat, ...]   # abs(z) strips LONG vs SHORT
```
A LONG setup at z=-1.8 and a SHORT setup at z=+1.8 produce **identical feature vectors**.
They land in the same cluster, share the same oracle_bias (e.g. 71% LONG / 29% SHORT),
and force the direction gate hack (oracle_marker → logistic → DMI → velocity fallback).

The snowflake schema fixes this at the root: split patterns by direction BEFORE clustering.
Two clean branches, each direction-pure:

```
                   ┌──> LONG  cluster space  (z < 0, looking for longs)
Pattern detected ──┤
                   └──> SHORT cluster space  (z > 0, looking for shorts)
```

Inside each branch, `abs(z)` still makes perfect sense:
- LONG branch:  abs(z) = depth below mean = "how deep into support"
- SHORT branch: abs(z) = height above mean = "how high into resistance"

Benefits:
1. Oracle bias per template becomes 100% — no mixed-direction templates
2. Direction gate eliminated — the branch IS the direction
3. Per-template regression models are directionally pure (OLS MFE, logistic dir)
4. Golden path oracle can now compute LONG ideal + SHORT ideal separately,
   then interleave into the true sequential optimal path

---

## Files to Modify

### 1. `training/fractal_clustering.py`

**No change to `extract_features()`** — the feature vector stays identical.
Direction split happens in `fit()`, not in features.

**Change `fit(self, patterns)`:**

```python
def fit(self, patterns: list):
    """
    Snowflake fit: split patterns into LONG and SHORT branches,
    cluster each independently, return combined library.
    """
    # Split by oracle direction
    long_patterns  = [p for p in patterns if getattr(p, 'oracle_marker', 0) > 0]
    short_patterns = [p for p in patterns if getattr(p, 'oracle_marker', 0) < 0]
    # Note: noise (oracle_marker==0) patterns are excluded from both branches —
    # they don't represent tradeable moves so shouldn't anchor any cluster.

    # Fit separate scaler + cluster tree per branch
    self._long_scaler,  long_templates  = self._fit_branch(long_patterns,  'LONG')
    self._short_scaler, short_templates = self._fit_branch(short_patterns, 'SHORT')

    # Merge into unified library with branch tag
    templates = []
    for t in long_templates:
        t.direction = 'LONG'
        templates.append(t)
    for t in short_templates:
        t.direction = 'SHORT'
        templates.append(t)

    self.templates = templates
    return templates

def _fit_branch(self, patterns: list, direction: str):
    """
    Fit scaler + recursive cluster tree for one directional branch.
    Returns (scaler, list[PatternTemplate])
    """
    from sklearn.preprocessing import StandardScaler
    if not patterns:
        return StandardScaler(), []

    X = np.array([self.extract_features(p) for p in patterns])
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Re-use existing _recursive_split logic
    templates = self._recursive_split(X_scaled, patterns, start_id=0)
    return scaler, templates
```

**Add `direction` field to `PatternTemplate` dataclass:**
```python
@dataclass
class PatternTemplate:
    ...
    direction: str = ''   # 'LONG' or 'SHORT' — set during fit()
```

**Change `find_nearest_template(p, scaler, templates)` to accept branch args:**
The existing signature passes in scaler + templates externally. The orchestrator
now calls it twice (once for each branch) based on the candidate's z_score sign.
No change to the method itself — just how it's called.

---

### 2. `training/orchestrator.py` — Three changes

**A. Checkpoint loading — split into two indices:**

```python
# OLD: one unified library
with open(lib_path, 'rb') as f:
    self.pattern_library = pickle.load(f)
with open(scaler_path, 'rb') as f:
    self.scaler = pickle.load(f)

# NEW: two directional indices + scalers
lib_long_path   = os.path.join(self.checkpoint_dir, 'pattern_library_long.pkl')
lib_short_path  = os.path.join(self.checkpoint_dir, 'pattern_library_short.pkl')
scaler_long_path  = os.path.join(self.checkpoint_dir, 'clustering_scaler_long.pkl')
scaler_short_path = os.path.join(self.checkpoint_dir, 'clustering_scaler_short.pkl')

with open(lib_long_path,    'rb') as f: self.pattern_library_long  = pickle.load(f)
with open(lib_short_path,   'rb') as f: self.pattern_library_short = pickle.load(f)
with open(scaler_long_path, 'rb') as f: self.scaler_long           = pickle.load(f)
with open(scaler_short_path,'rb') as f: self.scaler_short          = pickle.load(f)

# Build centroid indices for both branches
self._build_centroid_index(self.pattern_library_long,  self.scaler_long,  branch='long')
self._build_centroid_index(self.pattern_library_short, self.scaler_short, branch='short')
```

**B. Route candidate to correct branch at Gate 1:**

```python
# OLD: one centroid search
dist, tid = find_nearest_template(best_candidate, self.scaler, centroid_index)

# NEW: route by z_score sign
_cand_z = best_candidate.z_score
if _cand_z <= 0:
    # LONG setup: price below mean, looking for long entry
    dist, tid = find_nearest_template(best_candidate,
                                       self.scaler_long,
                                       self.centroid_index_long)
    _inferred_dir = 'long'
else:
    # SHORT setup: price above mean, looking for short entry
    dist, tid = find_nearest_template(best_candidate,
                                       self.scaler_short,
                                       self.centroid_index_short)
    _inferred_dir = 'short'
```

**C. Direction gate simplification:**

Since every template in `pattern_library_long` is a LONG template (oracle_bias ~100% LONG),
and every template in `pattern_library_short` is a SHORT template (oracle_bias ~100% SHORT),
the direction is now determined by which branch matched.

**Replace the entire direction gate block** (~40 lines, currently: oracle_marker → logistic →
bias → DMI → velocity) with:

```python
# Snowflake direction: branch determines direction
side = _inferred_dir   # 'long' or 'short' — set during centroid lookup above
```

Keep the `oracle_marker` check ONLY for audit/logging purposes (to verify the direction
matched what oracle expected). Log a warning if branch direction ≠ oracle_marker direction
for analysis, but do NOT override the branch direction.

---

### 3. Phase 3 clustering (wherever `fractal_clustering.py` `fit()` is called)

When saving checkpoints after clustering, save two separate files:

```python
long_templates  = [t for t in all_templates if t.direction == 'LONG']
short_templates = [t for t in all_templates if t.direction == 'SHORT']

# Build two sub-libraries
lib_long  = {t.template_id: _build_lib_entry(t) for t in long_templates}
lib_short = {t.template_id: _build_lib_entry(t) for t in short_templates}

with open(os.path.join(ckpt_dir, 'pattern_library_long.pkl'),  'wb') as f: pickle.dump(lib_long, f)
with open(os.path.join(ckpt_dir, 'pattern_library_short.pkl'), 'wb') as f: pickle.dump(lib_short, f)
with open(os.path.join(ckpt_dir, 'clustering_scaler_long.pkl'), 'wb') as f: pickle.dump(engine._long_scaler, f)
with open(os.path.join(ckpt_dir, 'clustering_scaler_short.pkl'),'wb') as f: pickle.dump(engine._short_scaler, f)
```

Also update `template_tiers.pkl` to include all templates from both branches.

---

### 4. Golden Path Oracle — Direction-Aware Ideal

Update `JULES_GOLDEN_PATH_ORACLE.md` logic: when computing sequential ideal, track separate
"free_at" cursors for LONG and SHORT since they don't share a position slot:

```python
# LONG and SHORT trades compete for the same position slot (one at a time).
# The greedy algorithm does NOT need separate cursors — a long and short
# cannot both be open simultaneously. Keep single free_at cursor.

# BUT: report LONG ideal and SHORT ideal separately in the profit gap:
long_candidates = [(ts, exit_ts, val) for ts, exit_ts, val, d in candidates if d=='long']
short_candidates = [(ts, exit_ts, val) for ts, exit_ts, val, d in candidates if d=='short']
# Run greedy on combined, then bucket results by direction for reporting
```

---

## What Does NOT Change

- `extract_features()` — feature vector stays identical (abs(z) is correct within each branch)
- `FractalClusteringEngine._recursive_split()` — reused by both branches
- `_optimize_template_task()` in `orchestrator_worker.py` — unchanged, optimization works per-template
- `TimeframeBeliefNetwork` — workers are direction-agnostic, their conviction/direction is separate
- `WaveRider` — unchanged
- Depth weights, tier system, brain lookup — unchanged

---

## Backward Compatibility

Add a fallback: if the split checkpoint files don't exist, fall back to old unified
`pattern_library.pkl` + `clustering_scaler.pkl` with a warning. This allows old checkpoints
to continue working while new `--fresh` builds produce the split format.

---

## Test Plan

1. Run `--fresh`: verify `pattern_library_long.pkl` and `pattern_library_short.pkl` created
2. Check template counts: `len(lib_long)` + `len(lib_short)` ≈ old `len(pattern_library)`
3. Spot-check template oracle bias: every template in `lib_long` should have `long_bias > 0.90`
4. Run forward pass: verify direction gate no longer shows wrong-direction entries
5. Compare total trades and PnL vs old unified schema run — should be similar or better
6. Check: "direction gate" warning log should be near-zero mismatches (branch dir ≈ oracle dir)
