# Jules Task: 1-Minute Anchor Refactor

**Priority**: URGENT — current system leaves $32K on the table in OOS
**Branch**: `feature/1m-anchor`
**Validation**: `python training/trainer.py --fresh --forward-pass`

---

## Problem

Templates mix patterns from 13 TFs (1s–15m). Oracle MFE is measured over each TF's
native horizon (5 min for 1s, 4 hours for 15m). K-Means clusters these into templates
regardless of TF origin, so `p75_mfe_ticks` averages 5-minute and 4-hour price swings.

Result: anchor patience expects 307+ ticks MFE, actual trades achieve 48 ticks.
Giveback protection is permanently disabled. 76% of trades exit at breakeven.

**Full investigation**: `docs/Active/PATTERN_SCALE_MISMATCH_REPORT.md`
**Architecture spec**: `docs/Active/SPEC_PATTERN_ANCHORING.md` (Sections 1-8)

---

## Three-Layer Architecture

All TFs still get computed (MarketState per bar), but their ROLES change:

| Layer | TFs | Role | Label |
|-------|-----|------|-------|
| **Above** (context) | 1D, 4h, 1h, 30m, 15m, 5m, 3m | "How we got here" — macro regime, trend direction | Context features in 16D vector |
| **Anchor** (discovery) | **1m** | "Where we are going" — pattern detection + oracle MFE | Only TF that produces PatternEvent objects |
| **Below** (context) | 30s, 15s, 5s, 1s | "How we are doing" — micro execution quality | Context features in 16D vector |

**Key principle**: Only 1m bars produce patterns. All other TFs enrich those patterns
with context features. The TBN still runs workers on all 11 TFs for conviction — this
change only affects discovery and clustering.

---

## File Changes (6 files)

### 1. `config/oracle_config.py` — SIMPLE EDIT

Replace the multi-TF lookahead dict with a single entry:

```python
# BEFORE:
ORACLE_LOOKAHEAD_BARS = {
    '1s':  300,   # 5 minutes
    '5s':  120,   # 10 minutes
    '15s':  60,   # 15 minutes
    '1m':   60,   # 1 hour
    '5m':   24,   # 2 hours
    '15m':  16,   # 4 hours
    '1h':    8,   # 8 hours
    '4h':    6,   # 24 hours
    '1D':    5,   # 5 days
    '1W':    4,   # 4 weeks
}

# AFTER:
ORACLE_LOOKAHEAD_BARS = {
    '1m':   16,   # 16 minutes — matches trade oscillation cycle (~8 min peak)
}
```

**Why 16**: Median profitable trade hold = 5 min, p75 = 8.5 min, oscillation cycle = 8 min.
16 minutes captures the full cycle with headroom. This anchors MFE at 20-80 ticks
(matching actual trade behavior) instead of 200-5,500.

Keep all other oracle_config constants unchanged.

---

### 2. `core/feature_extraction.py` — REWRITE

Replace the current 16D vector with a new layout that uses context TF features
instead of hierarchy features (depth, tf_seconds, parent chain).

```python
"""
Canonical 16D Feature Vector — Single Source of Truth
=====================================================
Both FractalClusteringEngine.extract_features() and
TimeframeBeliefNetwork.state_to_features() delegate here.

Three-layer architecture:
  ABOVE (how we got here):  macro context from 15m, 1h, 1D
  ANCHOR (where we go):     1m bar state (z, velocity, momentum, entropy)
  BELOW (how we're doing):  micro context from 15s, 5s

Vector layout (16D):
  [0]  abs(z_score)              — 1m z-score (anchor)
  [1]  log1p(|velocity|)         — 1m velocity (anchor)
  [2]  log1p(|momentum|)         — 1m momentum (anchor)
  [3]  entropy_normalized        — 1m entropy (anchor)
  [4]  context_15m_z             — 15m z-score (above: intermediate trend)
  [5]  context_1h_z              — 1h z-score (above: macro trend)
  [6]  context_1h_dmi_sign       — sign(1h DMI+  - DMI-) (above: macro direction)
  [7]  adx / 100                 — 1m ADX (anchor regime)
  [8]  hurst_exponent            — 1m Hurst (anchor regime)
  [9]  dmi_diff / 100            — 1m DMI diff (anchor direction)
  [10] context_15m_dmi_diff      — 15m DMI diff / 100 (above: intermediate trend strength)
  [11] context_1D_dmi_sign       — sign(1D DMI) (above: daily regime)
  [12] context_1D_z              — 1D z / 3.0 clamped [-1, 1] (above: daily band position)
  [13] tf_alignment_1m_15m       — sign(1m_dmi) * sign(15m_dmi) (near alignment)
  [14] term_pid                  — 1m PID (anchor)
  [15] oscillation_entropy_norm  — 1m oscillation coherence (anchor)
"""

import numpy as np


def extract_feature_vector(
    z_score: float, velocity: float, momentum: float,
    entropy_normalized: float,
    adx: float, hurst: float, dmi_diff: float,
    pid: float, osc_coherence: float,
    # Context from above TFs (default 0.0 when unavailable)
    context_15m_z: float = 0.0,
    context_1h_z: float = 0.0,
    context_1h_dmi_sign: float = 0.0,
    context_15m_dmi_diff: float = 0.0,
    context_1D_dmi_sign: float = 0.0,
    context_1D_z: float = 0.0,
    tf_alignment_1m_15m: float = 0.0,
) -> list:
    """Canonical 16D feature vector. Single source of truth.

    All inputs are raw values — compression (log1p, /100, clamp) applied here.
    Callers must NOT pre-compress.
    """
    return [
        abs(z_score),
        np.log1p(abs(velocity)),
        np.log1p(abs(momentum)),
        entropy_normalized,
        np.clip(context_15m_z / 3.0, -1.0, 1.0),    # normalize to [-1, 1]
        np.clip(context_1h_z / 3.0, -1.0, 1.0),
        float(np.sign(context_1h_dmi_sign)),          # -1, 0, or +1
        adx,                                           # already /100 by caller
        hurst,
        dmi_diff,                                      # already /100 by caller
        np.clip(context_15m_dmi_diff / 100.0, -1.0, 1.0),
        float(np.sign(context_1D_dmi_sign)),
        np.clip(context_1D_z / 3.0, -1.0, 1.0),
        tf_alignment_1m_15m,                           # -1, 0, or +1
        pid,
        osc_coherence,
    ]
```

**Removed parameters**: `tf_seconds`, `depth`, `parent_is_band_reversal`, `parent_z`,
`parent_dmi_diff`, `root_is_roche`, `tf_alignment` (old hierarchy features).

**Added parameters**: `context_15m_z`, `context_1h_z`, `context_1h_dmi_sign`,
`context_15m_dmi_diff`, `context_1D_dmi_sign`, `context_1D_z`, `tf_alignment_1m_15m`.

---

### 3. `training/fractal_discovery_agent.py` — MAJOR EDIT

#### 3a. Constants (lines 35-55)

Replace:
```python
TIMEFRAME_HIERARCHY = ['1D', '4h', '1h', '30m', '15m', '5m', '3m', '2m', '1m', '30s', '15s', '5s', '1s']
CONTEXT_ONLY_TIMEFRAMES = {'1D', '4h', '1h', '30m'}
PRIMARY_SIGNAL_TIMEFRAME = '15m'
```

With:
```python
# Single discovery TF — all patterns come from 1m bars
DISCOVERY_TIMEFRAME = '1m'

# Context TFs: computed for enrichment, no pattern discovery
# Above = "how we got here" (macro regime, trend)
CONTEXT_ABOVE = ['1D', '4h', '1h', '30m', '15m', '5m', '3m']
# Below = "how we are doing" (micro execution quality)
CONTEXT_BELOW = ['30s', '15s', '5s', '1s']
```

#### 3b. PatternEvent dataclass (line 57-78)

Add context fields:
```python
@dataclass
class PatternEvent:
    # ... existing fields ...

    # Context from above TFs (enrichment)
    context_above: Dict[str, Dict] = field(default_factory=dict)
    # Format: {'15m': {'z': float, 'adx': float, 'dmi_diff': float, 'hurst': float},
    #          '1h': {...}, '1D': {...}}

    # Context from below TFs (enrichment)
    context_below: Dict[str, Dict] = field(default_factory=dict)
    # Format: {'15s': {'z': float, ...}, '5s': {...}}
```

#### 3c. New method: `scan_atlas_1m()` (replaces `scan_atlas_topdown()`)

**Flow**:
1. Load ALL context TF data (1D through 3m, and 30s through 1s) — compute MarketState per bar
2. Build context lookup: for each context TF, index MarketState by timestamp range
   so we can look up "which 15m bar was active at this 1m timestamp?"
3. Load 1m data — compute MarketState per bar
4. For each 1m bar that triggers a pattern (cascade_detected or structure_confirmed):
   a. Look up the enclosing bar of each above-TF (15m, 1h, 1D)
   b. Look up the most recent bar of each below-TF (15s, 5s)
   c. Attach context_above and context_below dicts to the PatternEvent
   d. Consult oracle with `'1m'` timeframe (gets 16-bar lookahead)
5. Return list of PatternEvent objects (all at 1m, all depth=0)

**Context lookup logic** for above TFs:
```python
def _find_enclosing_bar(self, ts: float, tf: str, tf_states: dict) -> Optional[MarketState]:
    """Find the context TF bar that was active at timestamp ts.

    For above TFs: find the last completed bar whose timestamp <= ts.
    For below TFs: find the most recent bar whose timestamp <= ts.
    """
    # tf_states is a sorted list of (timestamp, MarketState)
    # Binary search for the last entry with timestamp <= ts
    ...
```

**Important**: Context TFs do NOT need windowed scanning. Load ALL bars for context TFs,
compute MarketState, index by timestamp. This is simpler than the current drill-down.

**What to DELETE from this file**:
- `scan_atlas_topdown()` — replaced by `scan_atlas_1m()`
- `scan_day_cascade()` — replaced by simplified 1m scan
- `_batch_scan_windowed()` — no more windowed drill-down
- `_merge_windows()` — no more window merging
- `_build_parent_chain()` — replaced by flat context dicts
- Remove `parent_type`, `parent_tf`, `parent_chain` from PatternEvent usage
  (keep fields for backward compat but always set to empty)

**What to KEEP**:
- `_consult_oracle()` — unchanged, just only called for 1m
- `_batch_scan_full()` — used for 1m discovery scan
- `_load_files_threaded()` — I/O helper
- `_find_files()` — file finder
- `scan_atlas_parallel()` — backward compat (single-TF flat scan)
- `detect_patterns()` — direct use

**Performance note**: Loading all context TFs upfront is more I/O than windowed scanning,
but GPU compute is faster (one batch per TF instead of many small windowed batches).
Total time should be comparable.

#### 3d. Context enrichment in `_batch_scan_full()` (for 1m scan)

After detecting patterns on 1m bars, enrich each PatternEvent:

```python
# For each detected pattern at 1m:
for p in detected:
    # Look up above-TF context
    for tf in CONTEXT_ABOVE:
        ctx_state = self._find_enclosing_bar(p.timestamp, tf, context_states[tf])
        if ctx_state:
            p.context_above[tf] = {
                'z': ctx_state.z_score,
                'adx': ctx_state.adx_strength,
                'dmi_diff': ctx_state.dmi_plus - ctx_state.dmi_minus,
                'hurst': ctx_state.hurst_exponent,
                'velocity': ctx_state.velocity,
                'momentum': ctx_state.momentum_strength,
            }

    # Look up below-TF context
    for tf in CONTEXT_BELOW:
        ctx_state = self._find_enclosing_bar(p.timestamp, tf, context_states[tf])
        if ctx_state:
            p.context_below[tf] = {
                'z': ctx_state.z_score,
                'adx': ctx_state.adx_strength,
                'dmi_diff': ctx_state.dmi_plus - ctx_state.dmi_minus,
                'hurst': ctx_state.hurst_exponent,
                'velocity': ctx_state.velocity,
                'momentum': ctx_state.momentum_strength,
            }
```

---

### 4. `core/fractal_clustering.py` — MODERATE EDIT

Update `extract_features()` (line 148-198) to pass context data to the new
`extract_feature_vector()` signature:

```python
@staticmethod
def extract_features(p: Any) -> List[float]:
    """Extracts 16D feature vector from a PatternEvent.
    Delegates to core.feature_extraction.extract_feature_vector().
    """
    from core.feature_extraction import extract_feature_vector

    state = getattr(p, 'state', None)

    # Extract context from above TFs
    ctx_above = getattr(p, 'context_above', {})
    ctx_15m = ctx_above.get('15m', {})
    ctx_1h = ctx_above.get('1h', {})
    ctx_1D = ctx_above.get('1D', {})

    # 1m DMI for alignment calc
    self_dmi_diff = 0.0
    if state:
        self_dmi_diff = getattr(state, 'dmi_plus', 0.0) - getattr(state, 'dmi_minus', 0.0)

    # Alignment: sign(1m_dmi) * sign(15m_dmi)
    ctx_15m_dmi = ctx_15m.get('dmi_diff', 0.0)
    self_dir = 1.0 if self_dmi_diff > 0 else (-1.0 if self_dmi_diff < 0 else 0.0)
    ctx_dir = 1.0 if ctx_15m_dmi > 0 else (-1.0 if ctx_15m_dmi < 0 else 0.0)
    alignment = self_dir * ctx_dir

    return extract_feature_vector(
        z_score=getattr(p, 'z_score', 0.0),
        velocity=getattr(p, 'velocity', 0.0),
        momentum=getattr(p, 'momentum', 0.0),
        entropy_normalized=getattr(p, 'entropy_normalized', 0.0),
        adx=getattr(state, 'adx_strength', 0.0) / 100.0 if state else 0.0,
        hurst=getattr(state, 'hurst_exponent', 0.5) if state else 0.5,
        dmi_diff=self_dmi_diff / 100.0,
        pid=getattr(state, 'term_pid', 0.0) if state else 0.0,
        osc_coherence=getattr(state, 'oscillation_entropy_normalized', 0.0) if state else 0.0,
        # Context from above
        context_15m_z=ctx_15m.get('z', 0.0),
        context_1h_z=ctx_1h.get('z', 0.0),
        context_1h_dmi_sign=ctx_1h.get('dmi_diff', 0.0),
        context_15m_dmi_diff=ctx_15m_dmi,
        context_1D_dmi_sign=ctx_1D.get('dmi_diff', 0.0),
        context_1D_z=ctx_1D.get('z', 0.0),
        tf_alignment_1m_15m=alignment,
    )
```

**Also update**: Any other call to `extract_features(p)` in this file — they should
all go through this same method, no changes needed (they call self.extract_features).

---

### 5. `core/timeframe_belief_network.py` — MODERATE EDIT

Update `state_to_features()` (line 670-693). The TBN workers run on multiple TFs,
but for template matching they need to produce a compatible 16D vector.

**Key decision**: When a TBN worker at 15m produces features for template matching,
it has its OWN state but doesn't have context from other TFs readily available.

**Solution**: TBN workers pass their own state as the anchor (slots 0-3, 7-9, 14-15)
and set context slots to 0.0 (unavailable). This is acceptable because:
- Template matching uses centroid DISTANCE — the context slots help differentiate
  clusters but missing context just means slightly worse matching
- The TBN's main job is conviction/direction, not template selection
- In live trading, context IS available from other workers — can be wired later

```python
@staticmethod
def state_to_features(state, tf_secs: int, depth: int = 0) -> list:
    """Convert MarketState -> 16D feature vector.
    Delegates to core.feature_extraction.extract_feature_vector().
    Context features default to 0.0 (not available from single-TF worker).
    """
    from core.feature_extraction import extract_feature_vector
    return extract_feature_vector(
        z_score=getattr(state, 'z_score', 0.0),
        velocity=getattr(state, 'velocity', 0.0),
        momentum=getattr(state, 'momentum_strength', 0.0),
        entropy_normalized=getattr(state, 'entropy_normalized', 0.0),
        adx=getattr(state, 'adx_strength', 0.0) / 100.0,
        hurst=getattr(state, 'hurst_exponent', 0.5),
        dmi_diff=(getattr(state, 'dmi_plus', 0.0) - getattr(state, 'dmi_minus', 0.0)) / 100.0,
        pid=getattr(state, 'term_pid', 0.0),
        osc_coherence=getattr(state, 'oscillation_entropy_normalized', 0.0),
        # Context: 0.0 defaults — TBN workers don't have cross-TF context here
    )
```

**Note**: `tf_secs` and `depth` params kept in signature for backward compat but
no longer used by `extract_feature_vector()`.

---

### 6. `training/trainer.py` — MINOR EDIT

Update Phase 2 discovery call (around line 4195):

```python
# BEFORE:
manifest = self.discovery_agent.scan_atlas_topdown(
    data_source,
    on_level_complete=checkpoint_callback,
    resume_manifest=resume_manifest,
    resume_levels=resume_levels,
    train_end=train_end
)

# AFTER:
manifest = self.discovery_agent.scan_atlas_1m(
    data_source,
    on_level_complete=checkpoint_callback,
    train_end=train_end
)
```

**Remove resume support** from this call — the 1m scan is fast enough (single TF)
that resume checkpointing is unnecessary. Can re-add later if needed.

Also update the import if `scan_atlas_topdown` was imported by name.

---

## What NOT to Change

- **`core/exit_engine.py`**: No changes. It reads `p75_mfe_ticks` and `avg_mfe_bar`
  from the pattern library. With 1m-anchored templates, these values will naturally
  be in the 20-80 tick range. No caps needed.

- **`core/execution_engine.py`**: Gate cascade logic unchanged. Candidate scoring,
  quality weights, conviction gates — all work the same. The `timeframe` field on
  candidates will always be `'1m'` and `depth` always 0.

- **TBN worker architecture**: Workers still run on all 11 TFs for conviction.
  The refactor only changes pattern DISCOVERY, not belief computation.

- **`core/bayesian_brain.py`**: Brain hashing unchanged. Template IDs still work.

---

## Checkpoint Compatibility

**Old checkpoints are INCOMPATIBLE**. The 16D feature vector layout changed.
Must run with `--fresh` to regenerate everything.

---

## Validation

Run: `python training/trainer.py --fresh --forward-pass`

**Success criteria** (compare vs baseline commit `678c9bc`):
- Template `p75_mfe_ticks` median < 100 (currently 486)
- Giveback-blocked trades < 100 (currently 455)
- OOS PnL >= $22,000 (currently $18,732)
- TP hit rate > 5% (currently 0.5%)
- SL/breakeven exit rate < 50% (currently 76%)

**Read these reports after**:
1. `reports/is_report.txt`
2. `reports/oos_report.txt`
3. `checkpoints/oos_analytics.txt`

---

## Implementation Order

1. `config/oracle_config.py` (1 min)
2. `core/feature_extraction.py` (5 min)
3. `training/fractal_discovery_agent.py` (30 min — bulk of work)
4. `core/fractal_clustering.py` (5 min)
5. `core/timeframe_belief_network.py` (5 min)
6. `training/trainer.py` (2 min)
7. Verify imports, run `--fresh --forward-pass`
