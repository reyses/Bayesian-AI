# BayesianBridge — Pattern Anchoring Architecture Spec

**Date**: 2026-03-10
**Author**: Moises / Claude
**Status**: DRAFT — Option B first, v2 follows
**Depends on**: PATTERN_SCALE_MISMATCH_REPORT.md

---

## Problem Statement

808 of 997 correct-direction OOS trades (81%) give back >90% of MFE and exit at
breakeven. Root cause: templates mix patterns from multiple timeframes (1s–15m),
producing aggregated MFE stats 8–30x larger than actual trade excursions. This
disables giveback protection and makes TP unreachable.

**Two-phase fix:**
- **Option B** (ship now): Anchor all pattern discovery at 1m, macro TFs become context features
- **v2** (ship later): Level-anchored architecture with CNN classification and Fibonacci-derived structural zones

---

# OPTION B: 1-Minute Anchor Refactor

## 0. Interim Fix (Today)

**File**: `core/exit_engine.py` line 220–224

```python
# BEFORE:
_anchor_mfe = 0.0
_anchor_bars = 0.0
if lib_entry:
    _anchor_mfe = lib_entry.get('p75_mfe_ticks', 0.0)
    _anchor_bars = lib_entry.get('avg_mfe_bar', 0.0)

# AFTER:
_anchor_mfe = 0.0
_anchor_bars = 0.0
if lib_entry:
    _anchor_mfe = min(lib_entry.get('p75_mfe_ticks', 0.0), 80.0)  # cap to realistic range
    _anchor_bars = min(lib_entry.get('avg_mfe_bar', 0.0), 16.0)    # cap to ~16 bars
```

**Expected impact**: ~349 trades freed from anchor patience, ~$5,852 additional PnL in OOS.
**Validation**: Run `--fresh --forward-pass`, compare OOS PnL and giveback count vs baseline.

---

## 1. Discovery: Single 1m Pass

### Current Architecture (Multi-TF)
`fractal_discovery_agent.py` → `scan_atlas_topdown()`:
- Scans 13 TFs (1D → 1s) in hierarchy
- Context TFs: 1D, 4h, 1h, 30m (enrich parent chain)
- Primary signal: 15m (macro scan starts here)
- Drill-down: 5m → 3m → 2m → 1m → 30s → 15s → 5s → 1s
- Each TF produces independent `PatternEvent` objects
- Oracle MFE measured at each pattern's native TF

### New Architecture (1m Anchor)
Discovery runs ONE pass on 1m bars. Higher TFs provide context features only.

**Changes to `fractal_discovery_agent.py`**:

```
BEFORE:
  TIMEFRAME_HIERARCHY = ['1D', '4h', '1h', ... '1s']  # 13 TFs
  PRIMARY_SIGNAL_TIMEFRAME = '15m'
  CONTEXT_ONLY_TIMEFRAMES = {'1D', '4h', '1h', '30m'}

AFTER:
  DISCOVERY_TIMEFRAME = '1m'                           # single discovery TF
  CONTEXT_TIMEFRAMES = ['1D', '4h', '1h', '30m', '15m', '5m', '3m']  # context only
```

**Discovery flow**:
1. Load all context TF data (1D through 3m) — compute MarketState per bar
2. Load 1m data — compute MarketState per bar
3. For each 1m bar, attach context from the enclosing bar of each higher TF:
   - `context_15m_z`, `context_15m_adx`, `context_15m_dmi_diff`
   - `context_1h_z`, `context_1h_adx`, etc.
   - `context_1D_z`, `context_1D_adx`, etc.
4. Detect patterns (BAND_REVERSAL, MOMENTUM_BREAK) on 1m bars only
5. Oracle MFE measured in 1m bars with single lookahead:
   - `ORACLE_LOOKAHEAD_BARS['1m'] = 60` (already 60 bars = 1 hour)

**What gets deleted**:
- Top-down drill-down logic (`current_windows`, `_merge_windows`)
- Multi-TF pattern discovery (no patterns from 15m, 5m, 30s, 15s, 5s, 1s)
- Depth tracking across TF levels (depth is always 0 at 1m)

**What stays**:
- `_consult_oracle()` — unchanged, just only called for 1m patterns
- `_detect_patterns()` — unchanged, runs on 1m MarketState
- Parent chain concept — replaced by flat context dict from higher TFs

---

## 2. Feature Vector: Replace Hierarchy with Context

### Current 16D Vector
```
[0]  abs(z_score)           — from pattern's native TF
[1]  log1p(|velocity|)      — from pattern's native TF
[2]  log1p(|momentum|)      — from pattern's native TF
[3]  entropy_normalized     — from pattern's native TF
[4]  log2(tf_seconds)       — encodes which TF discovered pattern (VARIES)
[5]  depth                  — fractal drill-down depth (VARIES: 0-8)
[6]  parent_is_band_reversal
[7]  adx / 100
[8]  hurst_exponent
[9]  dmi_diff / 100
[10] parent_z
[11] parent_dmi_diff
[12] root_is_roche
[13] tf_alignment
[14] term_pid
[15] oscillation_entropy_normalized
```

### New 16D Vector (1m + Context)
```
[0]  abs(z_score)           — 1m z-score (always 1m)
[1]  log1p(|velocity|)      — 1m velocity
[2]  log1p(|momentum|)      — 1m momentum
[3]  entropy_normalized     — 1m entropy
[4]  context_15m_z          — REPLACES log2(tf_seconds) — 15m band location
[5]  context_1h_z           — REPLACES depth — 1h band location
[6]  context_1h_dmi_sign    — REPLACES parent_is_band_reversal — 1h trend direction
[7]  adx / 100              — 1m ADX
[8]  hurst_exponent         — 1m Hurst
[9]  dmi_diff / 100         — 1m DMI diff
[10] context_15m_dmi_diff   — REPLACES parent_z — 15m trend strength
[11] context_1D_dmi_sign    — REPLACES parent_dmi_diff — daily regime
[12] context_1D_z           — REPLACES root_is_roche — daily band location
[13] tf_alignment_1m_15m    — sign(1m_dmi) * sign(15m_dmi) — near alignment
[14] term_pid               — unchanged
[15] oscillation_entropy    — unchanged
```

**Key design decision**: Higher TFs earn their weight as *features* (z-score position,
trend direction) rather than as independent pattern sources. The TBN still scores
multi-TF conviction — this change only affects how patterns are discovered and clustered.

**Impact on clustering**: All patterns share the same base resolution (1m).
`log2(tf_seconds)` was feature[4] and ranged from 0 (1s) to 16.4 (1D) — this
created massive variance in K-Means. Replacing it with `context_15m_z` (range: -3 to +3)
gives tighter clusters that are scale-homogeneous by construction.

**File**: `core/feature_extraction.py` — rewrite `extract_feature_vector()` signature
and mapping. Must update both callers:
- `core/fractal_clustering.py` → `extract_features()`
- `core/timeframe_belief_network.py` → `state_to_features()`

**CHECKPOINT COMPATIBILITY**: Old checkpoints use the 16D layout with tf_seconds/depth.
New checkpoints are incompatible. Must retrain from scratch (`--fresh`).

---

## 3. Oracle Config: Unified 1m Horizon

### Current: `config/oracle_config.py`
```python
ORACLE_LOOKAHEAD_BARS = {
    '1s':  300,   # 5 minutes
    '5s':  120,   # 10 minutes
    '15s':  60,   # 15 minutes
    '1m':   60,   # 1 hour
    '5m':   24,   # 2 hours
    '15m':  16,   # 4 hours
    ...
}
```

### New: Only 1m matters
```python
ORACLE_LOOKAHEAD_BARS = {
    '1m':   16,   # 16 minutes — matches median trade hold + oscillation cycle
}
```

**Why 16 bars, not 60**: Current 60 bars = 1 hour horizon. But actual trades resolve
in 1–5 minutes (median 5min profitable hold). 16 minutes captures the full oscillation
cycle (8 min peak) with room for late resolution. This anchors MFE at the scale trades
actually operate on.

**Result**: Template `p75_mfe_ticks` will be in the 20–80 tick range instead of 200–5,500.
TP, SL, and anchor patience become reachable by default.

---

## 4. Clustering: No Changes Needed

`core/fractal_clustering.py` runs K-Means on the 16D feature vector. With all patterns
at 1m resolution and the new feature layout, clustering naturally produces
scale-homogeneous templates. No code changes required — the input data is just cleaner.

**One exception**: `_compute_template_stats()` (line 245–278) aggregates MFE from
`pattern.oracle_meta`. Since all oracle MFE is now measured at 1m/16-bar horizon,
the aggregation produces coherent stats. The p75_mfe_ticks values will drop from
hundreds/thousands to tens — matching actual trade behavior.

---

## 5. Exit Engine: No Changes Needed

`core/exit_engine.py` already reads `p75_mfe_ticks` and `avg_mfe_bar` from the
pattern library. With 1m-anchored templates:
- `anchor_mfe_ticks` ≈ 30–80 (was 200–5,500)
- `anchor_mfe_bars` ≈ 8–16 (was 20–200)
- 30% patience threshold ≈ 9–24 ticks (was 60–1,650) — **reachable**
- TP ≈ 17–68 ticks (was 170–4,675) — **reachable**

The interim 80-tick cap from Section 0 becomes unnecessary after the refactor,
but is harmless to leave in as a safety net.

---

## 6. Execution Engine & TBN: Minimal Changes

**ExecutionEngine** (`core/execution_engine.py`):
- Gate cascade logic unchanged — still scores candidates by distance, conviction, etc.
- Candidate creation changes: `timeframe` field is always `'1m'`
- `depth` field is always 0 (no hierarchy)

**TimeframeBeliefNetwork** (`core/timeframe_belief_network.py`):
- Workers still run across all 11 TFs — this is the multi-TF *consensus* layer
- `state_to_features()` must use the new 16D layout (same as clustering)
- Template matching against centroids uses new feature space

---

## 7. Live Engine: Minimal Changes

`live/live_engine.py` already has a 1m bar aggregator. Changes:
- Pattern detection runs on 1m bars only (remove any sub-1m detection if present)
- Context features computed from higher-TF aggregated bars (already available)
- Feature vector construction uses new layout

---

## 8. Training Pipeline: Changes to `trainer.py`

Phase 2 (Pattern Discovery):
- Calls `scan_atlas_topdown()` → replace with `scan_atlas_1m()` (new method)
- Passes single TF to discovery agent

Phase 3 (Template Optimization):
- No changes — K-Means runs on whatever features it receives

Phase 4 (IS Forward Pass):
- Candidate generation uses 1m patterns only
- Feature extraction uses new layout

Phase 5 (OOS Validation):
- Same as Phase 4

---

## 9. Execution Sequence

1. **Interim fix** (today): 80-tick cap in exit_engine.py
2. **Branch**: `feature/1m-anchor`
3. **Modify** `config/oracle_config.py` — single 1m entry, 16 bars
4. **Modify** `core/feature_extraction.py` — new 16D layout
5. **Modify** `training/fractal_discovery_agent.py` — single 1m scan + context enrichment
6. **Modify** `core/fractal_clustering.py` → `extract_features()` — delegate to new layout
7. **Modify** `core/timeframe_belief_network.py` → `state_to_features()` — delegate to new layout
8. **Modify** `training/trainer.py` Phase 2 — call new discovery method
9. **Run** `python training/trainer.py --fresh --forward-pass`
10. **Compare** IS/OOS metrics vs baseline (commit de829f8)
11. **Merge** if OOS PnL improves and giveback count drops

**Success criteria**:
- Template p75_mfe_ticks median < 100 (currently 486)
- Giveback-blocked trades < 100 (currently 455)
- OOS PnL ≥ $22,000 (currently $18,732 — expect improvement from freed givebacks)
- TP hit rate > 5% (currently 0.5%)

---
---

# V2: Level-Anchored Architecture (CNN + Fibonacci)

**Status**: Design spec — build after Option B is validated
**Prerequisite**: Option B running in production with clean 1m-anchored data

---

## 10. Architecture Overview

The v2 system restores the original macro-to-micro cascade as an explicit hierarchy:

```
Layer 1: STRUCTURAL LEVELS (daily, ~90-day rolling window)
  → Detects horizontal S/R zones from clustered swing highs/lows
  → Lifecycle: FORMING → CONFIRMED → ACTIVE → BROKEN → FLIPPED
  → Output: table of active levels with strength scores

Layer 2: FIBONACCI ZONES (between adjacent levels)
  → Computes retracement zones (0.236, 0.382, 0.5, 0.618, 0.786)
  → Defines "where to look" when price approaches a level
  → Output: active Fib zones with expected bounce/break probability

Layer 3: CNN PATTERN CLASSIFIER (1m bars at level/zone interactions)
  → Classifies price behavior when entering a Fib zone near a level
  → Replaces both K-Means templates AND seed primitives
  → Output: shape class + confidence + direction signal

Layer 4: EXECUTION (existing gate cascade)
  → Entry: CNN shape at Fib zone near confirmed level → gate cascade
  → Exit: MFE = level-to-level (naturally calibrated)
  → Brain: P(win) per (level_type × fib_zone × shape_class) tuple
```

---

## 11. Level Detection Module

**New file**: `core/level_structure.py`

### Level Discovery (No Overfitting)

**Input**: Daily OHLC bars, rolling 90-day window
**Method**:
1. Identify swing highs/lows using N-bar confirmation (e.g., 3 bars lower on each side)
2. Cluster nearby pivots using DBSCAN with tolerance = 0.15% of current price
   - MNQ at 25,000 → tolerance ≈ 37.5 points (150 ticks)
3. Filter: clusters with < 3 touches are discarded (not yet "earned")
4. Strength score = `touch_count × recency_weight × avg_reaction_magnitude`
5. Maximum active levels: 8–12 (prevent over-segmentation)

### Level Lifecycle State Machine

```
FORMING    → touch_count < 3, no trade signals generated
CONFIRMED  → touch_count >= 3, generates "approach" signals
ACTIVE     → CONFIRMED + tested within last 20 days
BROKEN     → daily close through zone with magnitude > 1σ of recent ATR
FLIPPED    → BROKEN + price returns to test from opposite side
DECAYED    → ACTIVE but untested for > 45 days → reduce strength, eventually remove
```

### Level Data Structure
```python
@dataclass
class StructuralLevel:
    price_zone: Tuple[float, float]  # (lower_bound, upper_bound)
    touch_count: int
    state: LevelState                # FORMING/CONFIRMED/ACTIVE/BROKEN/FLIPPED/DECAYED
    first_touch: datetime
    last_touch: datetime
    strength: float                  # composite score
    break_direction: Optional[str]   # 'up' or 'down' (set on BROKEN)
    formation_side: str              # 'support' or 'resistance'
    reactions: List[LevelReaction]   # historical touch data for statistics
```

### Anti-Overfitting Controls
- Minimum cluster separation: levels must be > 0.5% apart (no micro-levels)
- Maximum level count: hard cap at 12 active levels
- Recency decay: strength *= 0.95 per untested day (half-life ≈ 14 days)
- Confirmation lag: FORMING → CONFIRMED requires 3 touches over > 5 days
  (prevents same-day noise from creating levels)

---

## 12. Fibonacci Zone Engine

**New file**: `core/fib_zones.py`

### Zone Computation
Between each pair of adjacent active levels, compute Fibonacci retracement zones:

```
Level A (support): 24,800
Level B (resistance): 25,400
Range: 600 points

Fib zones (from B toward A):
  0.236 retracement: 25,258
  0.382 retracement: 25,171
  0.500 retracement: 25,100
  0.618 retracement: 25,029
  0.786 retracement: 24,929
```

Each zone has a tolerance band (± 0.1% of range = ±0.6 points here).

### Zone Activation
A Fib zone becomes "active" when price enters its tolerance band. This is the
trigger for Layer 3 (CNN classification). Zone activation is the equivalent of
the current system's pattern detection — but anchored to structural significance
instead of statistical coincidence.

### MFE Target: Level-to-Level
When a trade enters at a Fib zone, the natural MFE target is the next structural
level in the trade's direction. This is self-calibrating:
- Small range between levels → small MFE expectation
- Large range → large MFE expectation
- No arbitrary oracle horizon needed

---

## 13. CNN Shape Classifier

**New file**: `core/cnn_classifier.py`

### Architecture: 1D Temporal CNN
```
Input:  16 × 1m normalized OHLC bars (64 values: O, H, L, C per bar)
        + 4 context features (15m_z, 1h_z, 1D_z, level_distance)
        = 68-dim input

Conv1D: 3 layers
  Layer 1: 32 filters, kernel=3, ReLU, BatchNorm
  Layer 2: 64 filters, kernel=3, ReLU, BatchNorm
  Layer 3: 64 filters, kernel=3, ReLU, BatchNorm
  GlobalMaxPool

Dense:  128 → 64 → N_classes (softmax)

Output: shape_class (learned vocabulary), confidence, direction_logit
```

### Why CNN Over Pearson Correlation (Primitives)
- Primitives use 20 hand-crafted shapes — CNN discovers what shapes actually matter
- Pearson can't capture asymmetry (front-skewed vs back-skewed with same correlation)
- CNN learns from labeled data (oracle outcomes) — shapes are tied to profitability
- Inference: ~0.1ms per classification on GPU (negligible in live loop)

### Training Data
Generated by Option B infrastructure:
1. Run Option B for N months, collect all 1m-anchored pattern events
2. Label each event with oracle outcome (direction, MFE, MAE)
3. For each event, extract 16-bar 1m OHLC window + context features
4. Train CNN on (window, context) → (direction, magnitude_class)
5. Magnitude classes: NOISE / SCALP / STANDARD / HOME_RUN (4-class)

### Integration Point
CNN replaces K-Means template matching at entry time:
- Current: `fractal_clustering.extract_features()` → centroid distance → template ID
- v2: `cnn_classifier.classify()` → shape_class + confidence → entry signal

Brain key becomes: `(level_state, fib_zone, shape_class)` instead of `(template_id, ...)`

---

## 14. Execution Integration

### Entry Flow (v2)
```
1. level_structure detects price approaching ACTIVE level
2. fib_zones identifies which Fib zone price is in
3. cnn_classifier classifies the 16-bar 1m window → shape + confidence
4. Brain lookup: P(win | level_state, fib_zone, shape_class)
5. Gate cascade: conviction, quality score, risk checks
6. Entry if all gates pass
```

### Exit Flow (v2)
```
MFE target: next structural level in trade direction
SL: derived from Fib zone width (e.g., 0.5 × zone_to_level distance)
Giveback: enabled immediately (no anchor patience — MFE is level-calibrated)
Breakeven: at 50% of level-to-level distance
```

### Brain Key Structure
```python
# Current:
state_key = f"{template_id}_{z_bucket}_{depth}"

# v2:
state_key = f"{level_state}_{fib_zone_id}_{shape_class}_{direction}"
# Example: "ACTIVE_support_fib382_V_REVERSAL_UP_long"
```

---

## 15. v2 Implementation Sequence

**Prerequisites**: Option B running, 3+ months of 1m-anchored trade data collected.

1. `core/level_structure.py` — level detection + lifecycle state machine
2. `core/fib_zones.py` — zone computation between active levels
3. Backtest level detection on ATLAS daily data — validate touch counts, anti-overfit
4. `core/cnn_classifier.py` — model architecture + training pipeline
5. Generate CNN training data from Option B's 1m pattern events
6. Train CNN, evaluate direction accuracy vs current 56% baseline
7. Integrate into execution engine — new entry flow
8. Retrain brain with new key structure
9. IS/OOS validation against Option B baseline
10. Live deployment

**Estimated timeline**: 4–8 weeks after Option B stabilizes.

---

## 16. Risk Assessment

| Risk | Mitigation |
|------|-----------|
| 1m anchor loses profitable higher-TF patterns | TBN still provides multi-TF conviction; context features preserve macro info |
| K reduction = fewer patterns per cluster | Monitor cluster sizes; adaptive K per pool if needed |
| CNN overfits to training period | Walk-forward validation; retrain monthly |
| Level detection creates too many levels | Hard cap at 12; DBSCAN separation minimum |
| Fib zones are too tight/wide | Tolerance = f(ATR), adaptive to volatility regime |
| v2 brain has sparse state space | Start with coarse keys (level_state × shape_class only), add Fib granularity after data accumulates |

---

## Appendix: File Change Summary

### Option B (Immediate)
| File | Change Type | Scope |
|------|------------|-------|
| `core/exit_engine.py` | EDIT | 2 lines — interim cap |
| `config/oracle_config.py` | EDIT | Reduce to single 1m entry |
| `core/feature_extraction.py` | REWRITE | New 16D layout |
| `training/fractal_discovery_agent.py` | MAJOR EDIT | Single 1m scan + context enrichment |
| `core/fractal_clustering.py` | MINOR EDIT | Update extract_features() delegation |
| `core/timeframe_belief_network.py` | MINOR EDIT | Update state_to_features() delegation |
| `training/trainer.py` | MINOR EDIT | Phase 2 calls new discovery method |

### v2 (Future)
| File | Change Type | Scope |
|------|------------|-------|
| `core/level_structure.py` | NEW | Level detection + lifecycle |
| `core/fib_zones.py` | NEW | Fibonacci zone computation |
| `core/cnn_classifier.py` | NEW | 1D CNN model + inference |
| `training/cnn_trainer.py` | NEW | CNN training pipeline |
| `core/execution_engine.py` | EDIT | New entry flow with levels/zones/CNN |
| `core/bayesian_brain.py` | EDIT | New key structure |
| `core/exit_engine.py` | EDIT | Level-to-level MFE targets |
