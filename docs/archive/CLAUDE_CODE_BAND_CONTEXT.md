# CLAUDE CODE INSTRUCTIONS: Standard Error Band Multi-TF Context
# Project: BayesianBridge MNQ Trading System
# Priority: HIGH — fixes systemic short bias
# Date: March 6, 2026

## PROBLEM

The system has a persistent SHORT bias. Root cause:

In `quantum_field_engine.py`, z_score = (price - regression_center) / sigma.
In an uptrend, regression_center LAGS price → z_score is persistently positive
→ every downstream direction decision defaults to SHORT.

The user's MANUAL method: check where price sits relative to Standard Error
Bands (1σ, 2σ, 3σ) across MULTIPLE timeframes simultaneously. When Daily
shows -2σ (support), 4H shows -1σ (approaching), and 1H shows -3σ (extreme),
the CONFLUENCE says LONG — regardless of what any single TF's z_score sign says.

The existing `TimeframeBeliefNetwork` (training/timeframe_belief_network.py)
already has 11 workers, each computing z_score at their own TF scale.
But this information is NOT structured as "which σ band am I in" and is NOT
used for multi-TF band confluence in the direction decision.

## SOLUTION

Add a `BandContext` dataclass to each worker's output, then add a
`get_band_confluence()` method to TimeframeBeliefNetwork that aggregates
band positions across all active TFs into a directional signal.

This replaces the z_score sign fallback at the bottom of the direction cascade.

---

## CHANGES TO: `training/timeframe_belief_network.py`

### 1. Add BandContext dataclass (after WorkerBelief)

```python
@dataclass
class BandContext:
    """Where price sits relative to Standard Error Bands at one TF."""
    tf_seconds: int
    z_score: float              # raw z_score from physics engine
    sigma: float                # regression sigma at this TF
    center: float               # regression center (fair value)
    band: int                   # which band: -3, -2, -1, 0, 1, 2, 3
    band_position: float        # continuous: -1.0 (lower extreme) to +1.0 (upper extreme)
    at_support: bool            # z <= -1.0 (below center, near/at lower bands)
    at_resistance: bool         # z >= 1.0 (above center, near/at upper bands)
    band_label: str             # human readable: '-2σ', '+1σ', 'center', etc.
```

**band calculation:**
```python
# Discrete band (integer sigma level, clamped to [-3, 3])
band = int(np.clip(np.round(z_score), -3, 3))

# Continuous position normalized to [-1, 1]
# -1.0 = at -3σ (extreme support), 0.0 = at center, +1.0 = at +3σ (extreme resistance)  
band_position = np.clip(z_score / 3.0, -1.0, 1.0)

# Support/resistance flags
at_support = z_score <= -1.0
at_resistance = z_score >= 1.0

# Label
if abs(z_score) < 0.5:
    band_label = 'center'
else:
    sign = '+' if z_score > 0 else '-'
    band_label = f'{sign}{abs(band)}σ'
```

### 2. Add band_context to WorkerBelief

Add one field to the existing `WorkerBelief` dataclass:

```python
@dataclass
class WorkerBelief:
    tf_seconds:    int
    dir_prob:      float
    pred_mfe:      float
    template_id:   int
    tf_bar_idx:    int
    conviction:    float
    wave_maturity: float = 0.0
    band_context:  Optional[BandContext] = None   # ← ADD THIS
```

### 3. Populate band_context in TimeframeWorker._analyze()

In the `_analyze` method of `TimeframeWorker`, after the existing physics
blend section, extract band context from the state:

```python
# ── Band Context (Standard Error Bands) ──────────────────────────
_z = float(getattr(state, 'z_score', 0.0))
_sigma = float(getattr(state, 'sigma_fractal', 0.0))
_center = float(getattr(state, 'center_position', 0.0))

_band_int = int(np.clip(np.round(_z), -3, 3))
_band_pos = float(np.clip(_z / 3.0, -1.0, 1.0))
_at_sup = _z <= -1.0
_at_res = _z >= 1.0

if abs(_z) < 0.5:
    _band_lbl = 'center'
else:
    _sign = '+' if _z > 0 else '-'
    _band_lbl = f'{_sign}{abs(_band_int)}σ'

_band_ctx = BandContext(
    tf_seconds=self.tf_seconds,
    z_score=_z,
    sigma=_sigma,
    center=_center,
    band=_band_int,
    band_position=_band_pos,
    at_support=_at_sup,
    at_resistance=_at_res,
    band_label=_band_lbl,
)
```

Then include `band_context=_band_ctx` in the `WorkerBelief(...)` constructor
at the end of `_analyze()`.

### 4. Add get_band_confluence() to TimeframeBeliefNetwork

New public method on the `TimeframeBeliefNetwork` class:

```python
def get_band_confluence(self) -> Optional[dict]:
    """
    Multi-TF Standard Error Band confluence.
    
    Aggregates band positions across all active workers to determine
    structural direction. This is the user's manual method automated:
    
    - If majority of TFs show price at support bands (z <= -1σ) → LONG
    - If majority show resistance bands (z >= +1σ) → SHORT  
    - If mixed → no signal (return direction=None)
    
    Weighting: higher TFs carry more weight (same as path conviction).
    
    Returns:
        {
            'direction': 'long' | 'short' | None,
            'strength': float,          # 0.0-1.0, how strong the confluence is
            'support_score': float,      # weighted sum of support signals
            'resistance_score': float,   # weighted sum of resistance signals
            'active_bands': int,         # how many TFs contributed
            'band_summary': str,         # "1h:-2σ | 30m:-1σ | 15m:-3σ → LONG"
            'per_tf': dict,              # {tf_label: BandContext} for each active worker
        }
        Returns None if < 3 active workers have band data.
    """
    active_bands = {}
    for tf, worker in self.workers.items():
        b = worker.current_belief
        if b is not None and b.band_context is not None:
            active_bands[tf] = b.band_context
    
    if len(active_bands) < 3:
        return None
    
    # Weighted scoring
    support_score = 0.0
    resistance_score = 0.0
    total_weight = 0.0
    
    per_tf = {}
    summary_parts = []
    
    for tf, ctx in active_bands.items():
        w = self._weight_map.get(tf, 1.0)
        total_weight += w
        tf_label = self._TF_LABELS.get(tf, str(tf))
        per_tf[tf_label] = ctx
        
        # Score: how deep into support or resistance bands
        # z <= -1 contributes to support_score (weighted by how extreme)
        # z >= +1 contributes to resistance_score
        if ctx.at_support:
            # Deeper into support = stronger signal
            # z=-1 → 1.0, z=-2 → 2.0, z=-3 → 3.0
            support_score += w * abs(ctx.z_score)
            summary_parts.append(f"{tf_label}:{ctx.band_label}")
        elif ctx.at_resistance:
            resistance_score += w * ctx.z_score
            summary_parts.append(f"{tf_label}:{ctx.band_label}")
        else:
            summary_parts.append(f"{tf_label}:center")
    
    # Normalize
    if total_weight > 0:
        support_score /= total_weight
        resistance_score /= total_weight
    
    # Direction decision
    # Need clear majority: one side must be >2x the other
    if support_score > resistance_score * 2 and support_score > 0.5:
        direction = 'long'
        strength = min(1.0, support_score / 3.0)
    elif resistance_score > support_score * 2 and resistance_score > 0.5:
        direction = 'short'
        strength = min(1.0, resistance_score / 3.0)
    else:
        direction = None  # mixed signals, no confluence
        strength = 0.0
    
    arrow = '→ LONG' if direction == 'long' else ('→ SHORT' if direction == 'short' else '→ MIXED')
    summary = ' | '.join(summary_parts) + f' {arrow}'
    
    return {
        'direction': direction,
        'strength': strength,
        'support_score': support_score,
        'resistance_score': resistance_score,
        'active_bands': len(active_bands),
        'band_summary': summary,
        'per_tf': per_tf,
    }
```

### 5. Add band_confluence to get_belief() output

Add to the existing `BeliefState` dataclass:

```python
@dataclass
class BeliefState:
    direction:              str
    conviction:             float
    predicted_mfe:          float
    active_levels:          int
    wave_maturity:          float = 0.0
    decision_wave_maturity: float = 0.0
    tf_beliefs:     Dict[int, WorkerBelief] = field(default_factory=dict)
    band_confluence: Optional[dict] = None   # ← ADD THIS
```

At the end of `get_belief()`, before `return BeliefState(...)`, add:

```python
band_confluence = self.get_band_confluence()
```

And include `band_confluence=band_confluence` in the BeliefState constructor.

### 6. Add band_context to worker snapshot

Update `get_worker_snapshot()` to include band data:

```python
def get_worker_snapshot(self) -> dict:
    snap = {}
    for tf, worker in self.workers.items():
        b = worker.current_belief
        if b is not None:
            entry = {
                'd':   round(b.dir_prob,      3),
                'c':   round(b.conviction,    3),
                'm':   round(b.wave_maturity, 3),
                'mfe': round(b.pred_mfe,      1),
            }
            # Add band context if available
            if b.band_context is not None:
                entry['z'] = round(b.band_context.z_score, 2)
                entry['band'] = b.band_context.band
                entry['band_label'] = b.band_context.band_label
            snap[self._TF_LABELS.get(tf, str(tf))] = entry
    return snap
```

---

## CHANGES TO: `live/live_engine.py`

### 7. Use band confluence in _determine_direction()

In the `_determine_direction` method of `LiveEngine`, add band confluence
as a NEW priority level BETWEEN the existing priorities.

Current priority order:
```
Priority 0: live direction bias (ping-pong refinement)
Priority 1: signed MFE regression
Priority 2: balanced direction logistic regression  
Priority 3: template aggregate bias
Priority 4: live DMI (trend-following)
Fallback: velocity sign  ← THIS IS WHERE THE SHORT BIAS LIVES
```

**Insert Priority 3.5: Band Confluence** (after template bias, before DMI):

```python
# Priority 3.5: Multi-TF band confluence (Standard Error Bands)
# This is the structural direction signal — replaces z_score sign fallback
if side is None:
    band_signal = self._belief_network.get_band_confluence()
    if band_signal is not None and band_signal['direction'] is not None:
        side = band_signal['direction']
        _p_long = 0.5 + (0.3 if side == 'long' else -0.3) * band_signal['strength']
        return side, _p_long, 'band_confluence'
```

**Also replace the final velocity fallback** to use band confluence first:

```python
# Priority 4: live DMI (trend-following)  
# ...existing DMI code...

# FINAL FALLBACK: band confluence > velocity sign
# Old code: side = 'long' if vel >= 0 else 'short'
# New code: prefer band signal, fall back to velocity only if no bands
if side is None:
    band_signal = self._belief_network.get_band_confluence()
    if band_signal is not None and band_signal['direction'] is not None:
        side = band_signal['direction']
        return side, 0.55 if side == 'long' else 0.45, 'band_fallback'

vel = getattr(s, 'particle_velocity', 0.0)
side = 'long' if vel >= 0 else 'short'
return side, 0.55 if side == 'long' else 0.45, 'velocity'
```

### 8. Log band confluence at entry (for diagnostics)

In `_check_entry()`, after the trade fires, add band info to the log:

```python
# After the existing logger.info(f"ENTRY: ...") line, add:
_band = self._belief_network.get_band_confluence()
if _band:
    logger.info(f"  BANDS: {_band['band_summary']}")
```

---

## CHANGES TO: `training/orchestrator.py`

### 9. Use band confluence in the forward pass direction cascade

In `run_forward_pass()`, find the direction decision block (search for
`# Priority 3: live DMI (trend-following)`). Add band confluence
as Priority 2.5, BEFORE DMI:

```python
# Priority 2.5: Multi-TF band confluence
if side is None:
    _band = belief_network.get_band_confluence()
    if _band is not None and _band['direction'] is not None:
        side = _band['direction']
```

And update the final velocity fallback to check bands first:

```python
# Current fallback:
#   _vel = getattr(_live_s, 'particle_velocity', 0.0)
#   side = 'long' if _vel >= 0 else 'short'
#
# Replace with:
if side is None:
    _band = belief_network.get_band_confluence()
    if _band is not None and _band['direction'] is not None:
        side = _band['direction']
    else:
        _vel = getattr(_live_s, 'particle_velocity', 0.0)
        side = 'long' if _vel >= 0 else 'short'
```

### 10. Add band confluence to oracle trade log

In the `pending_oracle` dict (built at trade entry in `run_forward_pass`),
add band diagnostic fields:

```python
# Add to pending_oracle dict:
'band_direction': _band['direction'] if _band else None,
'band_strength': round(_band['strength'], 3) if _band else 0.0,
'band_summary': _band['band_summary'] if _band else '',
```

### 11. Add band confluence to signal log

In the `_dm_rec` function (builds signal_log records), add band fields:

```python
# Add to the returned dict:
'band_dir': '',      # filled at entry: 'long' | 'short' | ''
'band_strength': 0.0,
'band_summary': '',
```

And populate them in the 'traded' record:

```python
_dm_entry['band_dir'] = _band['direction'] if _band else ''
_dm_entry['band_strength'] = round(_band['strength'], 3) if _band else 0.0
_dm_entry['band_summary'] = _band['band_summary'] if _band else ''
```

---

## DO NOT CHANGE

- `quantum_field_engine.py` — z_score computation stays the same
- `three_body_state.py` — no changes
- `wave_rider.py` — no changes
- `core/` anything else — no changes
- The existing physics blend in `TimeframeWorker._analyze()` — keep it,
  band context is ADDITIONAL data, not a replacement

---

## TESTING

After implementation:

### Quick smoke test (no ATLAS data needed)

```python
# In Python console:
from training.timeframe_belief_network import BandContext
import numpy as np

# Simulate a -2σ support zone
ctx = BandContext(
    tf_seconds=3600, z_score=-2.1, sigma=5.0, center=25100.0,
    band=-2, band_position=-0.7, at_support=True, at_resistance=False,
    band_label='-2σ'
)
print(ctx)
# Should print: BandContext(tf_seconds=3600, z_score=-2.1, ..., band_label='-2σ')
```

### Forward pass validation

Run the existing forward pass and check the report for:
1. `band_confluence` appearing in oracle_trade_log.csv
2. Direction distribution should shift — fewer SHORT trades in uptrend periods
3. Grep signal_log.csv for `band_dir` column to verify it's populated

```bash
python training/orchestrator.py --forward-pass --data DATA/ATLAS
# Then check:
# - oracle_trade_log.csv has band_direction column
# - Compare LONG/SHORT ratio vs previous run
```

### Diagnostic: band confluence agreement with oracle

After a forward pass, this one-liner shows if bands predict oracle direction:

```python
import pandas as pd
df = pd.read_csv('checkpoints/oracle_trade_log.csv')
# How often does band_direction match oracle direction?
df['band_correct'] = (
    ((df['band_direction'] == 'long') & (df['oracle_label'] > 0)) |
    ((df['band_direction'] == 'short') & (df['oracle_label'] < 0))
)
print(f"Band confluence accuracy: {df['band_correct'].mean():.1%}")
```

If band accuracy > 55%, the signal is working. Compare against baseline
(velocity fallback was ~50% = random).

---

## SUMMARY OF FILES CHANGED

| File | Change |
|------|--------|
| `training/timeframe_belief_network.py` | Add `BandContext` dataclass, populate in `_analyze()`, add `get_band_confluence()`, update `BeliefState`, update `get_worker_snapshot()` |
| `live/live_engine.py` | Add band confluence to `_determine_direction()` priority chain, log at entry |
| `training/orchestrator.py` | Add band confluence to forward pass direction cascade, add to oracle/signal logs |

Total: ~100 lines added across 3 files. No deletions. No structural changes.
