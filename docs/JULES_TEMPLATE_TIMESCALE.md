# Jules Task: Template Time-Scale — avg_mfe_bar per Template + Worker Time-Aware Exit

## Goal
Give every PatternTemplate its own time signature (`avg_mfe_bar`, `p75_mfe_bar`).
Pass this into the active position so the belief network workers can fire a
time-exhaustion exit signal once the trade has held past its historical peak bar.

Currently workers exit on wave_maturity and conviction only — they have zero
concept of time. A 15s leaf template that historically peaks at bar 12 (3 min)
gets held identically to a 1h template that peaks at bar 800 (3 hrs). This is
the root cause of MFE bleed and reversed trades holding 35,000+ bars.

---

## Files to Modify (in order)

### 1. `training/fractal_clustering.py`

**A. Add two fields to `PatternTemplate` dataclass** (after `regression_sigma_ticks`):

```python
# Time-scale calibration (populated by _aggregate_oracle_intelligence)
avg_mfe_bar:  float = 0.0   # mean bar index where MFE peaked (0-based, 15s bars)
p75_mfe_bar:  float = 0.0   # 75th-pct mfe_bar -- conservative "still moving" window
```

**B. In `_aggregate_oracle_intelligence`**, after the MFE/MAE stats block, add:

```python
# Time-scale: which bar did MFE peak?
mfe_bars = [
    p.oracle_meta.get('mfe_bar', None)
    for p in patterns
    if hasattr(p, 'oracle_meta') and p.oracle_meta.get('mfe_bar') is not None
    and p.oracle_meta.get('mfe_bar') >= 0
]
if len(mfe_bars) >= TEMPLATE_MIN_MEMBERS_FOR_STATS:
    template.avg_mfe_bar = float(np.mean(mfe_bars))
    template.p75_mfe_bar = float(np.percentile(mfe_bars, 75))
```

**C. In `_serialize_template`** (the dict returned for pattern_library), add:

```python
'avg_mfe_bar': getattr(template, 'avg_mfe_bar', 0.0),
'p75_mfe_bar': getattr(template, 'p75_mfe_bar', 0.0),
```

---

### 2. `training/timeframe_belief_network.py`

**A. Add a method `set_active_trade_timescale(avg_mfe_bar, p75_mfe_bar)`**:

```python
def set_active_trade_timescale(self, avg_mfe_bar: float, p75_mfe_bar: float):
    """
    Called at trade entry. Stores the template's historical time-to-MFE
    so get_exit_signal can fire a time-exhaustion signal when bars_held
    exceeds the expected peak window.
    """
    self._trade_avg_mfe_bar = avg_mfe_bar
    self._trade_p75_mfe_bar = p75_mfe_bar
    self._trade_bars_held   = 0

def tick_trade_bar(self):
    """Call once per 15s bar while a position is open."""
    self._trade_bars_held = getattr(self, '_trade_bars_held', 0) + 1

def clear_active_trade_timescale(self):
    """Call at trade exit."""
    self._trade_avg_mfe_bar = 0.0
    self._trade_p75_mfe_bar = 0.0
    self._trade_bars_held   = 0
```

**B. In `__init__`**, initialise:

```python
self._trade_avg_mfe_bar = 0.0
self._trade_p75_mfe_bar = 0.0
self._trade_bars_held   = 0
```

**C. Update `get_exit_signal`** to incorporate time exhaustion:

After the existing `tighten` and `widen` logic, add:

```python
# Time-exhaustion: if template has a time signature and we've held past
# 1.5× the avg_mfe_bar, the historical peak window is likely over.
# Tighten aggressively at 1.5×, urgent exit at 2.5× p75_mfe_bar.
_time_tighten = False
_time_urgent  = False
if self._trade_avg_mfe_bar > 0:
    _ratio = self._trade_bars_held / self._trade_avg_mfe_bar
    _p75   = self._trade_p75_mfe_bar if self._trade_p75_mfe_bar > 0 else self._trade_avg_mfe_bar * 1.5
    if self._trade_bars_held > _p75 * 2.5:
        _time_urgent  = True   # past 2.5× conservative peak → exit now
    elif _ratio > 1.5:
        _time_tighten = True   # past 1.5× average peak → trail tighter

tighten = tighten or _time_tighten
urgent  = urgent  or _time_urgent
```

Update `reason` string to include time signals:

```python
reason = ('urgent_flip'    if urgent and not _time_urgent else
          'time_exhausted' if _time_urgent  else
          'wave_mature'    if wave_mature > self.TIGHTEN_TRAIL_WAVE_MATURITY_THRESHOLD else
          'time_tighten'   if _time_tighten else
          'aligned_fresh'  if widen else
          'low_conviction' if not belief.is_confident else 'neutral')
```

---

### 3. `training/orchestrator.py`

**A. At trade entry** (both normal and worker-bypass paths), after `active_entry_depth = ...`:

```python
# Pass template time-scale to belief network
_tmpl_avg_mfe_bar = self.pattern_library.get(best_tid, {}).get('avg_mfe_bar', 0.0)
_tmpl_p75_mfe_bar = self.pattern_library.get(best_tid, {}).get('p75_mfe_bar', 0.0)
belief_network.set_active_trade_timescale(_tmpl_avg_mfe_bar, _tmpl_p75_mfe_bar)
```

For worker-bypass (no template), pass zeros:
```python
belief_network.set_active_trade_timescale(0.0, 0.0)
```

**B. In the position management loop**, at the top of `if self.wave_rider.position is not None:`,
after the `res = {'should_exit': False}` default, add:

```python
belief_network.tick_trade_bar()
```

**C. At trade exit** (both normal `res['should_exit']` path and max_hold/emergency_close path),
after clearing pending_oracle, add:

```python
belief_network.clear_active_trade_timescale()
```

**D. Log `avg_mfe_bar` in oracle_trade_records** for diagnostics:

```python
'tmpl_avg_mfe_bar': self.pattern_library.get(best_tid, {}).get('avg_mfe_bar', 0.0),
```

---

## Report Addition (orchestrator.py Phase 4 report)

Add a section after DYNAMIC EXIT QUALITY:

```
TIME-SCALE EXIT SUMMARY:
  time_exhausted exits:  N trades  avg PnL $xxx.xx
  time_tighten exits:    N trades  avg PnL $xxx.xx
  (these are template-time-aware exits vs wave_mature exits)
```

---

## What This Enables

- A depth-5 (1m) template with `avg_mfe_bar=18` (4.5 min) will tighten trail at bar 27,
  urgent-exit at bar ~48 — regardless of wave_maturity
- A depth-2 (1h) template with `avg_mfe_bar=800` (3.3 hrs) stays in the trade
  until bar 1200 before tightening
- Workers now have the SAME information a human trader has: "this pattern
  historically resolves in X minutes — if it hasn't by 1.5X, get out"

## Prerequisites

- Requires `--fresh` to populate `mfe_bar` in oracle_meta for all patterns
  (`mfe_bar` was added to `fractal_discovery_agent.py` on 2026-02-21)
- `avg_mfe_bar` will be 0.0 for templates built before this run
  → graceful fallback: time signals don't fire when avg_mfe_bar == 0

## Rules
- Do NOT modify any other exit logic
- Do NOT change the URGENT_EXIT_CONVICTION_THRESHOLD (currently 1.01 = disabled)
- Time-exhaustion exits use the same `urgent_exit=True` path in wave_rider.py
  so no changes to wave_rider are needed
- Keep changes surgical — only the 3 files above
