---
name: R3 Pre-Entry Pace Filter
description: Research line — use pace/momentum estimate at entry time to filter noise trades before committing capital
type: project
---

## R3: Pre-Entry Pace Filter

**Discovered**: 2026-03-11, from trade anatomy analysis

### Problem
43% of IS trades are counter-trend scalps (43.0%) or noise (10.8%). The system profits
on many of them because giveback catches micro-spikes, but the entries were wrong —
we're trading statistical noise and getting bailed out by fast exits.

Example: TID=55 LONG trade had oracle_mfe=3.5 ticks ($0.875 real move), but the system
saw a 44-tick spike, captured $13 via giveback, and looked like a winner. In truth the
entry should never have happened — the oracle saw no real move.

### Insight
Pace (tick_progress / time_progress) currently only runs during a position. But the
same concept could gate entries:

- **Pre-entry momentum**: Is current price movement consistent with the template's
  expected MFE trajectory? A template that expects +50 ticks over 8 minutes should
  show early momentum in the right direction at entry time.
- **Velocity alignment**: If `particle_velocity` is flat or adverse at entry, the
  template match is probably noise — the statistical pattern matched but the market
  isn't actually moving.
- **Noise spike detection**: A 44-tick spike in 30 seconds on a template that expects
  8 minutes to peak is a noise spike, not the pattern developing. The pace ratio
  (tick_progress >> time_progress) could flag this as "too fast to be real."

### Implementation Sketch
At entry time, before committing:
```python
# Already available at entry:
velocity = state.particle_velocity  # dp/dt
F_net = state.F_net                 # d²p/dt²
template_mfe = lib_entry['p75_mfe_ticks']
template_resolve = lib_entry['avg_mfe_bar'] * 4  # in 15s bars

# Expected pace: template_mfe / template_resolve = ticks/bar expected
expected_pace = template_mfe / max(1, template_resolve)

# Current pace proxy: velocity (ticks/bar already)
# If velocity << expected_pace and no acceleration, skip
if abs(velocity) < expected_pace * 0.3 and sign(F_net) != sign(velocity):
    skip("pre-entry pace too low")
```

### Scope
- Gate in `execution_engine.py` direction cascade or as new gate after conviction
- Needs: velocity, F_net (already in MarketState), template MFE stats (already in lib_entry)
- Risk: might filter real slow-developing trades that eventually peak
- Validate: compare oracle_mfe distribution of filtered vs passed trades

### Priority
After R1 (TF-bucketed clustering) and 4x timeframe fix validation. This is an
entry-side improvement — separate from exit tuning.
