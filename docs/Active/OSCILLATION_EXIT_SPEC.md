# Oscillation Exit Spec — Ride the Wave, Exit When It Dies

## Discovery
100% of long-hold optimal trades (18+ min) cross z=0 at least once.
95% cross zero MULTIPLE times — they oscillate. The current physics exit
(|z| < 0.5) kills these trades on the first cross, capturing 0.2% of the move.

The optimal exit isn't a z position. It's oscillation energy decay.

## What's Actually Happening

```
Entry: z=-2.5 (NMP triggers)
Bar 60:  z=-0.3  (physics would exit here — $25 profit)
Bar 120: z=+1.5  (price overshot the mean — now on the other side)
Bar 180: z=-0.8  (came back — oscillation)  
Bar 240: z=+0.4  (smaller swing — energy decaying)
Bar 300: z=-0.3  (even smaller — oscillation dying)
Bar 336: z=-2.2  (OPTIMAL EXIT — one final push in our direction — $116)
```

The trade isn't done at z=0. Price oscillates around the mean with
decaying amplitude. The optimal exit catches the last big swing in the
right direction before the oscillation fully damps out.

## Oscillation Metrics

From the EDA at optimal exit:
- **variance_ratio**: 0.42 (down from 0.46 at entry — energy leaving)
- **reversion_prob**: 0.95 (very high — price confined near mean)
- **wick_ratio**: 0.56 (high — indecision candles, no momentum)
- **p_at_center**: 0.51 (near the mean but not through it)
- **velocity**: near zero (momentum exhausted)
- **5m_z_se**: near zero (5m also settled)

## Proposed Exit: Oscillation Energy Tracker

### Core Idea
Track the oscillation amplitude over time. When amplitude decays below
a threshold, the oscillation is dying — exit on the next favorable swing.

### Implementation

**Oscillation State** (tracked per trade):
```python
class OscillationTracker:
    z_peaks = []        # local z maxima during trade
    z_troughs = []      # local z minima during trade  
    zero_crossings = 0  # count of z sign changes
    last_z_sign = 0     # +1 or -1
    amplitude_history = []  # peak-to-trough amplitudes
```

**At each bar:**
1. Record z_se
2. Detect z sign change → increment zero_crossings
3. Detect local peaks/troughs (z changes direction)
4. Compute current amplitude: latest peak - latest trough

**Exit conditions:**
1. `zero_crossings >= 2` (at least one full oscillation cycle)
2. `current_amplitude < entry_amplitude * DECAY_RATIO` (energy decayed)
3. `z_se` is favorable (in our trade direction for final capture)

```
DECAY_RATIO = 0.40  # exit when amplitude is 40% of entry amplitude
```

**Why not just hold forever?**
- Oscillation can restart (new energy from higher TF move)
- variance_ratio rising = new trend forming = oscillation over
- Hard stop still applies ($-150)

### Exit Rule (readable, not CNN):
```python
# Oscillation energy exit
if (tracker.zero_crossings >= 2 and
    tracker.current_amplitude < tracker.entry_amplitude * DECAY_RATIO and
    z_favorable):  # z in our direction
    exit('oscillation_decay')

# Regime change override
if variance_ratio > 1.0:
    exit('regime_shift')
```

## New ExNMP Seed: Oscillation Rider

This is a distinct entry setup that NMP currently catches but exits too early.

### Entry Conditions (same as NMP):
- |z_se| > 2.0 at 1m
- vr < 1.0

### What makes it an oscillation setup (from CNN entry patterns):
- Low 1m_bar_range (tight range, not trending)
- reversion_prob > 0.70 (mean-reverting regime confirmed)  
- 5m_z_se NOT extreme (higher TFs not pushing)
- 1h_z_se near zero (no hourly directional pressure)

### Exit (oscillation-specific):
- NOT |z| < 0.5 (that kills the trade)
- Track zero crossings + amplitude decay
- Exit when amplitude < 40% of entry amplitude + z favorable
- Hard stop -$150 unchanged

### Expected Impact
- Current: $35/day IS (physics exit, 0.2% capture)
- Long trades avg optimal: $200/trade, ~79 bars hold
- If we capture even 30% of the optimal on these: +$60/trade
- 226 long trades → potential +$13,560 IS
- That's $49/day additional on top of current

## Integration Into Pipeline

### Option A: Separate ExNMP
Add as a new entry type alongside CASCADE/KILL_SHOT/BASE_NMP:
- **OSCILLATION**: same NMP entry, oscillation exit rules
- Classified at entry by CNN entry patterns that predict oscillation behavior
- Separate exit physics, separate tracking

### Option B: Modify BASE_NMP Exit
Replace the BASE_NMP physics exit with oscillation-aware exit:
- If zero_crossings == 0: use current physics exit (z < 0.5)
- If zero_crossings >= 1: switch to oscillation mode (amplitude decay exit)
- Simpler implementation, no new tier

### Recommendation: Option B
Don't add complexity. Modify BASE_NMP exit to be oscillation-aware.
The oscillation tracker activates after the first zero crossing.
Before the first crossing, current physics exits still apply.

## Verification Plan
1. Implement OscillationTracker in BlendedEngine
2. Run forward pass IS with oscillation exit vs current physics exit
3. Compare $/day, capture rate, hold time distribution
4. If better: integrate into live engine
5. If not: revert, try CNN exit instead

## Risk Assessment
- **Risk**: holding too long on a trade that won't oscillate
  - Mitigated by: hard stop ($-150), regime shift exit (vr > 1.0)
- **Risk**: decay threshold too aggressive (exits too early)
  - Tunable: DECAY_RATIO is a single number, easy to sweep
- **Risk**: overfitting to IS oscillation patterns
  - Validation: must show on OOS before live deployment
