# Stateful Peak Monitor -- Entry + Exit Sensor

**Priority**: Immediate (next build)
**Date**: 2026-03-18

## Problem

Peak detection is stateless. It fires every bar where P_center/F_momentum
change exceeds thresholds. This causes:

1. **Entry stutter**: peak fires, enter, belief_flip exits at 1 bar,
   peak fires again, re-enter. 1,239 one-bar trades in OOS (-$1,568).
2. **Proxy exits**: giveback/regime_decay/tidal_wave/belief_flip are all
   indirect measures of "the move peaked." Peak detection measures it directly
   but isn't wired to exits.
3. **Clusters**: 3-5 trades in 30 seconds on the same peak, each capturing
   crumbs instead of one trade riding the full reversal.

## Solution

Stateful peak monitor that:
- Knows when a trade is open and in which direction
- Tracks the statistical trajectory (P_center, F_momentum) since entry
- Exits when the peak EXHAUSTS (P_center gives back, F_momentum rebuilds)
- Suppresses re-entry while monitoring the same peak event

## State Machine

```
IDLE
  |-- peak fires --> ENTER trade, save entry state --> TRACKING

TRACKING
  |-- P_center improving --> update peak_pc, HOLD
  |-- P_center flat/declining + F_momentum against --> EXHAUSTED --> EXIT
  |-- SL hit --> EXIT (safety net, always active)
  |-- peak fires AGAINST position --> EXIT (reversal of the reversal)

EXHAUSTED
  |-- EXIT trade --> COOLDOWN

COOLDOWN (N bars)
  |-- prevents re-entry on the same decaying peak event
  |-- cooldown expires --> IDLE
```

## Data Model

```python
@dataclass
class PeakMonitorState:
    active: bool = False
    direction: str = ''           # 'long' or 'short'
    entry_bar: int = 0
    entry_pc: float = 0.0         # P_center at entry
    entry_fm: float = 0.0         # F_momentum at entry
    peak_pc: float = 0.0          # best P_center since entry
    peak_bar: int = 0             # bar where peak_pc occurred
    cooldown_until: int = 0       # bar index when cooldown expires
```

## Update Logic (every bar while in trade)

```python
def update(self, bar_idx, current_pc, current_fm, trade_direction):
    if not self.active:
        return None  # no signal

    # Track statistical peak
    if trade_direction == 'long':
        if current_pc > self.peak_pc:
            self.peak_pc = current_pc
            self.peak_bar = bar_idx
        pc_giveback = (self.peak_pc - current_pc) / max(abs(self.peak_pc), 1e-6)
        fm_against = current_fm < 0  # negative momentum = against long
    else:  # short
        if current_pc < self.peak_pc:
            self.peak_pc = current_pc
            self.peak_bar = bar_idx
        pc_giveback = (current_pc - self.peak_pc) / max(abs(self.peak_pc), 1e-6)
        fm_against = current_fm > 0  # positive momentum = against short

    bars_since_peak = bar_idx - self.peak_bar
    bars_held = bar_idx - self.entry_bar

    # Exit conditions (calibrate thresholds from IS data)
    # 1. Statistical giveback: P_center retreated significantly from peak
    if pc_giveback > PC_GIVEBACK_THRESHOLD and bars_held >= MIN_HOLD_BARS:
        return 'peak_exhausted'

    # 2. Momentum confirmed against: F_momentum rebuilt against trade
    if (fm_against and abs(current_fm) > FM_AGAINST_THRESHOLD
            and bars_held >= MIN_HOLD_BARS):
        return 'peak_momentum_flip'

    # 3. Stale peak: peaked early, hasn't improved in N bars
    if bars_since_peak > STALE_PEAK_BARS and bars_held >= MIN_HOLD_BARS:
        return 'peak_stale'

    return None  # hold
```

## Integration Points

### Entry suppression (kills stutter)
- When peak monitor is in COOLDOWN, suppress new peak entry candidates
- Cooldown = 4-8 bars after exit (calibrate from cluster analysis)
- Non-peak candidates still allowed (different signal source)

### Exit cascade priority
- Peak monitor exit is checked FIRST in the cascade for ALL trades
  (not just peak entries -- any trade benefits from knowing "the move peaked")
- If peak monitor says EXIT, exit. Don't check belief_flip/tidal_wave/etc.
- SL remains always-active safety net (checked before peak monitor)

### Where it lives
- `core/exit_engine.py` or new `core/peak_monitor.py`
- Gets MarketState every bar (already available in BarProcessor loop)
- PositionState carries `peak_monitor: PeakMonitorState`

## Thresholds to Calibrate

| Parameter | Description | Calibrate from |
|-----------|-------------|---------------|
| PC_GIVEBACK_THRESHOLD | P_center retreat % that triggers exit | IS peak trade P_center trajectories |
| FM_AGAINST_THRESHOLD | F_momentum magnitude confirming reversal | IS momentum at exit for winners vs losers |
| MIN_HOLD_BARS | Minimum bars before peak exit can fire | IS hold duration vs PnL (sweet spot = 3-4 bars) |
| STALE_PEAK_BARS | Bars since peak_pc improved | IS time-to-MFE distribution |
| COOLDOWN_BARS | Re-entry suppression after exit | IS cluster gap analysis |

## Calibration Research

Run on IS data:
1. For each peak trade, record P_center and F_momentum every bar
2. At the trade's actual MFE bar, what was pc_giveback? (This is the "perfect" threshold)
3. At the actual exit bar, how much further did price go? (This is the regret)
4. Winners vs losers: what P_center trajectory separates them?

Save as `tools/peak_exit_calibration.py` for reuse.

## Expected Impact

- Kill 1,239 one-bar stutter trades (OOS: +$1,568 saved)
- Convert clusters of 3-5 trades into single 6-10 bar rides ($7/trade)
- Replace proxy exits (belief_flip, tidal_wave, regime_decay) with direct measurement
- Works for ALL trades, not just peak entries

## Risk

- Over-holding: if thresholds too loose, peak monitor holds through the reversal
- Mitigation: SL is always active, survival_stop is always active
- Calibration is key -- research first, then wire in
