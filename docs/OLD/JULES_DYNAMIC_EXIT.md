# JULES TASK: Dynamic TP/SL Adjustment via Worker Conviction

## Context
The system currently sets TP and SL at trade entry and never adjusts them mid-trade.
The belief network workers run every bar during the trade but their signals are ignored
after entry. Two problems result:

1. **Exits too early** — trailing stop fires before the wave completes
2. **Exits too late** — trail follows price into a reversal; workers already signalled EXIT

## Goal
After a trade is opened, each bar:
1. Poll `belief_network.get_belief()` → get conviction + direction
2. If conviction drops below threshold OR direction flips → tighten the trail stop
3. If conviction is HIGH and direction aligns → loosen trail stop (let the wave ride)
4. If the 5m worker's `wave_maturity` is high (>0.7) → expect reversal → tighten TP

---

## New Method: `TimeframeBeliefNetwork.get_exit_signal(side)`

**File**: `training/timeframe_belief_network.py`

```python
def get_exit_signal(self, side: str) -> dict:
    """
    Called every bar while a position is open.
    Returns a dict with exit adjustment recommendations.

    side: 'long' or 'short' — the current position direction.

    Returns:
        {
          'tighten_trail': bool,   # shrink trail stop distance
          'widen_trail':   bool,   # grow trail stop (conviction is high)
          'urgent_exit':   bool,   # exit NOW (direction flipped, high conviction)
          'conviction':    float,  # current path conviction
          'wave_maturity': float,  # 5m decision TF wave maturity
          'reason':        str,    # human-readable reason
        }
    """
    belief = self.get_belief()
    if belief is None:
        return {'tighten_trail': False, 'widen_trail': False,
                'urgent_exit': False, 'conviction': 0.0,
                'wave_maturity': 0.0, 'reason': 'no_belief'}

    trade_long = (side == 'long')
    belief_long = (belief.direction == 'LONG')
    direction_aligned = (trade_long == belief_long)
    wave_mature = belief.decision_wave_maturity  # 5m worker only

    # Urgent exit: high conviction in the OPPOSITE direction
    urgent = belief.is_confident and not direction_aligned and belief.conviction > 0.70

    # Tighten: conviction is low OR wave is mature (approaching reversal zone)
    tighten = (not belief.is_confident) or (wave_mature > 0.65)

    # Widen: strong conviction aligned with trade direction, wave is fresh
    widen = belief.is_confident and direction_aligned and wave_mature < 0.30

    reason = ('urgent_flip' if urgent else
              'low_conviction' if not belief.is_confident else
              'wave_mature' if wave_mature > 0.65 else
              'aligned_fresh' if widen else 'neutral')

    return {
        'tighten_trail': tighten and not urgent,
        'widen_trail':   widen,
        'urgent_exit':   urgent,
        'conviction':    belief.conviction,
        'wave_maturity': wave_mature,
        'reason':        reason,
    }
```

---

## Changes to `WaveRider.update_trail()` in `training/wave_rider.py`

Add an optional `exit_signal` parameter:

```python
def update_trail(self, price: float, bar_ts, ts_raw: float,
                 exit_signal: dict = None) -> dict:
```

Inside `update_trail()`, before checking the existing trail logic:

```python
if exit_signal:
    if exit_signal.get('urgent_exit'):
        # Immediate market exit
        return {'should_exit': True, 'exit_price': price,
                'pnl': ..., 'exit_reason': 'belief_flip'}

    if exit_signal.get('tighten_trail') and self.trail_ticks is not None:
        # Reduce trail by 30% (min: 2 ticks)
        _new_trail = max(2, int(self.trail_ticks * 0.70))
        self.trail_ticks = _new_trail

    if exit_signal.get('widen_trail') and self.trail_ticks is not None:
        # Increase trail by 20% (max: original_trail * 2.0)
        _max_trail = self._original_trail_ticks * 2
        self.trail_ticks = min(_max_trail, int(self.trail_ticks * 1.20))
```

`WaveRider` needs a `_original_trail_ticks` field set at trade open.

---

## Changes to `training/orchestrator.py`

In the per-bar position-management block (around line 677, where `wave_rider.update_trail()` is called):

```python
# Get exit signal from belief network every bar
_exit_sig = belief_network.get_exit_signal(active_side)
res = self.wave_rider.update_trail(price, None, ts_raw, exit_signal=_exit_sig)
```

Add the exit signal fields to the oracle trade record so we can analyse:
- `exit_belief_conviction` — conviction at exit bar
- `exit_wave_maturity` — wave maturity at exit bar
- `exit_reason` — now includes 'belief_flip', 'belief_mature', etc.

---

## Logging for Analysis

In the oracle trade record (both intra-day and EOD exits), add:
```python
'exit_conviction':    _exit_sig.get('conviction', 0.0),
'exit_wave_maturity': _exit_sig.get('wave_maturity', 0.0),
'exit_signal_reason': _exit_sig.get('reason', ''),
```

Report section to add after DIRECTION FLIPS:
```
DYNAMIC EXIT QUALITY:
  Belief-flip exits:  N trades  ->  avg PnL $X  (these would have been losers)
  Trail-tightened:    N trades  ->  avg PnL $Y
  Trail-widened:      N trades  ->  avg PnL $Z  (let winners run longer)
  Standard trail:     N trades  ->  avg PnL $W
```

---

## Files to Modify
1. `training/timeframe_belief_network.py` — add `get_exit_signal()` method (~35 lines)
2. `training/wave_rider.py` — update `update_trail()` to accept exit signal (~25 lines)
3. `training/orchestrator.py` — wire belief_network to update_trail + log new fields (~30 lines)

## Test Plan
1. Run OOS with new code, check that `exit_reason` now has 'belief_flip' entries
2. Verify `urgent_exit` never fires on a profitable trade mid-run (check exit_wave_maturity)
3. Compare avg PnL of tightened vs standard trail exits
4. Check that 'Reversed' exit bucket (87 trades, -$10,564) decreases — these should become
   'belief_flip' exits with smaller loss or even a small profit from early exit

## Success Criteria
- `Reversed` exit bucket shrinks from 87 trades → target < 50
- Overall PnL increases by 10%+ vs current $16,317
- No new 'wrong direction' exits (urgent_exit only fires when conviction > 0.70 opposite)
