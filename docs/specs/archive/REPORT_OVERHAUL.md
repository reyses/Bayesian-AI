# Report Overhaul — Actionable Data Only

## Problem
Current reports are 700+ lines of text with 90% noise. We run the same
bash analyses every time to extract the data we actually need.

## New Report Structure (single page, ~80 lines)

### SCORECARD (5 lines — the ONLY numbers that matter)
```
SCORECARD: OOS Feb-Mar 2026 (37 days, cat brain, proportional gate)
  PnL: $16,409  |  Trades: 3,957  |  WR: 65.4%  |  PF: 1.98
  $/trade: $4.15  |  $/day: $443  |  Max DD: $1,673 (7.7%)
  Win days: 33/37 (89%)  |  Worst day: -$1,048 (Feb 9)
  Without Feb 9: $17,457 PF=2.17
```

### WHAT'S WORKING (ranked by PnL contribution)
```
EXIT WINNERS (PF > 1.5):
  stop_loss        704 trades  PF=5.17  +$11,345  $16/trade  ← SL ratchet in trends
  survival_stop    169 trades  PF=inf   +$2,812   $17/trade  ← never loses
  regime_decay    1271 trades  PF=1.49  +$2,474   $1.95/trade

BEST HOURS: 14:00 ($8.92/tr), 15:00 ($6.11), 19:00 ($4.10)
SWEET SPOT: 2-5 min holds (PF 3.18, $8.09/trade, 74% WR)
```

### WHAT'S BROKEN (ranked by PnL damage)
```
EXIT LOSERS (PF < 1.0):
  peak_giveback   1135 trades  PF=0.64  -$2,009  -$1.77/trade  ← FIX: vol exit override
  tidal_wave       204 trades  PF=0.33  -$1,245  -$6.10/trade  ← FIX: suppress first 4 bars
  belief_flip       16 trades  PF=0.71  -$61     -$3.81/trade  ← KILL: remove entirely

OVER-HOLDING: 5m+ trades PF=0.16 (-$18/trade)  ← FIX: hard max 5min
WORST HOURS: 07:00 (-$0.94/tr), 03:00 (-$0.56)
```

### FIXABLE PROFIT (what we'd gain)
```
  Fix giveback PF 0.64 -> 1.5:    +$2,000 estimated
  Fix tidal_wave PF 0.33 -> 1.0:  +$1,245 estimated
  Hard max hold 5min:              +$500 estimated
  Remove belief_flip:              +$61
  TOTAL AVAILABLE:                 +$3,806 -> OOS from $16.4K to $20.2K
```

### DIRECTION SPLIT
```
  LONG:  3,492 (88%) WR=65.8% PF=1.99  ← dominant
  SHORT:   465 (12%) WR=62.2% PF=1.89  ← good quality when it fires
  → Peak-implied direction would increase SHORT to ~30%
```

### SENSOR GATE SUMMARY
```
  Peaks detected:    X
  Blocked (total):   X (XX:1 ratio)
    1m_sensor:       X
    cat_regime:      X
    adx_chop:        X
    fake_peak:       X (flagged, not blocked)
  → Gate allowing X% of detected peaks through
```

### ADAPTIVE STATE (what the system learned this run)
```
  Vol drop threshold: 0.50 -> 0.XX (adapted from 30 giveback exits)
  Cat regime distribution: TRENDING X%, CHOPPY X%, TRANSITION X%, EXHAUST X%
  Brain direction bias: LONG X%, SHORT X% (tid=-100)
```

### COMPARISON vs PRIOR RUN
```
  Metric          This Run    Prior      Delta
  PnL             $16,409     $22,812    -$6,403 (-28%)
  Trades          3,957       4,810      -853 (-18%)
  WR              65.4%       66.3%      -0.9%
  PF              1.98        2.0        -0.02
  → Gate strictness increased, fewer trades, similar quality
```

## What Gets REMOVED from Reports
- Daily session ledger (310 lines of per-day data — goes to CSV only)
- Worker agreement table (never actionable)
- Direction flips between entry/exit (noise)
- E[PnL] analysis (brain is dead for peak trades)
- Depth weights (all depth 8)
- Duplicate sections (regret appears twice)
- Oracle metadata (always 0 for peak trades)

## Implementation
1. New function: `_write_actionable_report()` in trainer.py
2. Old report kept as `_write_full_report()` for debugging (--verbose flag)
3. Default: actionable report only (~80 lines)
4. All raw data still saved to CSVs (trade log, signal log, etc.)
