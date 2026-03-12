---
name: Headroom & Nesting Rules (user's manual trading framework)
description: User's proven cross-timeframe gating rules from manual MES/MNQ trading — micro wave must fit inside macro container. This is the mathematical foundation for the Level Detector and a new execution gate.
type: user
---

## The Headroom Calculation

User developed this from manual trading experience. Prevents entering micro
patterns when the macro timeframe has no room for expansion.

### Core Math

- **Micro Target**: T_micro = μ_micro + 3σ_micro  (top of the micro wave)
- **Macro Ceiling**: C_macro = μ_macro + 2σ_macro  (macro resistance)
- **Trade Condition**: T_micro < C_macro  ("the wave fits in the ocean")

### Traffic Light

| State    | Macro Position  | Micro Position     | Action    |
|----------|----------------|--------------------|-----------|
| SAFE     | At mean (0σ)   | At breakout (2σ)   | EXECUTE   |
| WARNING  | At 1.5σ        | At mean (0σ)       | CAUTION   |
| BLOCKED  | At 3σ          | Anywhere           | FORBIDDEN |

### The Nesting Rule (Go Signal)

```
LONG = (ADX_micro > 25) AND (|Z_macro| < 1.0)
```

"Ride the micro wave ONLY IF the macro ocean is calm."

### Origin Story

User experienced the "150 to -50" disaster: micro screamed "Buy" (strong ADX,
good pattern), but macro was already at 3σ. The micro spike captured a brief
profit, then the macro reversal crushed it. The nesting rule would have flagged
the trade as BLOCKED instantly.

### Mapping to Our Architecture

Everything needed is already computed:
- `z_score` per TF worker → micro/macro z available
- `self_adx` in 16D feature vector → ADX_micro ready
- `parent_z` in feature vector → Z_macro ready
- TBN workers at 1h/30m/15m/5m/3m/1m → full nesting hierarchy

What's MISSING:
1. No explicit "headroom" check at entry time
2. No gate that says "macro at 3σ = BLOCKED regardless of micro"
3. Band confluence (Priority 4) aggregates but doesn't enforce nesting
4. The micro target projection (μ + 3σ) isn't compared against macro ceiling

### Implementation Path

New gate in ExecutionEngine or as a pre-entry check:
```python
# At entry time:
macro_z = abs(parent_worker.z_score)  # e.g., 1h worker
micro_adx = candidate.self_adx

# Nesting rule
if macro_z >= 2.0:
    skip("BLOCKED: macro at extreme, no headroom")
elif macro_z >= 1.5 and micro_adx < 30:
    skip("WARNING: limited headroom, need strong micro")
# else: SAFE — proceed
```
