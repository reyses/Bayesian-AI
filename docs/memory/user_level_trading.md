---
name: User's manual level-drawing experience
description: User hand-drew support/resistance levels on MES/MNQ charts that held for weeks. This is the ground truth the Level Detector must replicate.
type: user
---

## Manual Level Trading Background

The user has direct experience manually trading MNQ/MES futures using hand-drawn
support/resistance levels. Key facts:

1. **Set levels in Feb 2026** on MNQ that held through March — levels persist
2. **Hand-drew horizontal lines** on TradingView /MES 4h chart (not auto-generated)
   — these are the reference image at `examples/` showing what the system should produce
3. **Levels come from experience**: prior swing H/L, price stall/rejection zones,
   session references, volume concentration, repetition across timeframes
4. **Fibonacci as context framework**: Fib retracements put historical swings into
   context, but the real levels are where price repeatedly interacts — not just
   where the math says they should be
5. **Levels accumulate incrementally**: built day-by-day as new swings complete,
   old levels don't disappear — they stay until definitively broken
6. **Two distributions observed in OOS**: Jan high cluster (~25,300-25,500) and
   late Feb-Mar low cluster (~24,700-25,000) with crash between them

### What the system must replicate
- Detect key price levels automatically from historical price action
- Build levels incrementally (no lookahead)
- Persist levels across sessions until broken
- Find confluence (multiple sources agreeing on same price)
- Use levels as context gate: "am I entering near a key level? Which side?"
