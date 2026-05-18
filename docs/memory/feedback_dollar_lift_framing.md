---
name: $/day lift framing
description: How to frame proposed $/day improvements -- against the honest floor, not the inflated headline, and weight tail-risk reduction.
type: feedback
originSessionId: f4f2fb74-6511-49b0-a189-e2611540bf39
---
When proposing a feature that adds $X/day, the user expects the impact
framed against the HONEST FLOOR (the realistic deployment number after
caveats), not the optimistic headline.

**Why**: The deliverable's headline OOS PnL was $927/day, but after the B7
distribution-shift caveat the honest floor is $600-700/day. I described a
proposed +$200-500/day day-regime sizer as "$200-500/day, not
transformational" -- anchoring on $927. User correctly pushed back: on the
honest $600-700 floor that's +30-70% revenue, ~$50-125K/year. That IS
transformational by any reasonable definition.

**How to apply**:
1. State the HONEST FLOOR first, then compute the lift as a percentage of
   that floor -- not the inflated headline.
2. Translate $/day to $/year (×250) to make the scale visible.
3. Weight tail-risk reduction separately from mean lift. Filtering 3 of 5
   negative OOS days isn't just $300/day arithmetic -- it's eliminating
   the drawdown days that cause humans to pull the plug, plus it lets the
   strategy size UP confidently on predicted-good days.
4. Don't dismiss "incremental" lifts when the strategy is real-money and
   tail-risk-sensitive. Trading systems sustain on sequence-of-returns
   risk reduction, not just mean lift.
5. Don't lead with skepticism when an improvement compounds with existing
   rails (per-leg × per-hour × per-day sizing -- multiplicative). Each
   layer's lift compounds with the others; the day layer isn't an add, it's
   a scalar.
