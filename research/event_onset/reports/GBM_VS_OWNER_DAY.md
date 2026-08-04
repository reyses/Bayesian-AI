# Would the GBM have helped on the day we both traded? — **NO, and the day says why**

Owner's question (2026-08-04): *"it's risk management if we are in the right
direction? Let's backtest what the GBM would have saved/helped and what
failed"* — on 2024_09_16, the session we ran together.

Clean test: 2024_09_16 is excluded from every table, every fit, and the
event library. **The GBM has never seen this day.**

15 of 20 trades recovered from the decision log with a 1s timestamp (5 could
not be price-matched; the subset sums +3.15pt, not the day's +50.45 — read
the per-trade结构, not the total).

## The GBM's onset probabilities at our entries

| head | winners | losers | gap | permutation p (20k) |
|---|---|---|---|---|
| fakeout_poke | 0.500 | 0.418 | +0.082 | 0.176 |
| leg_descent | 0.237 | 0.340 | −0.103 | 0.513 |
| ultra_chop | 0.076 | 0.168 | −0.093 | 0.573 |
| stall | 0.504 | 0.480 | +0.024 | 0.766 |

**Nothing separates.** Every gap is noise at N=15, and the fakeout gap points
the wrong way for a veto (higher onset on the trades that WON). Any filter
built on these numbers would be curve-fitting 15 points.

Worked example of why: a "skip when chop >= 0.30" rule would have vetoed one
−10.89 loser AND one +14.11 winner — net **−3.22 worse**.

## What actually decided the day

| side | n | total | mean | winners |
|---|---|---|---|---|
| **short** | 9 | **+45.24** | +5.03 | 7/9 |
| **long** | 6 | **−42.09** | −7.02 | 2/6 |

The day was a descent. Being short paid +45; being long cost −42. **4 of 6
longs lost exactly −10.89 — the stop, every time.**

## The answer to the question as asked

**Yes — it is risk management CONDITIONAL on direction, and that is a
severe conditional.** On this day:
- Direction contributed a ±87pt swing between the two sides.
- The protection stack contributed by making every wrong-side trade cost
  exactly one stop instead of an open-ended loss — 4 identical −10.89s
  rather than four rides to the bottom.
- The GBM contributed **nothing measurable**: it does not know which side to
  be on, and it was never built to.

That is the same conclusion as the geometry control and the 112-day backtest,
now visible in a single session we both watched: **the machine bounds what
being wrong costs; it does not tell you which way to face.** The owner's
descent thesis did that, and it was right.
