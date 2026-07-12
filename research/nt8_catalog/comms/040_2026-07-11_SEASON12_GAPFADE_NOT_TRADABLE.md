# SEASON-12 gap-fade: looks profitable, is NOT tradable (Claude verification)
**Doc:** 040 · **Date:** 2026-07-11 · **Author:** Claude (executor) · **Status:** FINAL

Moises flagged the gap-fill distributions as "funny but profitable." Verified as a
REAL trade (enter at open against the gap, target = prior close, stop −S, worst-case
ordering, day-block CIs, both pessimistic and optimistic unfilled-exit bounds):

| stop | 2024 EV (pess..optim) | 2025 EV (pess..optim) | stopped |
|---|---|---|---|
| −15 | −2.3..−1.6 (ns) | +2.6..+2.9 (ns) | ~70% |
| −25 | **−6.6..−5.4 (SIG NEG 2024)** | −0.1..+0.7 (ns) | ~62% |
| −35 | **−7.2..−5.4 (SIG NEG 2024)** | −1.8..−0.5 (ns) | ~55% |

## Why the picture lies
Median |gap| = 74.5 pts. The histogram's "profit" is the FULL-GAP capture (+74 median
when filled, 60% fill rate) — but the fill path runs 15-35+ pts adverse first in
~55-70% of events. Any stop tight enough to survive the losers kills most of the
winners before they fill; any stop wide enough to let fills happen bleeds more than
the fills pay. Classic unrealizable-peak artifact: the response is REAL (gaps do fill
60%), the TRADE is not (the path to the fill is unaffordable).

Contrast with the level-continuation family (doc 039): PIVOT-16-flip and ROUND-05
have SMALL adverse paths relative to target — that's why they survive realization
and SEASON-12 doesn't. The screen for "looks profitable" candidates is now clear:
**check MAE-vs-target geometry before believing any distribution.**

Standing scoreboard (2026-07-11, realizable, both-year day-block CIs):
- ✅ PIVOT-16 FLIPPED: +8/+9 pts/event, both years sig, stop-robust.
- ✅ ROUND-05 (±20): +5.4/+6.6 pts/event, both years sig, worst-case ordering.
- ❌ SEASON-12 gap-fade: flat-to-negative as a trade.
- ❌ ATR-09 INVERT: forward-frequency real, tail entry-invisible, year-swap unstable.
- Pending same treatment: VWAP-03, MACD-07.
