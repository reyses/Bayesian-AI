# After a chop or a fakeout — does the preceding trend continue? (v2)

Leg-DECOUPLED rebuild. Trend = sign of the regression slope of 5s closes over a 90-min window; the zigzag marks the event only. `continues` = the slope after the event matches the slope before it.

**VALIDATION GATE: PASS** — IS chop K-series parity gap 0pp; no parity artifact.

**Base rate** — P(90-min trend direction persists across a leg boundary): IS 50% [49%, 51%] (n=15,977); OOS 52% [50%, 54%] (n=2,657).

## After a CHOP (K consecutive low-range legs) — P(preceding trend continues)

| K | n(IS) | IS P(continues) [CI] | n(OOS) | OOS P(continues) [CI] |
|--:|--:|---|--:|---|
| 1 | 5,105 | 52% [50%, 53%] | 385 | 56% [51%, 61%] |
| 2 | 2,714 | 52% [50%, 54%] | 116 | 50% [41%, 59%] |
| 3 | 1,657 | 54% [52%, 57%] | 45 | 53% [38%, 69%] |
| 4 | 1,062 | 55% [52%, 58%] | 19 | 74% [53%, 89%] |
| 5 | 700 | 56% [52%, 59%] | 9 | 89% [67%, 100%] |

## After a FAKEOUT (a notable counter-trend leg) — P(preceding trend continues)

- IS: 53% [51%, 54%] (n=5,621)
- OOS: 55% [52%, 58%] (n=1,153)

## Read

- Gate passed — no parity artifact. Compare each P(continues) to the base rate: ABOVE = the event is followed by trend continuation more than usual (the mechanical counter-trend entry is wrong, a direction gate has something to catch); AT base = no directional information; BELOW = the event predicts a reversal.
- Trust a cell only where IS and OOS agree.