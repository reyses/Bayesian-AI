# GBM-armed policy — 112 days: **significant LOSER**

The owner asked for a fresh day with the GBM arming. One blind-picked day
(2025_05_08, seeded, no peeking) returned **+11.57pt over 12 trades** in a
10-minute window. That number is noise, and here is the proof.

Identical pre-registered rules across every val-window day:

| | |
|---|---|
| days | 112 |
| trades | 1,344 |
| **mean** | **−8.81 pt/day, day-bootstrap 95% CI [−13.13, −4.73]** |
| median | −9.18 pt/day |
| winning days | 35% |
| net per trade | −0.734 pt |
| **gross per trade (friction added back)** | **+0.156 pt** |

The CI excludes zero on the losing side: this is not "no edge", it is a
**significant loser**. The blind day ranked in the top handful of 112.

## What the numbers say precisely

Gross +0.156 pt/trade against 0.89 pt of round-trip friction. The entries are
very slightly better than a coin flip and nowhere near paying for the spread.
That is exactly the prediction from the geometry control
(`bayes_tables/reports/tables_v0.md`): a better onset detector tells you an
event is FORMING, and the event resolves by barrier distance, so the
detector buys latency and never direction.

## What this does NOT say

The protection stack is not indicted. It did its job on every trade — losses
capped near 1-2pt by the 50% lock and the entry-touch halt, one +9.96 winner
allowed to run. A risk machine cannot rescue an entry rule with no edge; it
was never supposed to.

## The lesson worth keeping

A single day is worth nothing, and this run is the cleanest demonstration in
the program: the same rules, same model, same code — **+11.57 on the day I
happened to draw, −8.81/day across 112.** Any result reported from one
session is a coin toss dressed as evidence.
