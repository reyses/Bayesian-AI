# Legs measured from an ANCHOR (confirmed swing pivot)

D=10pt within T=60s of the anchor; heat/run over 300s; 112 days.

| regime | is_leg | n | p50 heat | p95 heat | mean MFE | mean run | P(run>0) [day-CI] |
|---|---|---|---|---|---|---|---|
| AT_ANCHOR | True | 47,606 | 12.25 | 94.00 | 47.72 | +11.64 | 63.4% [61.9%,65.1%] |
| AT_ANCHOR | False | 8,368 | 12.00 | 74.16 | 20.70 | -1.31 | 56.0% [53.4%,58.7%] |
| ON_CONFIRM | True | 47,606 | 22.50 | 105.75 | 38.11 | +1.57 | 51.6% [51.3%,51.9%] |
| ON_CONFIRM | False | 8,368 | 21.50 | 84.50 | 12.78 | -10.01 | 37.3% [36.0%,38.6%] |

## The result

| entry | P(run>0) | mean run | mean MFE | p50 heat |
|---|---|---|---|---|
| **AT_ANCHOR** (at the pivot) | **63.4%** | **+11.64** | 47.72 | 12.25 |
| ON_CONFIRM (8s later) | 51.6% | +1.57 | 38.11 | 22.50 |

n = 47,606 anchored legs, 112 days. **Median confirmation lag: 8 seconds.**

## What it means

The entire edge lives in the ~8 seconds between the anchor and the moment
the move is confirmed. Enter at the anchor: 63.4% and +11.64 per trade.
Enter when it is provable: 51.6% and +1.57 — gone. Heat nearly doubles too
(12.25 -> 22.50 at p50).

This also explains why the sliding-window study came out a coin flip (49%):
with no anchor, every second is its own reference, so the population mixes
anchor-entries with mid-move entries and averages the edge away. The owner
was right — the failure was the missing anchor, not the definition of a leg.

## The honest limit

AT_ANCHOR entry uses the pivot price, and a pivot is only *confirmed* 8s
later. So 63.4% is not free money — it is the value of identifying the anchor
IN REAL TIME. That is precisely what the owner's method claims to do (levels,
regions, oscillation edges), and precisely what no model in this program has
managed.

So the prize is now measured: **+10pt per trade sits between calling the
anchor and waiting for proof.** That is the thing worth building, and the
corpus of his anchor calls is the only training signal we have for it.
