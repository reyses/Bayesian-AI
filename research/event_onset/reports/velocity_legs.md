# Legs as the owner defines them: price displacement in a time window

An impulse fires when |close(t) - close(t-T)| >= D. Entry is at the trigger — the first moment it is OBSERVABLE. MAE/MFE/run are measured over the next 300s.

| D (pt) | T (s) | impulses | per day | p50 heat | p95 heat | median run | mean run | P(run>0) |
|---|---|---|---|---|---|---|---|---|
| 10 | 30 | 20,935 | 186.9 | 19.00 | 76.00 | -0.50 | -0.53 | 49.2% |
| 10 | 60 | 24,409 | 217.9 | 17.75 | 71.50 | -0.25 | -0.31 | 49.2% |
| 15 | 30 | 12,277 | 109.6 | 22.75 | 88.75 | -0.75 | -0.83 | 49.1% |
| 15 | 60 | 16,027 | 143.1 | 20.75 | 81.92 | -0.25 | -0.23 | 49.5% |
| 20 | 60 | 10,522 | 94.8 | 23.75 | 92.75 | -0.25 | -0.65 | 49.6% |
| 10 | 15 | 16,818 | 150.2 | 20.50 | 80.75 | -0.75 | -0.73 | 49.0% |
| 20 | 30 | 7,199 | 64.9 | 26.50 | 104.02 | -0.25 | -0.72 | 49.6% |

## What this says — and why it supersedes the zigzag study

Across every (D, T) cell, entering AT the impulse trigger:

- **P(run > 0) = 49.0% - 49.6%.** Seven independent parameterisations, all a
  coin flip. Chasing displacement has no edge, measured on 7,199-24,409
  samples per cell.
- **Heat is enormous**: p50 MAE 17.75-26.50pt, p95 MAE 71.50-104.02pt.
- **Median run is NEGATIVE** in every cell (-0.25 to -0.75).

### This corrects the earlier zigzag study

The zigzag version reported p50 heat of 2.25pt and p99 of 8.25pt. That
population is not the owner's: a zigzag leg is any >= 8pt move that
eventually reverses, including 40-minute drifts, and it is measured from the
leg's own START — a point only identifiable in hindsight.

Measured on the owner's definition (displacement in a time window) and
entered when the impulse is actually OBSERVABLE, the heat is 8x larger and
the edge is zero. The rosy number came from measuring the wrong object at a
hindsight-selected entry.

### The load-bearing consequence

**You cannot chase the impulse.** By the time D points have printed in T
seconds it is a coin flip with 19pt of median heat. Everything profitable on
2024_09_16 came from being positioned BEFORE the displacement — the owner's
short at the recovery peak, 168pt, MAE 0.

Which places the entire problem, again and finally, on ENTRY SELECTION
BEFORE THE MOVE. That is the owner's demonstrated skill (8 correct
directional calls in one session) and the machine's measured null (8
independent tests).
