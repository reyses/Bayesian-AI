# State-conditional test — the strategy inside its explicit state

Owner: *"the strategy is for an explicit state, and sonnet was trying to generalize it."* Correct: every prior sweep measured the unconditional EV. This encodes his live protocol — **observe the first n oscillations of a stable region pair, then harvest** — with his two-stage 80/70 frozen-MFE exit, honest fills, and the one-per-region-until-opposite-touch rule (NOT vacuous for fixed regions, unlike sigma bands).

Region stability tol 25% of range, entry proximity 15%, state dies 50% beyond a region, stop max(8pt, 25% of range), friction 0.89pt. Sessions: **603**. Excluded: 2024_09_16.

**The test is the n_osc gradient.** If observing the state first adds value, EV must rise from 1 → 2 → 3 observed oscillations.

| min range | observed osc | trades | /session | mean net | 95% CI | $/trade |
|---|---|---|---|---|---|---|
| 20pt | 1 | 9,887 | 16.40 | `-0.83` | `[-1.01, -0.65]` | `$-1.66` |
| 20pt | 2 | 510 | 0.85 | `-1.15` | `[-1.79, -0.50]` | `$-2.31` |
| 20pt | 3 | 16 | 0.03 | `-2.86` | `[-5.34, -0.16]` | `$-5.72` |
| | | | | | | |
| 35pt | 1 | 3,427 | 5.68 | `-0.89` | `[-1.21, -0.56]` | `$-1.77` |
| 35pt | 2 | 82 | 0.14 | `-1.22` | `[-3.53, +1.26]` | `$-2.43` |
| 35pt | 3 | 0 | — | too few | — | — |
| | | | | | | |
| 50pt | 1 | 1,738 | 2.88 | `-0.73` | `[-1.27, -0.17]` | `$-1.46` |
| 50pt | 2 | 42 | 0.07 | `-2.52` | `[-5.47, +0.54]` | `$-5.04` |
| 50pt | 3 | 0 | — | too few | — | — |
| | | | | | | |

**Best: range≥50pt, 1 observed osc → `-0.73pt/trade` ($-1.46), 2.88 trades/session, 95% CI `[-1.27, -0.17]`.**

Negative even inside the explicitly observed state. The mechanical encoding of "the state" does not rescue the strategy; whatever the owner conditions on is not captured by observed-oscillation stability, and the only remaining source of truth is his actual corpus entries versus the extremes he declined.
