# Does OBSERVED oscillation predict the next one?

Wait for the regime to identify itself, then harvest — rather than predicting it (oscillator/runaway discrimination is stuck ~0.57 AUC).

Traverse = z crosses from ±1.5σ through to the opposite band (cubic 5s w90, σ over 20min), edge-triggered, RTH only. At each completion: K = traverses in the prior 30min, fade toward the opposite band, stop 3x sigma (scales with the band), max hold 60min.
Friction `0.89pt` charged per attempt. Excluded: 2024_09_16.

Sessions: **539** · attempts: **54911**

## Outcome and edge vs K (prior observed traverses)

| K | N | complete | runaway | timeout | mean net (pt) | 95% CI | $/trade |
|---|---|---|---|---|---|---|---|
| 0 | 558 | 37.1% | 62.9% | 0.0% | `-1.16` | `[-2.49, +0.27]` | `$-2.33` |
| 1 | 612 | 46.1% | 53.9% | 0.0% | `-0.41` | `[-1.88, +1.08]` | `$-0.82` |
| 2 | 964 | 54.9% | 45.0% | 0.1% | `-1.02` | `[-2.22, +0.17]` | `$-2.04` |
| 3–4 | 5289 | 61.7% | 38.3% | 0.0% | `-0.58` | `[-1.09, -0.09]` | `$-1.16` |
| 5–+ | 47488 | 63.2% | 36.8% | 0.0% | `-0.62` | `[-0.78, -0.46]` | `$-1.24` |

**All attempts pooled:** N=54911, complete 62.4%, mean net `-0.63pt` 95% CI `[-0.78, -0.47]` → significant

## P(complete) by exact K

| K | N | P(complete) |
|---|---|---|
| 0 | 558 | 37.1% |
| 1 | 612 | 46.1% |
| 2 | 964 | 54.9% |
| 3 | 1769 | 58.6% |
| 4 | 3520 | 63.3% |
| 5 | 5649 | 63.1% |
| 6 | 7743 | 62.9% |
| 7 | 8827 | 63.7% |
| 8 | 8334 | 63.2% |
| 9 | 6798 | 63.0% |
| 10 | 4697 | 64.6% |
| 11 | 2800 | 63.4% |
| 12 | 1497 | 60.4% |
| 13 | 647 | 58.1% |
| 14 | 292 | 62.3% |
| 15 | 123 | 61.0% |
| 16 | 51 | 62.7% |

K=0 → K=16: `37.1%` → `62.7%` (+25.6pp). Chop begets chop.

