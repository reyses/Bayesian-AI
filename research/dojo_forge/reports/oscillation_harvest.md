# Does OBSERVED oscillation predict the next one?

Wait for the regime to identify itself, then harvest — rather than predicting it (oscillator/runaway discrimination is stuck ~0.57 AUC).

Traverse = z crosses from ±1.5σ through to the opposite band (cubic 5s w90, σ over 20min), edge-triggered, RTH only. At each completion: K = traverses in the prior 30min, fade toward the opposite band, stop 20pt (outside the fakeout distribution), max hold 60min.
Friction `0.89pt` charged per attempt. Excluded: 2024_09_16.

Sessions: **539** · attempts: **54911**

## Outcome and edge vs K (prior observed traverses)

| K | N | complete | runaway | timeout | mean net (pt) | 95% CI | $/trade |
|---|---|---|---|---|---|---|---|
| 0 | 558 | 63.6% | 36.4% | 0.0% | `-1.12` | `[-2.68, +0.44]` | `$-2.25` |
| 1 | 612 | 66.3% | 33.7% | 0.0% | `-0.47` | `[-2.01, +1.16]` | `$-0.94` |
| 2 | 964 | 70.9% | 29.0% | 0.1% | `-0.95` | `[-2.14, +0.20]` | `$-1.90` |
| 3–4 | 5289 | 77.4% | 22.6% | 0.0% | `-0.60` | `[-1.06, -0.15]` | `$-1.19` |
| 5–+ | 47488 | 78.5% | 21.5% | 0.0% | `-0.55` | `[-0.70, -0.40]` | `$-1.10` |

**All attempts pooled:** N=54911, complete 78.0%, mean net `-0.57pt` 95% CI `[-0.71, -0.42]` → significant

## P(complete) by exact K

| K | N | P(complete) |
|---|---|---|
| 0 | 558 | 63.6% |
| 1 | 612 | 66.3% |
| 2 | 964 | 70.9% |
| 3 | 1769 | 75.0% |
| 4 | 3520 | 78.6% |
| 5 | 5649 | 79.4% |
| 6 | 7743 | 78.6% |
| 7 | 8827 | 78.5% |
| 8 | 8334 | 78.5% |
| 9 | 6798 | 78.6% |
| 10 | 4697 | 80.1% |
| 11 | 2800 | 77.4% |
| 12 | 1497 | 77.6% |
| 13 | 647 | 69.1% |
| 14 | 292 | 75.3% |
| 15 | 123 | 63.4% |
| 16 | 51 | 68.6% |

K=0 → K=16: `63.6%` → `68.6%` (+5.0pp). Chop begets chop.

