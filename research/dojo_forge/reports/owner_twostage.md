# The owner's algorithm — swept across OSCILLATION SCALE

Every prior test fixed the band at 1.5σ (~13pt, **106 traverses per session**). His stated oscillation was 19640-19700 = **60pt**, and he took ONE trade in forty minutes. That is a 2.5x scale mismatch and a 10x frequency mismatch — I was measuring a different instrument and calling it his method.

Also recorded: my "one entry per region per oscillation" filter was **vacuous**. Band traverses ALTERNATE by construction, so the constraint is already satisfied by every traverse; both arms returned identical numbers. The re-arm rule blocked nothing.

Hard stop 20pt, friction 0.89pt, max hold 30min, ratchet arms above 2pt. **Honest fills** (close after the breach, never the floor — that error was worth +1.44pt/trade).
Sessions: **603**. Excluded: 2024_09_16.

| band | ~width | giveback | trades/session | mean net | 95% CI | $/trade |
|---|---|---|---|---|---|---|
| 3.0σ | ~26pt | 1-stage 20% | 9.1 | `-0.78` | `[-1.02, -0.55]` | `$-1.56` |
| 3.0σ | ~26pt | 1-stage 30% | 9.1 | `-0.74` | `[-1.00, -0.49]` | `$-1.47` |
| 3.0σ | ~26pt | **2-stage 80%/70%** | 9.1 | `-0.75` | `[-1.07, -0.43]` | `$-1.51` |
| 3.0σ | ~26pt | **2-stage 80%/60%** | 9.1 | `-0.63` | `[-1.01, -0.24]` | `$-1.26` |
| 3.0σ | ~26pt | **2-stage 90%/80%** | 9.1 | `-0.80` | `[-1.09, -0.52]` | `$-1.61` |
| 3.0σ | ~26pt | **2-stage 70%/60%** | 9.1 | `-0.54` | `[-0.95, -0.12]` | `$-1.09` |
| 3.5σ | ~30pt | 1-stage 20% | 4.4 | `-0.49` | `[-0.96, +0.07]` | `$-0.98` |
| 3.5σ | ~30pt | 1-stage 30% | 4.4 | `-0.50` | `[-0.96, +0.00]` | `$-1.00` |
| 3.5σ | ~30pt | **2-stage 80%/70%** | 4.4 | `-0.55` | `[-1.08, +0.04]` | `$-1.10` |
| 3.5σ | ~30pt | **2-stage 80%/60%** | 4.4 | `-0.33` | `[-0.95, +0.34]` | `$-0.65` |
| 3.5σ | ~30pt | **2-stage 90%/80%** | 4.4 | `-0.50` | `[-1.05, +0.09]` | `$-1.01` |
| 3.5σ | ~30pt | **2-stage 70%/60%** | 4.4 | `-0.26` | `[-0.90, +0.41]` | `$-0.52` |
| 4.0σ | ~35pt | 1-stage 20% | 2.6 | `-0.47` | `[-0.97, +0.03]` | `$-0.94` |
| 4.0σ | ~35pt | 1-stage 30% | 2.6 | `-0.63` | `[-1.14, -0.10]` | `$-1.26` |
| 4.0σ | ~35pt | **2-stage 80%/70%** | 2.6 | `-0.58` | `[-1.25, +0.11]` | `$-1.16` |
| 4.0σ | ~35pt | **2-stage 80%/60%** | 2.6 | `-0.53` | `[-1.26, +0.23]` | `$-1.07` |
| 4.0σ | ~35pt | **2-stage 90%/80%** | 2.6 | `-0.52` | `[-1.17, +0.14]` | `$-1.04` |
| 4.0σ | ~35pt | **2-stage 70%/60%** | 2.6 | `-0.39` | `[-1.23, +0.55]` | `$-0.77` |
| 4.5σ | ~39pt | 1-stage 20% | 1.8 | `-0.22` | `[-0.97, +0.50]` | `$-0.45` |
| 4.5σ | ~39pt | 1-stage 30% | 1.8 | `-0.49` | `[-1.23, +0.24]` | `$-0.99` |
| 4.5σ | ~39pt | **2-stage 80%/70%** | 1.8 | `-0.46` | `[-1.49, +0.60]` | `$-0.92` |
| 4.5σ | ~39pt | **2-stage 80%/60%** | 1.8 | `-0.03` | `[-1.32, +1.38]` | `$-0.06` |
| 4.5σ | ~39pt | **2-stage 90%/80%** | 1.8 | `-0.48` | `[-1.44, +0.43]` | `$-0.97` |
| 4.5σ | ~39pt | **2-stage 70%/60%** | 1.8 | `-0.19` | `[-1.44, +1.22]` | `$-0.38` |
| 5.0σ | ~44pt | 1-stage 20% | 1.4 | `-0.84` | `[-1.93, +0.23]` | `$-1.69` |
| 5.0σ | ~44pt | 1-stage 30% | 1.4 | `-0.96` | `[-2.08, +0.13]` | `$-1.92` |
| 5.0σ | ~44pt | **2-stage 80%/70%** | 1.4 | `-1.64` | `[-3.14, +0.04]` | `$-3.27` |
| 5.0σ | ~44pt | **2-stage 80%/60%** | 1.4 | `-1.08` | `[-2.77, +0.73]` | `$-2.16` |
| 5.0σ | ~44pt | **2-stage 90%/80%** | 1.4 | `-1.89` | `[-3.17, -0.59]` | `$-3.79` |
| 5.0σ | ~44pt | **2-stage 70%/60%** | 1.4 | `-1.31` | `[-2.98, +0.48]` | `$-2.62` |

**Best: band 4.5σ, 2-stage 80%/60% → `-0.03pt/trade` ($-0.06), 1.8 trades/session, 95% CI `[-1.32, +1.38]`.**

Negative at every scale. Scale is **not** the missing variable either, and the exit geometry stays dead regardless of oscillation size. What remains is *which* oscillations he chose to trade — answerable only from his actual entries, not from any sweep.
