# The owner's algorithm — swept across OSCILLATION SCALE

Every prior test fixed the band at 1.5σ (~13pt, **106 traverses per session**). His stated oscillation was 19640-19700 = **60pt**, and he took ONE trade in forty minutes. That is a 2.5x scale mismatch and a 10x frequency mismatch — I was measuring a different instrument and calling it his method.

Also recorded: my "one entry per region per oscillation" filter was **vacuous**. Band traverses ALTERNATE by construction, so the constraint is already satisfied by every traverse; both arms returned identical numbers. The re-arm rule blocked nothing.

Hard stop 20pt, friction 0.89pt, max hold 30min, ratchet arms above 2pt. **Honest fills** (close after the breach, never the floor — that error was worth +1.44pt/trade).
Sessions: **603**. Excluded: 2024_09_16.

| band | ~width | giveback | trades/session | mean net | 95% CI | $/trade |
|---|---|---|---|---|---|---|
| 3.0σ | ~26pt | 20% | 9.1 | `-0.78` | `[-1.02, -0.55]` | `$-1.56` |
| 3.0σ | ~26pt | 30% | 9.1 | `-0.74` | `[-1.00, -0.49]` | `$-1.47` |
| 3.5σ | ~30pt | 20% | 4.4 | `-0.49` | `[-0.96, +0.07]` | `$-0.98` |
| 3.5σ | ~30pt | 30% | 4.4 | `-0.50` | `[-0.96, +0.00]` | `$-1.00` |
| 4.0σ | ~35pt | 20% | 2.6 | `-0.47` | `[-0.97, +0.03]` | `$-0.94` |
| 4.0σ | ~35pt | 30% | 2.6 | `-0.63` | `[-1.14, -0.10]` | `$-1.26` |
| 4.5σ | ~39pt | 20% | 1.8 | `-0.22` | `[-0.97, +0.50]` | `$-0.45` |
| 4.5σ | ~39pt | 30% | 1.8 | `-0.49` | `[-1.23, +0.24]` | `$-0.99` |
| 5.0σ | ~44pt | 20% | 1.4 | `-0.84` | `[-1.93, +0.23]` | `$-1.69` |
| 5.0σ | ~44pt | 30% | 1.4 | `-0.96` | `[-2.08, +0.13]` | `$-1.92` |

**Best: band 4.5σ, giveback 20% → `-0.22pt/trade` ($-0.45), 1.8 trades/session, 95% CI `[-0.97, +0.50]`.**

Negative at every scale. Scale is **not** the missing variable either, and the exit geometry stays dead regardless of oscillation size. What remains is *which* oscillations he chose to trade — answerable only from his actual entries, not from any sweep.
