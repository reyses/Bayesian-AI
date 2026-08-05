# Entering at a cubic-sigma extreme — does the FADE carry an edge?

ENTRY ONLY. No stop, no target, no trail — optional stopping already settles the exit question (gross EV is zero for any exit pair on a martingale), so mixing exits in here would only manufacture phantom edge, as the earlier T×N sweep did.

Cubic endpoint on 5s, window 90 (7.5min, deployed spec); residual sigma over 20min. RTH only. **Edge-triggered** touches. Friction `0.89pt` round trip shown where a trade is implied.
Excluded: 2024_09_16. Control = random bars from the SAME sessions and clock window, so time-of-day volatility cannot pose as a sigma effect.

Sessions scanned: **603**. Control N = **21384**.

## Forward 5 min

Control: mean `+0.00pt` 95% CI `[-0.30, +0.32]` (N=21384)

| entry | N | mean ret | 95% CI | vs control Δ | Δ 95% CI | sig? | median | win% |
|---|---|---|---|---|---|---|---|---|
| −2σ FADE (long) | 27766 | `+0.48` | `[+0.16, +0.81]` | `+0.48` | `[+0.03, +0.92]` | **YES** | `+1.00` | 52.6% |
| +2σ FADE (short) | 27027 | `+0.07` | `[-0.24, +0.40]` | `+0.07` | `[-0.38, +0.51]` | **no** | `-0.25` | 49.1% |
| −2.5σ FADE (long) | 12004 | `+0.55` | `[+0.02, +1.07]` | `+0.55` | `[-0.05, +1.16]` | **no** | `+1.25` | 52.6% |
| +2.5σ FADE (short) | 11595 | `-0.22` | `[-0.74, +0.28]` | `-0.22` | `[-0.82, +0.39]` | **no** | `-0.50` | 48.7% |
| −3σ FADE (long) | 5072 | `+0.52` | `[-0.43, +1.61]` | `+0.52` | `[-0.55, +1.59]` | **no** | `+1.25` | 52.8% |
| +3σ FADE (short) | 4879 | `+0.26` | `[-0.67, +1.15]` | `+0.26` | `[-0.68, +1.21]` | **no** | `-0.50` | 48.4% |

## Forward 10 min

Control: mean `+0.00pt` 95% CI `[-0.45, +0.46]` (N=21384)

| entry | N | mean ret | 95% CI | vs control Δ | Δ 95% CI | sig? | median | win% |
|---|---|---|---|---|---|---|---|---|
| −2σ FADE (long) | 27766 | `+0.24` | `[-0.18, +0.67]` | `+0.24` | `[-0.38, +0.86]` | **no** | `+1.25` | 52.0% |
| +2σ FADE (short) | 27027 | `+0.00` | `[-0.42, +0.44]` | `+0.00` | `[-0.63, +0.63]` | **no** | `-0.75` | 48.5% |
| −2.5σ FADE (long) | 12004 | `+0.25` | `[-0.43, +0.96]` | `+0.25` | `[-0.55, +1.08]` | **no** | `+1.00` | 51.6% |
| +2.5σ FADE (short) | 11595 | `-0.21` | `[-0.95, +0.53]` | `-0.21` | `[-1.07, +0.63]` | **no** | `-0.75` | 48.6% |
| −3σ FADE (long) | 5072 | `+0.01` | `[-1.19, +1.28]` | `+0.01` | `[-1.35, +1.33]` | **no** | `+1.00` | 51.7% |
| +3σ FADE (short) | 4879 | `+0.09` | `[-1.20, +1.27]` | `+0.09` | `[-1.22, +1.39]` | **no** | `-0.25` | 49.5% |

## Forward 20 min

Control: mean `+0.00pt` 95% CI `[-0.61, +0.63]` (N=21384)

| entry | N | mean ret | 95% CI | vs control Δ | Δ 95% CI | sig? | median | win% |
|---|---|---|---|---|---|---|---|---|
| −2σ FADE (long) | 27766 | `+0.59` | `[+0.00, +1.15]` | `+0.59` | `[-0.28, +1.46]` | **no** | `+1.50` | 51.9% |
| +2σ FADE (short) | 27027 | `-0.03` | `[-0.58, +0.55]` | `-0.03` | `[-0.88, +0.84]` | **no** | `-1.00` | 48.4% |
| −2.5σ FADE (long) | 12004 | `+0.75` | `[-0.17, +1.68]` | `+0.75` | `[-0.37, +1.84]` | **no** | `+1.75` | 52.2% |
| +2.5σ FADE (short) | 11595 | `-0.16` | `[-1.09, +0.78]` | `-0.16` | `[-1.24, +0.97]` | **no** | `-0.50` | 49.1% |
| −3σ FADE (long) | 5072 | `-0.11` | `[-1.62, +1.59]` | `-0.11` | `[-1.84, +1.67]` | **no** | `+1.25` | 51.6% |
| +3σ FADE (short) | 4879 | `+0.56` | `[-1.10, +2.11]` | `+0.56` | `[-1.24, +2.36]` | **no** | `+0.00` | 49.8% |

## How to read this

A row is only interesting if **sig? = YES** — the Δ-vs-control CI must exclude zero. A positive mean with a CI spanning zero is noise, no matter how large the point estimate.
Any surviving edge must then clear friction (`0.89pt` = `$1.78`) before it is a trade rather than a statistic.

