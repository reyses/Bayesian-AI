# The owner's algorithm — swept across OSCILLATION SCALE

Every prior test fixed the band at 1.5σ (~13pt, **106 traverses per session**). His stated oscillation was 19640-19700 = **60pt**, and he took ONE trade in forty minutes. That is a 2.5x scale mismatch and a 10x frequency mismatch — I was measuring a different instrument and calling it his method.

Also recorded: my "one entry per region per oscillation" filter was **vacuous**. Band traverses ALTERNATE by construction, so the constraint is already satisfied by every traverse; both arms returned identical numbers. The re-arm rule blocked nothing.

Hard stop 20pt, friction 0.89pt, max hold 30min, ratchet arms above 2pt. **Honest fills** (close after the breach, never the floor — that error was worth +1.44pt/trade).
Sessions: **100**. Excluded: 2024_09_16.

| band | ~width | giveback | trades/session | mean net | 95% CI | $/trade |
|---|---|---|---|---|---|---|
| 1.5σ | ~13pt | 10% | 101.0 | `-0.80` | `[-0.95, -0.65]` | `$-1.59` |
| 1.5σ | ~13pt | 20% | 101.0 | `-0.78` | `[-0.93, -0.62]` | `$-1.56` |
| 1.5σ | ~13pt | 30% | 101.0 | `-0.78` | `[-0.95, -0.60]` | `$-1.56` |
| 2.0σ | ~17pt | 10% | 50.9 | `-0.71` | `[-0.92, -0.49]` | `$-1.42` |
| 2.0σ | ~17pt | 20% | 50.9 | `-0.71` | `[-0.93, -0.49]` | `$-1.43` |
| 2.0σ | ~17pt | 30% | 50.9 | `-0.67` | `[-0.93, -0.39]` | `$-1.33` |
| 2.5σ | ~22pt | 10% | 22.0 | `-0.69` | `[-1.02, -0.37]` | `$-1.38` |
| 2.5σ | ~22pt | 20% | 22.0 | `-0.67` | `[-1.02, -0.35]` | `$-1.35` |
| 2.5σ | ~22pt | 30% | 22.0 | `-0.55` | `[-0.99, -0.09]` | `$-1.11` |
| 3.0σ | ~26pt | 10% | 8.9 | `-0.60` | `[-1.15, -0.06]` | `$-1.19` |
| 3.0σ | ~26pt | 20% | 8.9 | `-0.61` | `[-1.19, -0.05]` | `$-1.23` |
| 3.0σ | ~26pt | 30% | 8.9 | `-0.50` | `[-1.17, +0.25]` | `$-0.99` |
| 4.0σ | ~35pt | 10% | 2.8 | `+0.28` | `[-0.79, +1.33]` | `$+0.57` |
| 4.0σ | ~35pt | 20% | 2.8 | `+0.60` | `[-0.58, +1.81]` | `$+1.20` |
| 4.0σ | ~35pt | 30% | 2.8 | `+0.45` | `[-0.83, +1.81]` | `$+0.90` |

**Best: band 4.0σ, giveback 20% → `+0.60pt/trade` ($+1.20), 2.8 trades/session, 95% CI `[-0.58, +1.81]`.**

Positive but the CI spans zero. Wider bands mean far fewer trades, so this is underpowered by construction — the honest read is *suggestive, not established*.
