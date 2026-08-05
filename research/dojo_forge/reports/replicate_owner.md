# Replicating the owner's trade as a rule

His method: identify the oscillation, enter at an extreme, **hold**, and exit on a marker that retains X% of the PEAK open profit. The distinguishing feature is not the entry — it is that he held. Every mechanical exit tested so far leaves early.

Entries: the same ±1.5σ extreme touches as every prior test. Ratchet arms only once peak profit exceeds 2pt. Hard stop 20pt. Friction 0.89pt. Max hold 30min.
Sessions **87**, trades **8791**, mean MFE `38.10pt`. Excluded: 2024_09_16.

## Benchmarks (identical entries)

- BAND exit: `-0.60pt` 95% CI `[-0.95, -0.24]`
- HOLD to opposite extreme: `-0.59pt` 95% CI `[-0.95, -0.23]`

## Owner rule — retain X% of peak open profit

| retain | mean net | 95% CI | vs BAND Δ | Δ 95% CI | sig? | $/trade |
|---|---|---|---|---|---|---|
| 50% | `-0.77` | `[-0.98, -0.56]` | `-0.17` | `[-0.50, +0.18]` | **no** | `$-1.54` |
| 60% | `-0.78` | `[-0.97, -0.58]` | `-0.17` | `[-0.50, +0.18]` | **no** | `$-1.55` |
| 70% | `-0.78` | `[-0.95, -0.60]` | `-0.18` | `[-0.51, +0.16]` | **no** | `$-1.56` |
| 80% | `-0.78` | `[-0.93, -0.62]` | `-0.18` | `[-0.51, +0.16]` | **no** | `$-1.56` |
| 90% | `-0.80` | `[-0.95, -0.65]` | `-0.19` | `[-0.53, +0.15]` | **no** | `$-1.59` |

**Best: retain 50% → `-0.77pt/trade` ($-1.54).**

**Not significantly better than the band.** The rule does not reproduce what he did — which means the replicable part is not the exit geometry, and the search should move to what he conditioned the ENTRY on.
