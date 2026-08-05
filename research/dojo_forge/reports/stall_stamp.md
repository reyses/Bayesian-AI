# Does the STALL stamp earn its place in the watcher?

A stall = N seconds with **no new favourable extreme** on the open position (not flatness). Shipped into `watch --stall N` on the strength of ONE leg — the same standard that produced the acceleration-inflection rule, which then died out of sample. This is the check.

Entries: the same ±1.5σ band touches used in every prior test (cubic 5s w90, edge-triggered, RTH). Friction 0.89pt. Max hold 30min.
Sessions: **72**, trades: **7306**. Excluded: 2024_09_16.

Benchmark — BAND exit: mean net `-0.78pt` 95% CI `[-1.17, -0.37]`. MFE mean `38.43pt`.

| stall N | fired | exit mean net | 95% CI | vs band Δ | Δ 95% CI | sig? | % of MFE captured |
|---|---|---|---|---|---|---|---|
| 5s | 7306 | `-0.81` | `[-0.88, -0.74]` | `-0.03` | `[-0.43, +0.36]` | **no** | 23.9% |
| 8s | 7306 | `-0.82` | `[-0.93, -0.70]` | `-0.04` | `[-0.43, +0.34]` | **no** | 26.2% |
| 12s | 7306 | `-0.88` | `[-1.02, -0.73]` | `-0.10` | `[-0.47, +0.28]` | **no** | 28.7% |
| 20s | 7306 | `-0.89` | `[-1.08, -0.69]` | `-0.11` | `[-0.47, +0.25]` | **no** | 33.0% |
| 30s | 7306 | `-0.92` | `[-1.16, -0.67]` | `-0.14` | `[-0.48, +0.21]` | **no** | 37.2% |

**Best: stall 5s → `-0.81pt/trade` ($-1.62), capturing 23.9% of MFE.**

No significant difference from the band exit. The stall stamp is **not an edge** — but as a halt-and-ask prompt it costs nothing versus the band, so it may stay as an attention device provided it is never sold as an exit rule.

Note the absolute level: `-0.81pt` is still a LOSING trade. Beating the band is not the same as making money, and no exit rule tested has yet cleared friction.
