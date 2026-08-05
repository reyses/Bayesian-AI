# The two-stage exit scored on its OWN objective: risk

Owner: the exit is **designed not to lose money** — warn at a 20% retrace of MFE, exit 10% further, ratchet on new highs. Every prior evaluation scored it on EV; this scores the left tail. Precedent: the BE+2 stop was "dead" on EV and then measured **−46% vol / −24% DD at zero EV cost** when finally scored as the risk control it was.

Identical entries, paired. Honest fills, 20pt hard stop, friction 0.89pt, arm above 2pt. Sessions: **603**. Excluded: 2024_09_16.

## Entries at ±1.5σ — 54,904 trades, 539 sessions, 101.9/session

| exit | mean | 95% CI | std | %losers | p05 | p01 | CVaR5 | worst | maxDD (pt) |
|---|---|---|---|---|---|---|---|---|---|
| BAND | `-0.57` | `[-0.71, -0.43]` | 16.93 | 53.2% | `-20.89` | `-20.89` | `-20.89` | `-20.89` | 31,427 |
| RATCH-80 | `-0.80` | `[-0.86, -0.74]` | 7.32 | 31.5% | `-20.89` | `-20.89` | `-20.89` | `-20.89` | 43,753 |
| **TWO-STAGE 80/70 (entry-armed)** | `-0.83` | `[-0.89, -0.77]` | 7.70 | 34.9% | `-20.89` | `-20.89` | `-20.89` | `-20.89` | 45,522 |
| **PROTECT PROTOCOL (region-armed)** | `-0.81` | `[-0.97, -0.66]` | 19.08 | 38.6% | `-20.89` | `-20.89` | `-20.89` | `-20.89` | 44,818 |

Paired deltas, TWO-STAGE − BAND (day-block bootstrap, 4,000 resamples):
- mean: `-0.262` CI `[-0.405, -0.122]`
- std: `-9.23` CI `[-9.88, -8.61]` (-55% of BAND std)
- CVaR5: `+0.00` CI `[+0.00, +0.00]` — improvement `+0%` of BAND CVaR5 `-20.89`
- %losers: `-0.183` CI `[-0.189, -0.177]`
- maxDD: BAND 31,427pt → TWO 45,522pt (`+45%`)
- full −20pt stop-outs: BAND 22.0% → TWO 9.5%

### The design guarantee, by cushion size

The floor at peak=2pt is paper against a 20pt/5s bar; at peak=30pt it is armor. Pooling them answers the wrong question (blind-reimplementation finding: pooled P(loss|armed) ≈ 27% is dominated by tiny cushions).

| peak at exit | N | P(loss) | mean net | worst |
|---|---|---|---|---|
| 2–5pt | 34,170 | 36.4% (12447/34170) | `-0.03` | `-20.89` |
| 5–10pt | 10,799 | 11.2% (1213/10799) | `+2.18` | `-20.89` |
| 10–20pt | 3,538 | 3.7% (131/3538) | `+6.26` | `-20.89` |
| ≥20pt | 1,053 | 0.9% (10/1053) | `+18.73` | `-20.89` |

- armed overall (peak > 2pt): **90.3%** · P(loss|armed) **27.85%** (13801/49560)
- worst armed `-20.89pt` · worst unarmed `-20.89pt`

### PROTECT PROTOCOL — the guarantee where it actually arms

- reached the arm zone (≥85% of the entry→opposite-band distance): **61.6%** of trades (33,808)
- cushion at exit among armed: mean `19.7pt`
- **P(loss | region-armed): 0.59%** (198/33808) · mean `+11.40pt` · worst `-20.89pt`
- paired vs BAND on the same armed subset: mean delta `+4.25` CI `[+4.07, +4.43]`
- never-armed complement (fail-safe stop territory): 38.4% of trades, mean `-20.38pt`

**VERDICT: the design claim fails on its own axis** — tail improvement `+0%` < 20%.

## Entries at ±3σ — 4,885 trades, 538 sessions, 9.1/session

| exit | mean | 95% CI | std | %losers | p05 | p01 | CVaR5 | worst | maxDD (pt) |
|---|---|---|---|---|---|---|---|---|---|
| BAND | `-0.39` | `[-1.31, +0.57]` | 34.12 | 61.9% | `-20.89` | `-20.89` | `-20.89` | `-20.89` | 3,723 |
| RATCH-80 | `-0.78` | `[-1.02, -0.55]` | 8.25 | 35.9% | `-20.89` | `-20.89` | `-20.89` | `-20.89` | 3,932 |
| **TWO-STAGE 80/70 (entry-armed)** | `-0.74` | `[-1.00, -0.49]` | 9.07 | 38.6% | `-20.89` | `-20.89` | `-20.89` | `-20.89` | 3,704 |
| **PROTECT PROTOCOL (region-armed)** | `-0.80` | `[-1.47, -0.11]` | 24.60 | 52.2% | `-20.89` | `-20.89` | `-20.89` | `-20.89` | 4,654 |

Paired deltas, TWO-STAGE − BAND (day-block bootstrap, 4,000 resamples):
- mean: `-0.340` CI `[-1.199, +0.496]`
- std: `-24.96` CI `[-30.96, -20.74]` (-73% of BAND std)
- CVaR5: `+0.00` CI `[+0.00, +0.00]` — improvement `+0%` of BAND CVaR5 `-20.89`
- %losers: `-0.234` CI `[-0.250, -0.217]`
- maxDD: BAND 3,723pt → TWO 3,704pt (`-1%`)
- full −20pt stop-outs: BAND 51.5% → TWO 10.0%

### The design guarantee, by cushion size

The floor at peak=2pt is paper against a 20pt/5s bar; at peak=30pt it is armor. Pooling them answers the wrong question (blind-reimplementation finding: pooled P(loss|armed) ≈ 27% is dominated by tiny cushions).

| peak at exit | N | P(loss) | mean net | worst |
|---|---|---|---|---|
| 2–5pt | 2,614 | 44.1% (1154/2614) | `-0.54` | `-20.14` |
| 5–10pt | 1,134 | 17.5% (198/1134) | `+1.67` | `-16.89` |
| 10–20pt | 469 | 7.0% (33/469) | `+5.48` | `-11.14` |
| ≥20pt | 171 | 2.9% (5/171) | `+21.15` | `-18.39` |

- armed overall (peak > 2pt): **89.8%** · P(loss|armed) **31.68%** (1390/4388)
- worst armed `-20.14pt` · worst unarmed `-20.89pt`

### PROTECT PROTOCOL — the guarantee where it actually arms

- reached the arm zone (≥85% of the entry→opposite-band distance): **46.2%** of trades (2,259)
- cushion at exit among armed: mean `33.6pt`
- **P(loss | region-armed): 0.31%** (7/2259) · mean `+20.78pt` · worst `-10.14pt`
- paired vs BAND on the same armed subset: mean delta `+1.48` CI `[+0.02, +2.89]`
- never-armed complement (fail-safe stop territory): 53.8% of trades, mean `-19.36pt`

**VERDICT: the design claim fails on its own axis** — tail improvement `+0%` < 20%.

## Scope note

A validated fail-safe does not create expectancy: on a losing entry stream it makes you lose *less badly*. Its value is as the safety layer under the owner's SELECTIVE entries — which remain the open question (corpus, not compute).

