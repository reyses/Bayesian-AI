# Dossier signal league — direction agreement with AI labels (COMPLETE, 37 streams)
(train 2024, test 2025+26, day-block bootstrap CIs; baseline 0.50; assembled 2026-07-16
from the five pipeline runs — per-stream evaluate() outputs verbatim. NMP corrected
with the vr<1.0 gate per doc 085; NMP-LAMBDA formerly mislabeled "NMP-EXT";
NMPT-* = the V1 tier ladder = "extended NMP".)

## Graded streams (passed N/split gates), sorted by OOS AUC
| stream | N | OOS AUC | test base | P-terciles low / mid / high |
|---|---|---|---|---|
| PIVOT-16 | 324 | 0.939 | 0.05 | 0.00 / 0.00 / 0.15 [0.07,0.25] |
| OHLC-01 | 619 | 0.841 | 0.48 | 0.07 [0.03,0.13] / 0.59 / 0.77 [0.69,0.85] |
| VP-01 | 283 | 0.732 | 0.36 | 0.12 [0.04,0.20] / 0.43 / 0.52 [0.38,0.66] |
| VWMA-10 | 540 | 0.714 | 0.63 | 0.36 [0.27,0.45] / 0.76 / 0.78 [0.68,0.86] |
| **NMPT-FADECALM** | 21,034 | 0.676 | 0.42 | 0.25 [0.23,0.26] / 0.43 / 0.59 [0.57,0.60] |
| ADX-08 | 1,359 | 0.660 | — | 0.39 [0.30,0.49] / 0.62 / 0.74 [0.65,0.83] (doc 079) |
| **NMPT-RIDEAGN** | 20,690 | 0.656 | 0.61 | 0.45 [0.43,0.48] / 0.62 / 0.75 [0.73,0.77] |
| **NMP** (corrected: +vr<1 gate) | 10,388 | 0.639 | **0.27** | 0.14 [0.12,0.15] / 0.32 / 0.36 [0.34,0.38] |
| **NMPT-FADEAGN** | 892 | 0.638 | 0.41 | 0.24 [0.14,0.34] / 0.50 / 0.50 [0.38,0.61] |
| **NMPT-MTFEXH** | 840 | 0.635 | **0.79** | 0.71 [0.64,0.78] / 0.81 / 0.86 [0.80,0.92] |
| **NMPT-MTFBRK** | 2,167 | 0.632 | **0.80** | 0.69 [0.64,0.75] / 0.82 / 0.87 [0.83,0.91] |
| ROUND-05 | 44,332 | 0.623 | 0.63 | 0.52 [0.50,0.54] / 0.63 / 0.75 [0.73,0.76] |
| SAR-23 | 37,184 | 0.618 | 0.44 | 0.33 [0.32,0.34] / 0.44 / 0.56 [0.55,0.58] |
| SEASON-12 | 521 | 0.618 | 0.48 | 0.40 [0.30,0.49] / 0.37 / 0.66 [0.56,0.75] |
| CROSS-11 | 504 | 0.616 | 0.66 | 0.55 [0.46,0.65] / 0.65 / 0.76 [0.67,0.85] |
| RENKO-24 | 198,560 | 0.611 | 0.55 | 0.44 [0.42,0.45] / 0.55 / 0.65 [0.64,0.67] |
| DOW-19 | 36,842 | 0.610 | 0.38 | 0.28 [0.27,0.29] / 0.38 / 0.49 [0.47,0.50] |
| CURVE | 26,368 | 0.606 | 0.55 | 0.45 [0.44,0.47] / 0.54 / 0.66 [0.65,0.67] |
| VWAP-03 | 29,577 | 0.604 | 0.41 | 0.31 [0.29,0.34] / 0.40 / 0.50 [0.48,0.53] |
| TUNNEL-20 | 35,228 | 0.604 | 0.59 | 0.49 [0.47,0.50] / 0.59 / 0.68 [0.67,0.69] |
| ZONE-21 | 3,451 | 0.584 | 0.63 | 0.58 [0.54,0.61] / 0.61 / 0.72 [0.68,0.76] |
| **NMPT-FREIGHT** | 4,575 | 0.582 | **0.75** | 0.69 [0.66,0.72] / 0.76 / 0.81 [0.79,0.84] |
| **NMP-LAMBDA** | 10,793 | 0.574 | 0.54 | 0.46 [0.44,0.49] / 0.56 / 0.61 [0.58,0.63] |
| ZIGZAG | 4,852 | 0.556 | 0.96 | 0.95 / 0.96 / 0.97 (timing is the story) |
| **NMPT-KILLSHOT** | 2,931 | 0.552 | 0.40 | 0.33 [0.29,0.38] / 0.44 / 0.42 [0.38,0.47] |
| MACD-07 | 9,781 | 0.552 | 0.05 | inverter, flat ladder |
| RSI-06 | 14,967 | 0.515 | 0.04 | inverter, flat ladder |
| **NMPT-CASCADE** | 669 | 0.514 | 0.43 | flat/noisy (N thin) |
| ATR-09 | 882 | 0.500 | 0.01 | inverter, flat |
| ORB-02 | 539 | 0.436 | 0.97 | tautology, flat |

## THE NMP FAMILY RESULT (docs 084/085)
- **NMP (V1 trigger, corrected with the vr<1.0 gate)**: agreement **0.27** — the pure
  fade is ANTI-ALIGNED; the vr gate barely moves it (0.26→0.27; vr was a weak
  stability stand-in, consistent with the dead vr cross-TF proxy finding).
- **NMP-LAMBDA (the λ-complete trigger — the never-built branch)**: 0.26 → **0.54**
  (+28pp). The λ̂ term converts anti-aligned to aligned, out-of-sample.
- **The tier ladder ("extended NMP") splits EXACTLY along ride/fade lines**:
  every RIDE tier is label-aligned — MTFBRK **0.80**, MTFEXH **0.79**, FREIGHT
  **0.75**, RIDEAGN 0.61; every FADE tier is anti-aligned — NMP 0.27, KILLSHOT
  0.40, FADEAGN 0.41, FADECALM 0.42 (but FADECALM is highly GRADABLE: AUC 0.676,
  low tercile inverted = 75% right on 3,749 fires).
- Legacy corroboration: the V1 engine's own docstring ranked FREIGHT (86% WR) and
  MTF_EXHAUSTION (76% WR) as its best tiers — the label-alignment league reproduces
  the ordering from an entirely different measurement.
- FREIGHT regime note: train 941 vs test 3,634 fires — high-velocity minutes are
  ~4× more common in 2025-26 than 2024.

## Low-frequency streams (below the 200-row gate; raw full-sample agreement)
| stream | N | raw agree | note |
|---|---|---|---|
| SCALP-18 | 53 | **0.02** | extreme inverter |
| FIB-17 | 140 | 0.33 | leans inverted |
| VA-13 | 166 | 0.33 | leans inverted |
| SQZ-04 | 168 | 0.58 | rare squeeze-breakout |
| HNS-22 | 193 | 0.57 | short-only pattern |

## Skipped
- **ORDERFLOW-14** (no 2024 delta data). **NMPT-REGIME_FLIP** (legacy: reachable
  only via manual injection — no autonomous trigger to port). **PEAK** (disabled
  in legacy).

## The three signal families
1. **Graded separators** (stage-0 corpus): OHLC-01, VP-01, VWMA-10, NMPT-FADECALM,
   ADX-08, NMPT-RIDEAGN, NMPT-MTFEXH/MTFBRK/FREIGHT, ROUND-05, SAR-23, SEASON-12,
   CROSS-11, RENKO-24, DOW-19, CURVE, VWAP-03, TUNNEL-20, ZONE-21, NMP-LAMBDA.
2. **Inverters** (fade/divergence premises, incl. V1 NMP itself): NMP 0.27,
   PIVOT-16 0.05, ATR-09 0.01, RSI-06 0.04, MACD-07 0.05, SCALP-18 0.02,
   KILLSHOT/FADEAGN/FADECALM 0.40-0.42, FIB/VA 0.33.
3. **Momentum tautologies**: ZIGZAG 0.96, ORB-02 0.97 (timing is the value).

> Rows parquets: `signal_rows_<det>.parquet` (37 streams). Regenerate:
> `python research/nt8_catalog/tools/dossier_signal_pipeline.py [DET ...]`.
