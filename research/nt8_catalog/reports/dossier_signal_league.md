# Dossier signal league — direction agreement with AI labels (COMPLETE, 27 streams)
(train 2024, test 2025+26, day-block bootstrap CIs; baseline 0.50; assembled 2026-07-16
from the four pipeline runs — per-stream evaluate() outputs verbatim)

## Graded streams (passed N/split gates), sorted by OOS AUC
| stream | N | OOS AUC | test base | P-terciles low / mid / high |
|---|---|---|---|---|
| PIVOT-16 | 324 | 0.939 | 0.05 | 0.00 [0,0] / 0.00 [0,0] / 0.15 [0.07,0.25] |
| OHLC-01 | 619 | 0.841 | 0.48 | 0.07 [0.03,0.13] / 0.59 [0.50,0.67] / 0.77 [0.69,0.85] |
| VP-01 | 283 | 0.732 | 0.36 | 0.12 [0.04,0.20] / 0.43 [0.31,0.57] / 0.52 [0.38,0.66] |
| VWMA-10 | 540 | 0.714 | 0.63 | 0.36 [0.27,0.45] / 0.76 [0.67,0.84] / 0.78 [0.68,0.86] |
| ADX-08 | 1,359 | 0.660 | — | 0.39 [0.30,0.49] / 0.62 / 0.74 [0.65,0.83] (doc 079) |
| **NMP** | 10,993 | 0.648 | **0.26** | **0.13 [0.11,0.14]** / 0.31 [0.29,0.33] / 0.36 [0.34,0.38] |
| ROUND-05 | 44,332 | 0.623 | 0.63 | 0.52 [0.50,0.54] / 0.63 [0.62,0.65] / 0.75 [0.73,0.76] |
| SAR-23 | 37,184 | 0.618 | 0.44 | 0.33 [0.32,0.34] / 0.44 [0.42,0.45] / 0.56 [0.55,0.58] |
| SEASON-12 | 521 | 0.618 | 0.48 | 0.40 [0.30,0.49] / 0.37 [0.26,0.47] / 0.66 [0.56,0.75] |
| CROSS-11 | 504 | 0.616 | 0.66 | 0.55 [0.46,0.65] / 0.65 [0.55,0.75] / 0.76 [0.67,0.85] |
| RENKO-24 | 198,560 | 0.611 | 0.55 | 0.44 [0.42,0.45] / 0.55 [0.54,0.56] / 0.65 [0.64,0.67] |
| DOW-19 | 36,842 | 0.610 | 0.38 | 0.28 [0.27,0.29] / 0.38 [0.37,0.40] / 0.49 [0.47,0.50] |
| CURVE | 26,368 | 0.606 | 0.55 | 0.45 [0.44,0.47] / 0.54 [0.52,0.55] / 0.66 [0.65,0.67] |
| VWAP-03 | 29,577 | 0.604 | 0.41 | 0.31 [0.29,0.34] / 0.40 [0.37,0.42] / 0.50 [0.48,0.53] |
| TUNNEL-20 | 35,228 | 0.604 | 0.59 | 0.49 [0.47,0.50] / 0.59 [0.58,0.61] / 0.68 [0.67,0.69] |
| ZONE-21 | 3,451 | 0.584 | 0.63 | 0.58 [0.54,0.61] / 0.61 [0.57,0.65] / 0.72 [0.68,0.76] |
| **NMP-EXT** | 10,793 | 0.574 | **0.54** | 0.46 [0.44,0.49] / 0.56 [0.53,0.58] / 0.61 [0.58,0.63] |
| ZIGZAG | 4,852 | 0.556 | 0.96 | 0.95 [0.94,0.97] / 0.96 [0.95,0.97] / 0.97 [0.96,0.98] |
| MACD-07 | 9,781 | 0.552 | 0.05 | 0.05 [0.04,0.06] / 0.04 [0.03,0.05] / 0.07 [0.06,0.08] |
| RSI-06 | 14,967 | 0.515 | 0.04 | 0.04 [0.03,0.05] / 0.04 [0.03,0.05] / 0.05 [0.04,0.06] |
| ATR-09 | 882 | 0.500 | 0.01 | 0.01 / 0.01 / 0.01 (flat) |
| ORB-02 | 539 | 0.436 | 0.97 | 0.98 / 0.96 / 0.97 (flat) |

## THE NMP RESULT (λ-completion thesis, first label-alignment validation)
- **NMP** = V1 master equation as it ran live (|z_se|>1.8481 → FADE, λ hardcoded 0;
  re-arm at |z|<0.4752; canonical L3_1m_z_se_15, verified thresholds): agreement
  **0.26** — the pure fade is ANTI-ALIGNED (labels ride the move it fades). Its low
  tercile INVERTED = 87% right on 1,909 OOS fires.
- **NMP-EXT** = the COMPLETED equation (λ̂<0 fade / λ̂≥0 ride; λ̂ = k=21 trailing OLS
  slope of log(|z_se|+0.1) on closed 1m bars, verified derivation): flips 59.6% of
  fires to ride and lifts agreement **0.26 → 0.54 (+28pp)**. The missing λ term IS
  the difference between anti-aligned and aligned — measured, out-of-sample.

## Low-frequency streams (below the 200-row gate; raw full-sample agreement)
| stream | N | fires/day | raw agree | note |
|---|---|---|---|---|
| SCALP-18 | 53 | 0.09 | **0.02** | extreme inverter (pullback fires mid-countermove) |
| FIB-17 | 140 | 0.24 | 0.33 | leans inverted; daily ADX(7)>25 gate is restrictive |
| VA-13 | 166 | 0.29 | 0.33 | leans inverted |
| SQZ-04 | 168 | 0.29 | 0.58 | mild momentum alignment; squeeze-breakout rare on 5s |
| HNS-22 | 193 | 0.34 | 0.57 | short-only pattern; mild alignment |

## Skipped (not fabricable)
- **ORDERFLOW-14**: delta data covers 2025-07-30 → 2026-01-29 only (no 2024 train
  year; 2024 GLBX trades dumps not on disk). Revisit if 2024 trades are sourced.

## The three signal families (structural finding)
1. **Graded separators** — OHLC-01, VP-01, VWMA-10, ADX-08, NMP-EXT, ROUND-05,
   SAR-23, SEASON-12, CROSS-11, RENKO-24, DOW-19, CURVE, VWAP-03, TUNNEL-20,
   ZONE-21: balanced-ish base + real tercile ladders. The stage-0 corpus.
2. **Inverters** — NMP (0.26!), PIVOT-16 (0.05), ATR-09 (0.01), RSI-06 (0.04),
   MACD-07 (0.05), SCALP-18 (0.02), FIB-17/VA-13 (0.33): EVERY fade/divergence/
   pullback premise — including the V1 master equation itself — is anti-aligned;
   they fire mid-move while the label rides the move. Flipped, they are
   continuation confirmations.
3. **Momentum tautologies** — ZIGZAG (0.96), ORB-02 (0.97): right by construction;
   the value is timing (see zigzag_phase_in_label.md).

> Per-stream coefs and train/test Ns: run logs (docs 081/083/084) and
> `signal_rows_<det>.parquet` (all 27 saved; regenerate via dossier_signal_pipeline.py).
