# Dossier signal league — direction agreement with AI labels (COMPLETE, 39 streams)
(train 2024, test 2025+26, day-block bootstrap CIs; baseline 0.50; assembled 2026-07-16.
Full 37-stream table: git 0ad35951 version of this file + docs 081/083/084/085.
This snapshot adds the template-era pattern layer, doc 086.)

## Template-era pattern layer (PTRN-*, from the resurrected K-means template engine's
## event vocabulary — core/pattern_utils.py @09cd30d8, formulas verbatim)
| stream | N | OOS AUC | test base | P-terciles low / mid / high |
|---|---|---|---|---|
| PTRN-ENGULF | 29,917 | 0.616 | 0.62 | 0.51 [0.50,0.53] / 0.63 / 0.72 [0.71,0.74] |
| PTRN-HAMMER | 4,484 | 0.615 | 0.55 | 0.43 [0.39,0.47] / 0.57 / 0.65 [0.61,0.68] |

- ENGULF: direction in the formula (bull=long/bear=short); aligned 0.62 — an
  engulfing bar is a continuation event: the ride>fade law again.
- HAMMER: classic bullish reading (declared adaptation; legacy used patterns as
  state flags). Mild alignment 0.55, gradable 0.43→0.65.
- DOJI: skipped as a directional stream (no direction — would be fabrication);
  candidate as a combiner FEATURE later. Geometric patterns (COMPRESSION/WEDGE/
  BREAKDOWN) + the full 16-D K-means template engine: see doc 086 (resurrection
  proposal — the 2024-frozen template stream).

## Top of the full league (39 graded+thin streams; complete table in git history)
| stream | N | OOS AUC | test base | headline |
|---|---|---|---|---|
| PIVOT-16 | 324 | 0.939 | 0.05 | pure inverter |
| OHLC-01 | 619 | 0.841 | 0.48 | both tails actionable |
| VP-01 | 283 | 0.732 | 0.36 | low tercile 0.12 |
| VWMA-10 | 540 | 0.714 | 0.63 | — |
| NMPT-FADECALM | 21,034 | 0.676 | 0.42 | inv low tercile 75% |
| ADX-08 | 1,359 | 0.660 | — | doc 079 |
| NMPT-RIDEAGN | 20,690 | 0.656 | 0.61 | high tercile 0.75 |
| NMP (corrected) | 10,388 | 0.639 | 0.27 | anti-aligned fade |
| NMPT-MTFBRK | 2,167 | 0.632 | 0.80 | best-aligned tier |
| NMPT-MTFEXH | 840 | 0.635 | 0.79 | — |
| ROUND-05 | 44,332 | 0.623 | 0.63 | — |
| SAR-23 | 37,184 | 0.618 | 0.44 | — |
| PTRN-ENGULF | 29,917 | 0.616 | 0.62 | template-era layer |
| PTRN-HAMMER | 4,484 | 0.615 | 0.55 | template-era layer |
| RENKO-24 | 198,560 | 0.611 | 0.55 | biggest stream |
| CURVE | 26,368 | 0.606 | 0.55 | causal labeler-cubic |
| NMP-LAMBDA | 10,793 | 0.574 | 0.54 | +28pp over fade |
| NMPT-FREIGHT | 4,575 | 0.582 | 0.75 | ride, aligned |
| ... | | | | (full table: git 0ad35951) |

## Structural laws (across all 39)
1. **Ride > fade, always**: every ride-mode stream is label-aligned (0.55-0.80);
   every fade/divergence premise is anti-aligned (0.01-0.43) and works only
   inverted or graded.
2. **Calibration holds at scale**: pooled combiner stays diagonal as streams are
   added (see combiner_preview.md).
3. **Timing beats identity for tautologies** (ZIGZAG/ORB: 0.96+ base, value is
   in the phase).
