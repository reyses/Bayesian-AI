# Tier day rule backtest — RIDE_AGAINST

Threshold: BLEED = tier PnL <= -$200.0, HARVEST = tier PnL >= +$200.0.

Z-scores and decision thresholds are calibrated on IS. OOS uses the same IS normalization and the same IS-calibrated threshold (honest walk-forward).

**Rule A** is live-readable: three rolling market-state features computable at any bar. Useful as morning kill switch and intraday re-check.

**Rule B** adds `day_entry_range` — only measurable once a day has accumulated trades. Evaluable intraday, not at open.

## Rule A — 3 live-readable features

Features (name, sign):
  - `+z(mean_5m_variance_ratio)`
  - `-z(mean_1h_variance_ratio)`
  - `+z(mean_1h_z_range)`

Baseline RIDE_AGAINST PnL IS: $-16,468 across 273 days.
Baseline RIDE_AGAINST PnL OOS: $+6,476 across 66 days.

Bleed days (<-$200) IS: 92 | OOS: 15
Harvest days (>+$200) IS: 76 | OOS: 25

### IS sweep — threshold calibrated on IS

| Top-X% skip | Days skipped | Bleed caught | Bleed catch % | Harvest FP | Harv FP % | $ on skipped | $ delta vs baseline | Threshold score |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 10% | 27 | 14/92 | 15% | 3/76 | 4% | $-6,032 | $+6,032 | +2.61 |
| 15% | 41 | 20/92 | 22% | 6/76 | 8% | $-9,712 | $+9,712 | +1.91 |
| 20% | 55 | 24/92 | 26% | 9/76 | 12% | $-11,722 | $+11,722 | +1.48 |
| 25% | 68 | 31/92 | 34% | 12/76 | 16% | $-12,928 | $+12,928 | +1.29 |
| 30% | 82 | 38/92 | 41% | 13/76 | 17% | $-18,514 | $+18,514 | +1.02 |
| 35% | 96 | 43/92 | 47% | 16/76 | 21% | $-19,907 | $+19,907 | +0.85 |
| 40% | 109 | 47/92 | 51% | 21/76 | 28% | $-23,035 | $+23,035 | +0.53 |
| 45% | 123 | 53/92 | 58% | 22/76 | 29% | $-25,364 | $+25,364 | +0.29 |
| 50% | 136 | 58/92 | 63% | 26/76 | 34% | $-27,162 | $+27,162 | +0.13 |

### OOS — IS-calibrated threshold frozen

IS-calibrated threshold applied to OOS (no OOS tuning).

| IS Top-X% | IS threshold | OOS days skipped | OOS bleed caught | OOS Bleed catch % | OOS harvest FP | OOS FP % | OOS $ on skipped | OOS $ delta vs baseline |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 10% | +2.61 | 8 | 1/15 | 7% | 0/25 | 0% | $-1,780 | $+1,780 |
| 15% | +1.91 | 12 | 3/15 | 20% | 1/25 | 4% | $-1,265 | $+1,265 |
| 20% | +1.48 | 17 | 4/15 | 27% | 2/25 | 8% | $-2,116 | $+2,116 |
| 25% | +1.29 | 17 | 4/15 | 27% | 2/25 | 8% | $-2,116 | $+2,116 |
| 30% | +1.02 | 19 | 6/15 | 40% | 2/25 | 8% | $-3,045 | $+3,045 |
| 35% | +0.85 | 19 | 6/15 | 40% | 2/25 | 8% | $-3,045 | $+3,045 |
| 40% | +0.53 | 22 | 6/15 | 40% | 4/25 | 16% | $-2,006 | $+2,006 |
| 45% | +0.29 | 24 | 7/15 | 47% | 5/25 | 20% | $-1,908 | $+1,908 |
| 50% | +0.13 | 24 | 7/15 | 47% | 5/25 | 20% | $-1,908 | $+1,908 |

## Rule B — 4 features incl. intraday entry range

Features (name, sign):
  - `+z(mean_5m_variance_ratio)`
  - `-z(mean_1h_variance_ratio)`
  - `+z(mean_1h_z_range)`
  - `-z(day_entry_range)`

Baseline RIDE_AGAINST PnL IS: $-16,468 across 273 days.
Baseline RIDE_AGAINST PnL OOS: $+6,476 across 66 days.

Bleed days (<-$200) IS: 92 | OOS: 15
Harvest days (>+$200) IS: 76 | OOS: 25

### IS sweep — threshold calibrated on IS

| Top-X% skip | Days skipped | Bleed caught | Bleed catch % | Harvest FP | Harv FP % | $ on skipped | $ delta vs baseline | Threshold score |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 10% | 27 | 12/92 | 13% | 3/76 | 4% | $-5,374 | $+5,374 | +2.75 |
| 15% | 41 | 17/92 | 18% | 7/76 | 9% | $-7,681 | $+7,681 | +2.41 |
| 20% | 55 | 25/92 | 27% | 7/76 | 9% | $-13,001 | $+13,001 | +1.95 |
| 25% | 68 | 29/92 | 32% | 10/76 | 13% | $-13,648 | $+13,648 | +1.63 |
| 30% | 82 | 34/92 | 37% | 11/76 | 14% | $-16,381 | $+16,381 | +1.29 |
| 35% | 96 | 38/92 | 41% | 15/76 | 20% | $-15,286 | $+15,286 | +0.98 |
| 40% | 109 | 42/92 | 46% | 19/76 | 25% | $-16,608 | $+16,608 | +0.58 |
| 45% | 123 | 48/92 | 52% | 23/76 | 30% | $-17,920 | $+17,920 | +0.29 |
| 50% | 136 | 53/92 | 58% | 27/76 | 36% | $-20,613 | $+20,613 | -0.00 |

### OOS — IS-calibrated threshold frozen

IS-calibrated threshold applied to OOS (no OOS tuning).

| IS Top-X% | IS threshold | OOS days skipped | OOS bleed caught | OOS Bleed catch % | OOS harvest FP | OOS FP % | OOS $ on skipped | OOS $ delta vs baseline |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 10% | +2.75 | 8 | 2/15 | 13% | 0/25 | 0% | $-2,128 | $+2,128 |
| 15% | +2.41 | 10 | 3/15 | 20% | 0/25 | 0% | $-2,378 | $+2,378 |
| 20% | +1.95 | 14 | 3/15 | 20% | 1/25 | 4% | $-1,573 | $+1,573 |
| 25% | +1.63 | 15 | 4/15 | 27% | 1/25 | 4% | $-3,227 | $+3,227 |
| 30% | +1.29 | 17 | 4/15 | 27% | 2/25 | 8% | $-2,116 | $+2,116 |
| 35% | +0.98 | 19 | 5/15 | 33% | 2/25 | 8% | $-2,813 | $+2,813 |
| 40% | +0.58 | 22 | 5/15 | 33% | 4/25 | 16% | $-1,628 | $+1,628 |
| 45% | +0.29 | 23 | 6/15 | 40% | 4/25 | 16% | $-1,926 | $+1,926 |
| 50% | -0.00 | 29 | 6/15 | 40% | 7/25 | 28% | $-885 | $+885 |
