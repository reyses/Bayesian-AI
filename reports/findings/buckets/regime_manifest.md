**DEPRECATED — LOOKAHEAD ARTIFACT, DO NOT QUOTE**

# Bucket manifest - axis: `regime`

_Generated 2026-05-09T22:46:25.945217_

Total days indexed: 348

## Cells

| cell | n_days | n_IS | n_OOS | %total | rep_day | eff_ratio | net_move | range |
|------|-------:|-----:|------:|-------:|---------|----------:|---------:|------:|
| UP_SMOOTH | 63 | 34 | 12 | 18.1% | 2025_11_16 | +0.074 | +170.5 | 306.0 |
| UP_CHOPPY | 23 | 15 | 4 | 6.6% | 2025_12_18 | +0.042 | +314.2 | 397.2 |
| DOWN_SMOOTH | 44 | 25 | 10 | 12.6% | 2025_05_18 | +0.067 | -125.5 | 232.0 |
| DOWN_CHOPPY | 19 | 12 | 5 | 5.5% | 2025_07_15 | +0.039 | -172.2 | 257.5 |
| FLAT_SMOOTH | 71 | 51 | 7 | 20.4% | 2025_12_01 | +0.014 | +79.8 | 286.8 |
| FLAT_CHOPPY | 128 | 71 | 33 | 36.8% | 2025_10_15 | +0.017 | +123.2 | 454.2 |

## Notes

- Representative day = day in cell whose `efficiency_ratio` is
  closest to the cell median.
- `eff_ratio`, `net_move`, `range` are the day-aggregate metrics from
  `DATA/ATLAS/regime_labels_2d.csv`.
- Cells with `n_days < ~10` are too thin for reliable conditioning;
  flag them and pool with the parent cell or the FLAT/CHOPPY default.
