# Bucket manifest - axis: `dow`

_Generated 2026-05-09T22:44:41.129551_

Total days indexed: 348

## Cells

| cell | n_days | n_IS | n_OOS | %total | rep_day | eff_ratio | net_move | range |
|------|-------:|-----:|------:|-------:|---------|----------:|---------:|------:|
| Mon | 57 | 35 | 11 | 16.4% | 2026_02_09 | +0.029 | +190.0 | 473.5 |
| Tue | 58 | 35 | 11 | 16.7% | 2025_10_07 | +0.025 | -109.2 | 289.5 |
| Wed | 59 | 36 | 12 | 17.0% | 2025_07_23 | +0.024 | +92.0 | 280.0 |
| Thu | 60 | 35 | 13 | 17.2% | 2025_03_20 | +0.022 | -154.0 | 368.0 |
| Fri | 59 | 34 | 13 | 17.0% | 2025_01_10 | +0.045 | -272.8 | 497.8 |

## Notes

- Representative day = day in cell whose `efficiency_ratio` is
  closest to the cell median.
- `eff_ratio`, `net_move`, `range` are the day-aggregate metrics from
  `DATA/ATLAS/regime_labels_2d.csv`.
- Cells with `n_days < ~10` are too thin for reliable conditioning;
  flag them and pool with the parent cell or the FLAT/CHOPPY default.
