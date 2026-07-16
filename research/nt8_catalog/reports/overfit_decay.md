# Overfit-decay shelf-life sweep (doc 075 standard)

Pooled combiner (same design as combiner_preview.py: BASE + consensus + per-stream one-hots, GLOBAL det list) fit on rolling training windows, evaluated weekly OOS. shelf_life_weeks = first qualifying eval week where a 3-week rolling mean of edge (AUC-0.5) drops below 70% of initial_edge (mean of first 2 qualifying eval weeks). Right-censored at the available horizon if it never crosses.

- Pool: 713926 fires, 38 streams, 2024-01-02 .. 2026-03-19
- Feature columns (43): BASE=['pivot_age_min', 'sig_with_leg', 'tod', 'inter'] + consensus + is_<det> x 38 streams
- Window starts: every 4 weeks from 2024-01-01 through 2025-12-31, 27 candidate starts per training-length pass
- Skip rule: training windows with < 5000 fires skipped (8wk pass: 0 skipped; 16wk pass: 0 skipped); eval weeks with < 500 fires skipped (see per-window skip counts in the run log)
- Runtime guard: fit-time budget 120s/window; never triggered (all fits stayed under budget; no window was subsampled)

## Pass 1 — 8-week training windows

| window start | N_train | initial_edge | shelf_life_weeks |
|---|---|---|---|
| 2024-01-01 | 43024 | 0.2169 | 81 |
| 2024-01-29 | 45908 | 0.1869 | censored @ 100 |
| 2024-02-26 | 46624 | 0.2161 | 73 |
| 2024-03-25 | 46088 | 0.2374 | 57 |
| 2024-04-22 | 44836 | 0.2045 | 65 |
| 2024-05-20 | 43331 | 0.2109 | 61 |
| 2024-06-17 | 53202 | 0.1973 | censored @ 80 |
| 2024-07-15 | 57627 | 0.1899 | censored @ 76 |
| 2024-08-12 | 52893 | 0.1653 | censored @ 72 |
| 2024-09-09 | 51838 | 0.2154 | 45 |
| 2024-10-07 | 48093 | 0.2168 | 41 |
| 2024-11-04 | 45678 | 0.2311 | 26 |
| 2024-12-02 | 45358 | 0.1989 | 33 |
| 2024-12-30 | 48755 | 0.2050 | 18 |
| 2025-01-27 | 59441 | 0.2156 | 14 |
| 2025-02-24 | 63186 | 0.2001 | 22 |
| 2025-03-24 | 58647 | 0.2155 | 7 |
| 2025-04-21 | 54766 | 0.1408 | censored @ 37 |
| 2025-05-19 | 39608 | 0.1988 | 11 |
| 2025-06-16 | 37908 | 0.1947 | censored @ 30 |
| 2025-07-14 | 48057 | 0.1487 | censored @ 26 |
| 2025-08-11 | 37606 | 0.2012 | censored @ 23 |
| 2025-09-08 | 41359 | 0.1710 | censored @ 19 |
| 2025-10-06 | 58627 | 0.1834 | censored @ 15 |
| 2025-11-03 | 51037 | 0.1563 | censored @ 12 |
| 2025-12-01 | 40937 | 0.1676 | censored @ 8 |
| 2025-12-29 | 51610 | 0.1823 | censored @ 4 |

- Observed (uncensored) shelf-life: N=14 windows. MODE = 7 weeks, MEDIAN = 37.0 weeks.
- Censoring: 13 of 27 windows never crossed the threshold within their available eval horizon (right-censored: true shelf-life >= the reported horizon for those windows) [13 censored-with-positive-edge, 0 had non-positive initial_edge (decay undefined)].
- 0 of 27 windows had insufficient eval data (< 2 qualifying eval weeks available, usually windows starting near the end of the sweep range where the data horizon runs out) — excluded from MODE/MEDIAN.
- CAVEAT: MODE/MEDIAN above are computed over UNCENSORED windows only; since 13/27 windows are censored, the true population shelf-life is likely LONGER than these numbers suggest (naive underestimate, no survival-curve correction applied here per spec).

## Pass 2 — 16-week training windows (comparison)

| window start | N_train | initial_edge | shelf_life_weeks |
|---|---|---|---|
| 2024-01-01 | 89648 | 0.2174 | 73 |
| 2024-01-29 | 91996 | 0.2391 | 57 |
| 2024-02-26 | 91460 | 0.2058 | 65 |
| 2024-03-25 | 89419 | 0.2110 | 61 |
| 2024-04-22 | 98038 | 0.1995 | 57 |
| 2024-05-20 | 100958 | 0.1910 | censored @ 76 |
| 2024-06-17 | 106095 | 0.1667 | censored @ 72 |
| 2024-07-15 | 109465 | 0.2125 | 45 |
| 2024-08-12 | 100986 | 0.2161 | 41 |
| 2024-09-09 | 97516 | 0.2311 | 26 |
| 2024-10-07 | 93451 | 0.1971 | censored @ 56 |
| 2024-11-04 | 94433 | 0.2123 | 29 |
| 2024-12-02 | 104799 | 0.2130 | 14 |
| 2024-12-30 | 111941 | 0.1978 | 22 |
| 2025-01-27 | 118088 | 0.2169 | 7 |
| 2025-02-24 | 117952 | 0.1360 | censored @ 37 |
| 2025-03-24 | 98255 | 0.2048 | 11 |
| 2025-04-21 | 92674 | 0.1946 | censored @ 30 |
| 2025-05-19 | 87665 | 0.1472 | censored @ 26 |
| 2025-06-16 | 75514 | 0.2023 | censored @ 23 |
| 2025-07-14 | 89416 | 0.1741 | censored @ 19 |
| 2025-08-11 | 96233 | 0.1848 | censored @ 15 |
| 2025-09-08 | 92396 | 0.1549 | censored @ 12 |
| 2025-10-06 | 99564 | 0.1683 | censored @ 8 |
| 2025-11-03 | 102647 | 0.1855 | censored @ 4 |
| 2025-12-01 | 101934 | n/a | insufficient eval data (only 0 qualifying wk) |
| 2025-12-29 | 81747 | n/a | insufficient eval data (only 0 qualifying wk) |

- Observed (uncensored) shelf-life: N=13 windows. MODE = 57 weeks, MEDIAN = 41.0 weeks.
- Censoring: 12 of 27 windows never crossed the threshold within their available eval horizon (right-censored: true shelf-life >= the reported horizon for those windows) [12 censored-with-positive-edge, 0 had non-positive initial_edge (decay undefined)].
- 2 of 27 windows had insufficient eval data (< 2 qualifying eval weeks available, usually windows starting near the end of the sweep range where the data horizon runs out) — excluded from MODE/MEDIAN.
- CAVEAT: MODE/MEDIAN above are computed over UNCENSORED windows only; since 12/27 windows are censored, the true population shelf-life is likely LONGER than these numbers suggest (naive underestimate, no survival-curve correction applied here per spec).

## Files

- Raw log: `nt8_catalog\reports\overfit_decay_run.log`
- Per-window/eval-week rows: `nt8_catalog\reports\overfit_decay_rows.parquet`