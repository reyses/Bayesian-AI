# Cross-TF nesting analysis - Cross-TF nesting - Full 14 months  (1W: 20 | 1D: 78 | 4h: 437 | 1h: 610 | 15m: 3087)

Generated: 2026-04-29 22:22

## Inputs

| TF | Peaks | Segments | Source |
|---|---:|---:|---|
| 1W | 20 | 19 | human_peaks_2025-01-03_to_2026-03-20_1W.json |
| 1D | 78 | 77 | human_peaks_2025-01-01_to_2026-03-20_1D.json |
| 4h | 437 | 436 | auto_peaks_2025-01-02_to_2026-03-20_4h.json |
| 1h | 610 | 609 | auto_peaks_2025-01-01_to_2026-03-20_1h.json |
| 15m | 3087 | 3086 | auto_peaks_2025-01-01_to_2026-03-20_15m.json |

## Nesting summary

Mean number of child-TF peaks inside each parent-TF segment:

| Parent | Child | Parent Segs | Child Peaks Total | Mean / parent | Median / parent | Max / parent | Empty parents |
|---|---|---:|---:|---:|---:|---:|---:|
| 1W | 1D | 19 | 81 | 4.3 | 3 | 22 | 0 |
| 1W | 4h | 19 | 423 | 22.3 | 17 | 124 | 0 |
| 1W | 1h | 19 | 584 | 30.7 | 22 | 175 | 0 |
| 1W | 15m | 19 | 2948 | 155.2 | 105 | 891 | 0 |
| 1D | 4h | 77 | 448 | 5.8 | 5 | 23 | 0 |
| 1D | 1h | 77 | 614 | 8.0 | 6 | 33 | 0 |
| 1D | 15m | 77 | 3065 | 39.8 | 28 | 177 | 0 |
| 4h | 1h | 436 | 667 | 1.5 | 1 | 7 | 29 |
| 4h | 15m | 436 | 3144 | 7.2 | 6 | 34 | 0 |
| 1h | 15m | 609 | 3156 | 5.2 | 4 | 24 | 6 |

## Interpretation

- If child peaks/parent stays roughly constant per parent direction, lower TFs nest cleanly inside higher TFs.
- High variance in peaks/parent means low TFs encode independent structure, not just sub-noise of high TFs.
- Empty parents = parent segments that have NO child peaks (very rare; would indicate a marker miss).

## Files

- Chart: `reports/findings/regime_eda\2026-04-29_cross_tf_nesting_full_14mo.png`
- CSV: `reports/findings/regime_eda\2026-04-29_cross_tf_nesting_full_14mo.csv`
