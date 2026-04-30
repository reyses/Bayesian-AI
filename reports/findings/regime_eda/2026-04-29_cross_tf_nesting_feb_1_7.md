# Cross-TF nesting analysis - Feb 1-7 OOS week

Generated: 2026-04-29 21:53

## Inputs (4 nested resolutions of human ground truth)

| TF | Peaks | Segments |
|---|---:|---:|
| 4h | 9 | 8 |
| 1h | 9 | 8 |
| 15m | 36 | 35 |
| 1m | 42 | 41 |

## Nesting summary

Mean number of child-TF peaks inside each parent-TF segment:

| Parent | Child | Parent Segs | Child Peaks Total | Mean / parent | Median / parent | Max / parent | Empty parents |
|---|---|---:|---:|---:|---:|---:|---:|
| 4h | 1h | 8 | 11 | 1.4 | 1 | 2 | 0 |
| 4h | 15m | 8 | 35 | 4.4 | 3 | 9 | 0 |
| 4h | 1m | 8 | 42 | 5.2 | 0 | 36 | 6 |
| 1h | 15m | 8 | 35 | 4.4 | 4 | 9 | 0 |
| 1h | 1m | 8 | 40 | 5.0 | 0 | 34 | 6 |
| 15m | 1m | 35 | 40 | 1.1 | 0 | 13 | 29 |

## Interpretation

- If child peaks/parent stays roughly constant per parent direction, lower TFs nest cleanly inside higher TFs.
- High variance in peaks/parent means low TFs encode independent structure, not just sub-noise of high TFs.
- Empty parents = parent segments that have NO child peaks (very rare; would indicate a marker miss).

## Files

- Chart: `reports/findings/regime_eda\2026-04-29_cross_tf_nesting_feb_1_7.png`
- CSV: `reports/findings/regime_eda\2026-04-29_cross_tf_nesting_feb_1_7.csv`
