# NMP pure z-round-trip vs R-trigger (Feb 2024)

USD/point = 2.38 (from baseline)  |  exit at |z|<= 0.4752, enter |z|>= 1.8481

| metric | **z-round-trip** | R-trigger baseline |
|---|---|---|
| trades | 1526 | 1860 |
| trades/day | 73 | 89 |
| $/trade mean | -0.59 | -4.75 |
| $/trade median | +1.79 | -5.00 |
| PF-based Trade WR | -0.087 (PF 0.91) | -0.730 (PF 0.27) |
| $/day mean | -43 | -421 |
| $/day 95% day-block CI | [-144, +54] | [-469, -378] |
| winning days | 10/21 | 0/21 |
| avg hold (min) | 4.5 | ~short (R-stop) |
| EOD-timeout exits | 0% | n/a |

Verdict: BEATS the R-trigger baseline on $/day (-43 vs -421). Still INCONCLUSIVE (CI includes 0).
CAVEAT: z reverts partly because the mean slides to price (trend) -> 'reverted in z, lost in price'.