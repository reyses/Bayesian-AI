# Joint Confluence: Mechanical EDA
This report explicitly counts conditional intersections of indicators to evaluate confluence without black-box ML models.

## 1. Confluence Impact Table
| Base Signal | Confluence Signal | Total Base Events | Base Win Rate | Confluence Events (N) | Confluence Win Rate | Lift (pp) |
|---|---|---|---|---|---|---|
| VWAP | + APZ | 12143 | 0.6199 | 1020 | 0.5892 | -3.06 pp |
| VWAP | + SQZ | 12143 | 0.6199 | 380 | 0.5974 | -2.25 pp |
| VWAP | + CAN | 12143 | 0.6199 | 363 | 0.6584 | +3.85 pp |
| VWAP | + MA | 12143 | 0.6199 | 769 | 0.6216 | +0.17 pp |
| APZ | + VWAP | 19625 | 0.6268 | 1020 | 0.5892 | -3.75 pp |
| APZ | + SQZ | 19625 | 0.6268 | 26 | 0.6923 | +6.56 pp |
| APZ | + CAN | 19625 | 0.6268 | 448 | 0.6295 | +0.27 pp |
| APZ | + MA | 19625 | 0.6268 | 844 | 0.6386 | +1.19 pp |
| SQZ | + VWAP | 17727 | 0.9808 | 380 | 0.9737 | -0.71 pp |
| SQZ | + APZ | 17727 | 0.9808 | 26 | 0.9615 | -1.93 pp |
| SQZ | + CAN | 17727 | 0.9808 | 1462 | 0.9891 | +0.82 pp |
| SQZ | + MA | 17727 | 0.9808 | 115 | 0.9913 | +1.05 pp |
| CAN | + VWAP | 26021 | 0.6244 | 363 | 0.6446 | +2.02 pp |
| CAN | + APZ | 26021 | 0.6244 | 448 | 0.5982 | -2.62 pp |
| CAN | + SQZ | 26021 | 0.6244 | 1462 | 0.6354 | +1.10 pp |
| CAN | + MA | 26021 | 0.6244 | 462 | 0.6364 | +1.19 pp |
| MA | + VWAP | 6586 | 0.5935 | 769 | 0.5995 | +0.59 pp |
| MA | + APZ | 6586 | 0.5935 | 844 | 0.5367 | -5.68 pp |
| MA | + SQZ | 6586 | 0.5935 | 115 | 0.6348 | +4.13 pp |
| MA | + CAN | 6586 | 0.5935 | 462 | 0.6147 | +2.12 pp |