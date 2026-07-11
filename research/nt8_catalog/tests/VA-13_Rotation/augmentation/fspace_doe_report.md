# Logistic Regression DOE: VA-13_Rotation
**Total Events:** 132
**Base Rate:** 0.1212
**ROC AUC:** 0.4467
**Log Loss:** 0.6179

## Magnitude Weighted Evaluation
> Weights applied during fit based on MFE (wins) and MAE (losses).
> **OOS Guard:** Probabilities generated via Stratified 5-Fold Cross-Validation.

| Tier | N | Mean Post. | Actual WR | Base Delta | Mean MFE | Mean MAE |
|---|---|---|---|---|---|---|
| (0.009399999999999999, 0.144] | 14 | 0.0726 | 0.1429 | +2.16 pp | 4.25 | -3.46 |
| (0.144, 0.214] | 13 | 0.1868 | 0.2308 | +10.96 pp | 17.19 | -3.40 |
| (0.214, 0.245] | 13 | 0.2303 | 0.0769 | -4.43 pp | 2.56 | -5.85 |
| (0.245, 0.303] | 13 | 0.2731 | 0.1538 | +3.26 pp | 11.25 | -5.42 |
| (0.303, 0.362] | 13 | 0.3258 | 0.1538 | +3.26 pp | 6.71 | -3.50 |
| (0.362, 0.403] | 13 | 0.3835 | 0.0000 | -12.12 pp | 1.73 | -5.67 |
| (0.403, 0.456] | 13 | 0.4283 | 0.0769 | -4.43 pp | -6.19 | -13.65 |
| (0.456, 0.538] | 13 | 0.4906 | 0.0769 | -4.43 pp | 5.00 | -4.38 |
| (0.538, 0.602] | 13 | 0.5676 | 0.2308 | +10.96 pp | 0.62 | -4.92 |
| (0.602, 0.937] | 14 | 0.7087 | 0.0714 | -4.98 pp | 7.04 | -5.18 |