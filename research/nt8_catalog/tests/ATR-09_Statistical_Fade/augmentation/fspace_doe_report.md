# Logistic Regression DOE: ATR-09_Statistical_Fade
**Total Events:** 151
**Base Rate:** 0.1258
**ROC AUC:** 0.6619
**Log Loss:** 0.5783

## Magnitude Weighted Evaluation
> Weights applied during fit based on MFE (wins) and MAE (losses).
> **OOS Guard:** Probabilities generated via Stratified 5-Fold Cross-Validation.

| Tier | N | Mean Post. | Actual WR | Base Delta | Mean MFE | Mean MAE |
|---|---|---|---|---|---|---|
| (0.014499999999999999, 0.0813] | 16 | 0.0522 | 0.0625 | -6.33 pp | 25.20 | -12.84 |
| (0.0813, 0.137] | 15 | 0.1109 | 0.0667 | -5.92 pp | 19.65 | -11.82 |
| (0.137, 0.193] | 15 | 0.1611 | 0.0667 | -5.92 pp | 8.33 | -11.33 |
| (0.193, 0.266] | 15 | 0.2190 | 0.0667 | -5.92 pp | 19.00 | -12.10 |
| (0.266, 0.344] | 15 | 0.2995 | 0.1333 | +0.75 pp | 17.13 | -11.30 |
| (0.344, 0.405] | 15 | 0.3761 | 0.0667 | -5.92 pp | 12.88 | -10.95 |
| (0.405, 0.465] | 15 | 0.4304 | 0.1333 | +0.75 pp | 18.32 | -11.60 |
| (0.465, 0.526] | 15 | 0.4954 | 0.2000 | +7.42 pp | 21.62 | -11.23 |
| (0.526, 0.699] | 15 | 0.6214 | 0.2667 | +14.08 pp | 24.55 | -10.03 |
| (0.699, 0.983] | 15 | 0.8319 | 0.2000 | +7.42 pp | 24.05 | -9.67 |