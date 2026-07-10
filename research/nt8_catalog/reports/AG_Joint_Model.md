# Joint Bayesian Model (Logistic Regression)
**Total Events Trained:** 82102
**Base Rate (Intercept implied):** 0.6072

## 1. Feature Coefficients (Conditioned Weights)
| Feature | Coefficient | Odds Ratio |
|---|---|---|
| vwap_state | 0.0451 | 1.0461 |
| apz_state | 0.0423 | 1.0432 |
| sqz_state | 2.5682 | 13.0422 |
| can_state | 0.0093 | 1.0093 |
| ma_state | 0.0471 | 1.0482 |
| Intercept | 0.0204 | 1.0206 |

## 2. Posterior Tier Separation (Calibration)
We bucket the events by their predicted posterior probability to see if confluence generates lift.
| Tier (Percentile) | N | Mean Posterior | Actual Win Rate | Delta vs Base |
|---|---|---|---|---|
| (0.4797, 0.4938] | 8682 | 0.4925 | 0.4916 | -11.56 pp |
| (0.4938, 0.4945] | 9175 | 0.4945 | 0.5154 | -9.18 pp |
| (0.4945, 0.5028] | 11655 | 0.5024 | 0.4841 | -12.31 pp |
| (0.5028, 0.5074] | 15026 | 0.5070 | 0.4945 | -11.26 pp |
| (0.5074, 0.5157] | 8955 | 0.5155 | 0.5357 | -7.15 pp |
| (0.5157, 0.5164] | 4785 | 0.5164 | 0.5325 | -7.47 pp |
| (0.5164, 0.9301] | 21687 | 0.8522 | 0.8702 | +26.30 pp |
| (0.9301, 0.9337] | 2137 | 0.9313 | 0.7323 | +12.52 pp |

## 3. Verdict
If the top tier exhibits a Real > +10pp lift over the base rate, confluence provides a tradable edge.