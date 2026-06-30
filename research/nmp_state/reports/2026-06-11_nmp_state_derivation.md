# NMP State Derivation Validation Report
Generated on 2026-06-11

**Days analyzed**: 2025_01_01, 2025_03_06, 2025_05_20, 2025_08_01, 2025_10_15

## 1. Parity (executed)
- lambda_hat slope vs np.polyfit (200 random windows): max |err| = 3.89e-16 -> **PASS**
- lambda_se vs np.polyfit cov: max |err| = 1.80e-16 -> **PASS**
- vr rolling vs brute-force (500-bar synthetic): max |err| = 1.53e-12 -> **PASS**

## 2. Threshold Recalibration
Matching quantile for `P(|z_21| > 2.0)` (7.9972%):
- **Z* (entry)** for `|z_15|` = **1.8481**

Matching quantile for `P(|z_21| < 0.5)` (27.8337%):
- **Z* (exit)** for `|z_15|` = **0.4752**

## 3. $\hat{\lambda}$ Null Calibration
Distribution of t-stat across different k values:

| TF | k | mean | std | 5% | 95% | Propose Abstain Band |
|---|---|---|---|---|---|---|
| 1m | 12 | -0.02 | 1.43 | -2.38 | 2.23 | [-2.0, 2.0] |
| 1m | 21 | 0.00 | 1.32 | -2.23 | 2.08 | [-2.0, 2.0] |
| 1m | 30 | 0.01 | 1.25 | -2.02 | 2.02 | [-2.0, 2.0] |
| 5m | 12 | 0.01 | 1.19 | -1.86 | 1.83 | [-2.0, 2.0] |
| 5m | 21 | -0.03 | 0.99 | -1.69 | 1.56 | [-2.0, 2.0] |
| 5m | 30 | -0.04 | 1.04 | -1.67 | 1.68 | [-2.0, 2.0] |

*Proposed abstain band is based on standard t-stat significance (~95% confidence bounds)*

## 4. `vr` exact vs proxy correlation
| TF | Proxy Pair | Spearman | Status |
|---|---|---|---|
| 5s | sigma_5s / sigma_slow | 0.636 | FAIL |
| 15s | sigma_15s / sigma_slow | 0.572 | FAIL |
| 1m | sigma_1m / sigma_slow | 0.419 | FAIL |
| 5m | sigma_5m / sigma_slow | 0.730 | FAIL |
| 15m | sigma_15m / sigma_slow | 0.384 | FAIL |


## 5. Trigger-rate parity
- **V1 exact trigger rate** (`|z_21| > 2.0 AND vr_exact < 1.0`): **7.1133%**
- **V2 scaled trigger rate (exact vr)** (`|z_15| > 1.8481 AND vr_exact < 1.0`): **7.5146%**
- **V2 scaled trigger rate (proxy vr)** (`|z_15| > 1.8481 AND vr_proxy < 1.0`): **8.0544%**

## Recommendation
Quantile-matched $Z^*$ on $z_{15}$ is recommended to maintain alignment with the V2 185D schema logic. Depending on the `vr_proxy` correlation status, `vr_exact` from raw closes might be required instead of the `price_sigma_w` ratio.