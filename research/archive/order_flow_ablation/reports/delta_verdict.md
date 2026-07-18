# Order Flow Ablation Study Verdict

This document summarizes the findings from the Phase 2 Marginal-Value Ablation Study on the proposed order flow delta features (`cum_delta`, `divergence`, `delta`). The objective was to determine if reconstructing order flow data via a proxy model is worthwhile for the 2024 dataset, based on the predictive edge of these features on the 2025 dataset.

## 1. Ablation Results

A strictly temporal walk-forward split (first 66% train, last 34% test) was used to evaluate a 30-minute forward return prediction using XGBoost.

| Model Variant | 30m Forward Return R² | Difference |
| :--- | :--- | :--- |
| **Baseline (416D)** | -0.1860 | - |
| **Baseline + Delta** | -0.3396 | **-0.1537** (Degraded) |

> [!WARNING]
> The addition of order flow delta features significantly degraded out-of-sample R² performance.

## 2. Feature Importance

The delta features failed to rank among the top predictors in the combined model. The top 10 features were entirely dominated by existing structural features from the baseline 416D grid:

| Rank | Feature | Importance |
| :--- | :--- | :--- |
| 1 | `F_35` (Baseline) | 0.0839 |
| 2 | `F_31` (Baseline) | 0.0208 |
| 3 | `F_187` (Baseline) | 0.0207 |
| 4 | `F_55` (Baseline) | 0.0177 |
| 5 | `F_36` (Baseline) | 0.0165 |
| 6 | `F_37` (Baseline) | 0.0164 |
| 7 | `F_34` (Baseline) | 0.0155 |
| 8 | `F_30` (Baseline) | 0.0150 |
| 9 | `F_6` (Baseline) | 0.0149 |
| 10 | `F_106` (Baseline) | 0.0146 |

*(Note: `cum_delta` and `divergence` did not breach the top 10).*

## 3. Sub-Period Stability

A rolling 4-week sub-period stability check was performed on the test set to evaluate if the proxy degraded over time. The results demonstrate consistent failure across all out-of-sample periods:

| Period (4-Week Chunk) | Sample Size (N) | R² Score |
| :--- | :--- | :--- |
| 2025-11-16 | 17,165 | -0.5829 |
| 2025-12-14 | 280,219 | -0.2400 |
| 2026-01-11 | 75,990 | -0.3392 |
| 2026-02-08 | 138,151 | -0.4961 |

## Conclusion: Purchase Gate FAILED

The ablation study confirms that order flow delta features provide **zero marginal predictive value** over the existing 416D feature set. In fact, their inclusion increases overfitting, actively harming out-of-sample generalization. 

Consequently, the purchase gate has **FAILED**. 
- We will **not** purchase the 2024 Databento MBO data.
- We will **not** proceed with building or deploying the Delta-Reconstruction proxy, as there is no edge to capture.

*(Note: While 2025 roll-week gaps were briefly evaluated, the conclusive failure of the delta features renders a free re-stitch moot for this feature set.)*
