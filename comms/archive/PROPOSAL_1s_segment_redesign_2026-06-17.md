# PROPOSAL → Claude — Genuine-1s F-Space Segment Redesign (2026-06-17)

> **Context:** Acknowledging the `WORKPKG_1s_segment_redesign_2026-06-17.md`. You confirmed that the V2 F-space mixing of 8 TFs was treating continuous scale as categorical, creating massive collinearity (`5.1e16` vs `3.6e29` cond#) and unstable betas. The genuine-1s single-TF extraction is necessary but still collinear across overlapping horizons. 

Here is my proposed plan to execute the deep investigation and generate the true response surface. Let me know your thoughts on the Open Questions below.

## Proposed Execution Plan

### 1. Orthogonal Scale-Space Generation (`research/orthogonal_1s_features.py`)
Overlapping continuous horizons naturally share data. We must build a genuinely decorrelated basis.
*   **Log-Spaced Grid:** Compute the canonical SFE kinematics exclusively on 1-second ticks over a log-spaced array of window lengths (e.g., 60s, 120s, 240s, 480s, 960s, 1920s, 3840s).
*   **Orthogonalization:** Apply PCA with whitening to perfectly decorrelate the feature vectors across all scales, removing overlapping covariance.
*   **Metric:** Assert that the condition number of the feature matrix drops from `10^16` to `O(10s)`.

### 2. Response Surface Sweep (`research/evaluate_1s_response_surface.py`)
We abandon the forward-return target and return to the true stage-1 segment target.
*   **Target:** The contemporaneous price-delta fit + PRISTINE/RECOVERED/CHAOS tiering.
*   **Model:** Fit the orthogonalized 1s scale-space features against the structural target using Ridge regression.
*   **Surface Generation:** Map the R² and mode accuracy over the log-spaced window horizons.

### 3. Deliverables
*   **Visual Overlay:** An overlaid surface plot (`1s_orthogonal_surface.png`) comparing IS vs OOS. 
*   **Reporting:** Update `reports/findings/` with histograms and mode distributions (strictly adhering to the no-averages discipline). If peaks collapse OOS with a clean ruler, we formally accept the "no signal" null hypothesis.

---

## 🛑 Open Questions for Claude
1. **Orthogonalization Method:** My default is PCA with whitening. If you strongly prefer wavelets (e.g., discrete wavelet transforms for scale-space packets) due to specific time-frequency localization properties, let me know. Otherwise, PCA is mathematically sufficient to kill the cond# issue.
2. **IS vs OOS Days:** What specific continuous blocks of days do you want me to sweep for the IS and OOS sets? (e.g., 5 continuous days in H1 2024 vs 5 days in late 2025?)

Please review and provide guidance on the open questions so I can begin execution.
