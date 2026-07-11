# Genuine-1s F-Space Segment Redesign: Findings Report

## 1. Executive Summary
The attempt to salvage the 1s F-Space "segment" concept using Gram-Schmidt orthogonalization was mathematically successful but empirically fatal.

While we completely cured the multi-collinearity that plagued the raw representation, the resulting causal evaluation explicitly proves that **the 1s feature space possesses zero forward predictive power.** The previous illusion of success was entirely an artifact of non-causal tiering overlapping with a trivial contemporaneous identity.

**Verdict: The 1-second continuous F-Space thesis is DEAD.** The mathematical representation of segments using L2/L3 kinematics provides no tradeable edge.

## 2. Methodology

Per the agreed-upon orthogonalization plan:
1. **Gram-Schmidt Deflation:** We implemented block-wise Gram-Schmidt orthogonalization by strictly increasing window sizes ($N=60$ through $N=3840$). 
2. **Surface Extraction (Marginal):** To preserve horizon identity, we extracted the raw `L2(N)` and `L3(N)` features for each scale.
3. **Dual-Target Evaluation:** We evaluated the feature spaces against two targets:
   * **Target A (Contemporaneous - Diagnostic):** $y = P_t - P_{t-N}$
   * **Target B (Causal Forward - Tradeable):** $y = P_{t+N} - P_t$
4. **Validation Schema:** Ridge Regression (5-Fold CV) scored via $R^2$, executed across a 40-day scale-up covering IS (H1 2024) and three OOS blocks (2025-2026).

## 3. Results & Findings

### The Collinearity Cure
Gram-Schmidt worked flawlessly. By deflating smaller horizons from larger horizons, the condition number of the feature block collapsed from an astronomical $O(10^{19})$ to a pristine $O(100)$ (e.g., $149 - 182$). The collinearity that broke previous models was completely neutralized.

### Target A: The Diagnostic Illusion (Contemporaneous Fit)
Across all 40 days and all 7 horizons, the 5-CV $R^2$ against the contemporaneous price delta was **exactly 1.0**. 

**Why this matters:** The features (price speed, accel, jerk) are mathematically derived from the exact same price history as the target delta. The structural "tiering" of a segment (Pristine vs Chaos) is simply a convoluted measurement of the past price delta. The segment perfectly describes *what just happened*.

*(Note: Surface A is identically 1.0 across the entire horizon spectrum)*

### Target B: The Tradeable Reality (Causal Forward Fit)
When the exact same feature spaces are tasked with predicting the forward price delta (Target B), the results are catastrophic.

Across **every single horizon, on every single day, in both IS and OOS blocks**, the 5-CV $R^2$ is **strictly negative**.

| Block | Horizon | Mean $R^2$ | Max $R^2$ | Min $R^2$ |
| :--- | :--- | :--- | :--- | :--- |
| IS_H1_2024 | All | < 0 | < 0 | < 0 |
| OOS1_H1_2025 | All | < 0 | < 0 | < 0 |
| OOS2_H2_2025 | All | < 0 | < 0 | < 0 |
| OOS3_2026 | All | < 0 | < 0 | < 0 |

A negative $R^2$ indicates that the model performs worse than simply predicting a mean of zero. There is no predictive signal hidden in the orthogonalized 1s kinematics.

## 4. Conclusion and Next Steps

The segment redesign package has fulfilled its purpose by definitively proving the null hypothesis.

1. **Abandon the Continuous 1s Filter:** We will formally retire the 1s F-Space continuous representation. The kinematic derivatives (speed/accel) do not precede price movement; they merely reflect it.
2. **Pivot to the Geometric/Kinematic Entry:** Our remaining edge lies entirely in the *discrete* `Orange Kalman` architecture. The entry logic defined in the `blue_line` geometric inflection framework (as visualized in previous walkthroughs) is causal and robust.
3. **Proceed to Geometric Rollover Exit:** As per our implementation plan, we will now abandon continuous segment trailing stops and implement the **Causal Geometric Exit** (Acceleration Flip, Velocity Decay, or Dual-Timeline Pinch) for the Kalman strategy.

Claude, the results are conclusive. Gram-Schmidt worked to fix the math, but the underlying physical phenomenon is dead. Let's move on to the Causal Geometric Exit for the discrete entries. Let me know your thoughts so we can tackle the next item in the backlog.
