# Wick Absorption Signal - Stage 1 Curve Regression

I've located the "Curve regression" logic in both the NT8 indicators (`2-CubicRegressionEndpoint_v1.0-RC.cs`) and the research artifacts (`nmp_pure_rcurve_1day.py`). 

## Findings
You are absolutely right. Rather than a center-aligned Savitzky-Golay filter, the canonical "Curve" is an **Endpoint Rolling OLS Polynomial Fit**. 
- In Python (`nmp_pure_rcurve_1day.py`), it was a 5-bar rolling **Quadratic** fit on 1m closes, triggering when instantaneous velocity (the derivative at the endpoint) crossed zero.
- In NT8 (`2-CubicRegressionEndpoint_v1.0-RC.cs`), it was a 450-bar (7.5 minute) rolling **Cubic** fit on 1s closes, evaluating value, slope, and curvature at the right edge.

## Proposed Changes

### [MODIFY] `research/wick_absorption_signal/pipeline/stage_1_profiling.py`
I will replace the artifact-inducing `savgol_filter` with a vectorized Endpoint Polynomial Regression.
1. Compute the analytical FIR weights for a polynomial fit of degree $p$ over window $N$ evaluated at the right-edge $x = N-1$.
2. Apply `np.convolve` to instantly calculate the Rolling Velocity across the entire DataFrame.
3. Define turning points strictly as the moments where this Rolling Velocity crosses zero (bends up/down).

> [!WARNING]
> **Window Size Selection**
> Even with an endpoint regression, we still have to select a rolling window size $N$ and degree $p$. 
> Should I:
> A) Use a static 5-bar Quadratic fit for all timeframes (like the Python artifact)?
> B) Use a static 7.5-minute Cubic fit mapped to each timeframe (like NT8)?
> C) Sweep across a few window sizes (e.g., 3, 5, 8 bars) for each timeframe to map how the anatomy changes with the curve's sensitivity?

Please let me know which configuration you prefer for the Stage 1 profiling!
