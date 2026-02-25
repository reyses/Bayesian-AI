# Bolt's Journal

2024-05-23 — [Initial Setup]
Learning: Starting fresh. Focus on measurable improvements in the Python scientific stack.
Action: Profile hot paths in core/ and training/.

2024-05-23 — [Vectorized Scaler Transform]
Learning: `sklearn.preprocessing.StandardScaler.transform` has significant overhead (~125µs per call) when transforming single samples in a tight loop. This dominates the runtime of high-frequency components like `TimeframeBeliefNetwork.tick`.
Action: Pre-extracted `mean_` and `scale_` arrays and replaced `transform()` with direct NumPy vectorization `(x - mean) / scale` (single-sample inference).
Impact: ~46x speedup (260µs vs 12ms for 100k iters) in the feature scaling step.

2024-05-24 — [Pre-converted Coefficients]
Learning: `np.array(list)` has measurable overhead when called repeatedly in tight loops (e.g. `TimeframeBeliefNetwork.tick`).
Action: Pre-converted `dir_coeff` and `mfe_coeff` from lists to numpy arrays in `TimeframeBeliefNetwork.__init__`. Removed `np.array()` wrapper in `_logistic_prob` and `_ols_mfe`.
Impact: ~1.25x speedup in the belief network analysis loop.
