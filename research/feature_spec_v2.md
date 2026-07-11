# Feature Space V2 (416D) Specification
*Last Updated: 2026-07-10*

This document serves as the absolute ground truth for the feature space engineered in the `core_v2` module (`statistical_field_engine.py` and `features.py`). It explicitly corrects the legacy 185D documentation and maps to the actual **416-dimensional** matrix generated for ML consumption.

## The 8 Timeframes (TFs)
The feature space operates simultaneously across 8 hierarchical timeframes (TFs), anchored at 5-second intervals via a last-closed-bar alignment (zero-lookahead):
- `5s`, `15s`, `1m`, `5m`, `15m`, `1h`, `4h`, `1D`

## 416D CNN Grid Architecture
The CNN ingests a flat matrix of `416` features (`8 TFs × 52 features per TF`). 
*(Note: A global L0 `time_of_day` feature is computed but excluded from the 416D CNN spatial grid).*

Each timeframe contains exactly **52 features** structured across 5 distinct layers:

### L1: Window-Free Primitives (8 features)
Raw structural elements of the bar.
1. `price_velocity_1b`: `close[t] - close[t-1]`
2. `price_accel_1b`: `velocity[t] - velocity[t-1]`
3. `vol_velocity_1b`: `volume[t] - volume[t-1]`
4. `vol_accel_1b`: `vol_velocity[t] - vol_velocity[t-1]`
5. `bar_range`: `high - low`
6. `body`: `close - open`
7. `upper_wick`: `high - max(open, close)`
8. `lower_wick`: `min(open, close) - low`

### L2: Rolling Window Statistics (9 features)
Metrics smoothed over the `N_BASE` window for each TF.
9. `price_velocity_w`: Secant velocity `(close[t] - close[t-N]) / N`
10. `price_accel_w`: `(velocity_1b[t] - velocity_1b[t-N]) / N`
11. `vol_velocity_w`: `(volume[t] - volume[t-N]) / N`
12. `vol_accel_w`: `(vol_velocity_1b[t] - vol_velocity_1b[t-N]) / N`
13. `price_mean_w`: Rolling mean of close over `N` bars
14. `price_sigma_w`: Rolling standard deviation of close (ddof=1) over `N` bars
15. `vol_mean_w`: Rolling mean of volume over `N` bars
16. `vol_sigma_w`: Rolling standard deviation of volume (ddof=1) over `N` bars
17. `vwap_w`: Volume-weighted average price over `N` bars

### L3: Advanced Statistical & Geometrical Exceptions (11 features)
Complex metrics involving independent regressions and Hurst scaling.
18. `z_se`: Standardized close against its own OLS fit. `(close - OLS_mean) / SE_close`
19. `z_high`: Standardized high against its own OLS fit.
20. `z_low`: Standardized low against its own OLS fit.
21. `SE_high`: OLS residual standard deviation for high (wick dispersion).
22. `SE_low`: OLS residual standard deviation for low.
23. `hurst`: Hurst exponent via R/S analysis (window = `N_BASE * 8`).
24. `reversion_prob`: Ornstein-Uhlenbeck first-passage analytical probability.
25. `swing_noise`: Maximum draw-up/draw-down normalized by tick over 30 bars.
26. `z_close_vs_high`: Close normalized by high's OLS fit.
27. `z_close_vs_low`: Close normalized by low's OLS fit.
28. `band_pos`: Close's fractional position between the OLS high band and OLS low band.

### L4: Nightmare Protocol (NMP) States (12 features)
Indicators originally designed for the NMP system, targeting variance scaling and stability (λ).
29. `vr_exact`: Variance ratio (10-bar std / 60-bar std).
30. `z_21`: Exact 21-bar linear regression standardization.
31. `lambda_hat_12`: OLS slope of `log(|z_se| + 0.1)` over 12 bars (stability exponent).
32. `lambda_se_12`: Standard error of lambda over 12 bars.
33. `lambda_t_12`: T-statistic of lambda over 12 bars.
34. `lambda_hat_21`: OLS slope over 21 bars.
35. `lambda_se_21`: SE over 21 bars.
36. `lambda_t_21`: T-stat over 21 bars.
37. `lambda_hat_30`: OLS slope over 30 bars.
38. `lambda_se_30`: SE over 30 bars.
39. `lambda_t_30`: T-stat over 30 bars.
40. `vr_proxy`: Placeholder / proxy variance ratio for fast access.

### L5: Intra-Bar 1s Distributions (12 features)
Micro-structure features profiling the 1-second ticks *inside* each timeframe's closed bar.
41. `ldist_n`: Number of 1s ticks in the bar.
42. `ldist_min`: Minimum 1s close.
43. `ldist_q1`: 25th percentile of 1s closes.
44. `ldist_median`: Median 1s close.
45. `ldist_q3`: 75th percentile of 1s closes.
46. `ldist_max`: Maximum 1s close.
47. `ldist_mean`: Mean of 1s closes.
48. `ldist_std`: Standard deviation of 1s closes.
49. `ldist_skew`: Skewness of 1s closes.
50. `ldist_kurtosis`: Kurtosis of 1s closes.
51. `ldist_outlier_pct`: Percentage of 1s ticks outside 2 standard deviations.
52. `ldist_level`: OLS fitted value at the exact bar end.

## Zero-Lookahead Guarantees
- All features are computed exclusively using strictly trailing windows `[t - W + 1, t]`.
- Timeframe up-sampling uses the _last closed bar_ principle (no forward filling into the future).
- R/S analysis and OLS endpoints inherently evaluate at `x = window - 1`.

## Storage Geometry
Parquets are broken down by layer family and day inside `DATA/ATLAS/FEATURES_5s_v2/`:
- `L0/YYYY_MM_DD.parquet`
- `L1_5s/YYYY_MM_DD.parquet`
- `L2_5s/YYYY_MM_DD.parquet` ... etc.
The ML pipeline seamlessly merges these into the 416D grid during `load_features`.
