# Entry-anchored, magnitude-preserving probe (owner critique)
4,182 frames, 25 days, 42 features (level feats entry-anchored as displacement; NO z-standardization).

- **GBM OOS IC (entry-anchored, raw): +0.0803**, 95% CI [-0.0158, +0.1672] incl 0
- baseline (exclude-levels, z-linear/raw-GBM): +0.027 / +0.050

## A/B cohort (bad − good, entry-anchored feature effect size, |d|)
| feature | Cohen d (bad vs good) |
|---|---|
| swing_noise_30_dEntry | -0.54 |
| hurst_30 | -0.50 |
| price_sigma_30_dEntry | -0.47 |
| lambda_hat_30 | -0.46 |
| lambda_t_30 | -0.44 |
| vol_velocity_30 | -0.39 |
| vol_mean_30_dEntry | -0.31 |
| vol_accel_30 | -0.28 |
| bar_range | -0.27 |
| lambda_t_21 | -0.27 |
| ldist_kurtosis_dEntry | +0.27 |
| lambda_hat_21 | -0.25 |

Read: whether entry-anchoring the level features (magnitude kept) raises IC over excluding them tells us if the owner's anchoring fix adds signal. The A/B effect sizes show which entry-anchored features separate winners from losers — the magnitude-preserving, spec-free view. Still 16 test days = underpowered; at-scale build_dataset resolves.
