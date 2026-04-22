# Regression-line Cohen d at zigzag pivots

**Threshold**: $15.0. **Min leg**: $0.0.

IS pivot events: 26,760 | OOS: 7,593

**Hypothesis**: at each zigzag pivot, regression-line slopes and residuals across multiple windows predict the direction of the next leg. Uses smoothed OLS β and residual — not raw velocity/z features.

## Features computed

For each pivot, at 5 window sizes (W=10, 20, 60, 180, 720 1m bars):
- `beta_W` — OLS slope (points per bar)
- `res_W` — price minus fitted value at pivot
- `res_W_norm` — residual / in-window residual std
- `beta_sign_align` — fraction of windows where β sign matches beta_60
- `beta_60_minus_720`, `beta_10_minus_60`, `beta_20_minus_180` — acceleration proxies (short slope minus long slope)

## Top features by IS |d|

| Rank | Feature | d_IS | d_OOS | UP mean | DOWN mean | Walk-fwd |
|---:|---|---:|---:|---:|---:|---|
| 1 | `res_10_norm` | -2.460 | -2.400 | -0.923 | +0.865 | ✓ |
| 2 | `res_20_norm` | -2.019 | -1.899 | -1.024 | +0.929 | ✓ |
| 3 | `res_10` | -1.445 | -1.603 | -7.917 | +7.580 | ✓ |
| 4 | `res_60_norm` | -1.179 | -1.133 | -0.857 | +0.696 | ✓ |
| 5 | `res_20` | -1.161 | -1.344 | -10.940 | +10.323 | ✓ |
| 6 | `res_60` | -0.716 | -0.824 | -13.686 | +12.403 | ✓ |
| 7 | `res_180_norm` | -0.674 | -0.647 | -0.612 | +0.421 | ✓ |
| 8 | `res_180` | -0.426 | -0.497 | -15.574 | +12.370 | ✓ |
| 9 | `beta_10` | -0.412 | -0.450 | -1.154 | +0.858 | ✓ |
| 10 | `beta_10_minus_60` | -0.391 | -0.424 | -1.034 | +0.908 | ✓ |
| 11 | `res_720_norm` | -0.364 | -0.320 | -0.532 | +0.176 | ✓ |
| 12 | `res_720` | -0.241 | -0.269 | -21.083 | +7.491 | ✓ |
| 13 | `beta_20` | -0.164 | -0.181 | -0.404 | +0.165 | ✓ |
| 14 | `beta_20_minus_180` | -0.158 | -0.175 | -0.343 | +0.218 | ✓ |
| 15 | `beta_60` | -0.035 | -0.039 | -0.120 | -0.050 | — |
| 16 | `beta_60_minus_720` | -0.035 | -0.039 | -0.094 | -0.026 | — |
| 17 | `beta_sign_align` | -0.032 | -0.002 | +0.665 | +0.672 | — |
| 18 | `beta_180` | -0.006 | -0.008 | -0.061 | -0.054 | — |
| 19 | `beta_720` | -0.001 | -0.001 | -0.025 | -0.025 | — |

## Walk-forward stable features

Sign match IS/OOS AND min(|d_IS|, |d_OOS|) >= 0.15.

| Feature | d_IS | d_OOS | min\|d\| | UP mean IS | DOWN mean IS |
|---|---:|---:|---:|---:|---:|
| `res_10_norm` | -2.460 | -2.400 | 2.400 | -0.923 | +0.865 |
| `res_20_norm` | -2.019 | -1.899 | 1.899 | -1.024 | +0.929 |
| `res_10` | -1.445 | -1.603 | 1.445 | -7.917 | +7.580 |
| `res_20` | -1.161 | -1.344 | 1.161 | -10.940 | +10.323 |
| `res_60_norm` | -1.179 | -1.133 | 1.133 | -0.857 | +0.696 |
| `res_60` | -0.716 | -0.824 | 0.716 | -13.686 | +12.403 |
| `res_180_norm` | -0.674 | -0.647 | 0.647 | -0.612 | +0.421 |
| `res_180` | -0.426 | -0.497 | 0.426 | -15.574 | +12.370 |
| `beta_10` | -0.412 | -0.450 | 0.412 | -1.154 | +0.858 |
| `beta_10_minus_60` | -0.391 | -0.424 | 0.391 | -1.034 | +0.908 |
| `res_720_norm` | -0.364 | -0.320 | 0.320 | -0.532 | +0.176 |
| `res_720` | -0.241 | -0.269 | 0.241 | -21.083 | +7.491 |
| `beta_20` | -0.164 | -0.181 | 0.164 | -0.404 | +0.165 |
| `beta_20_minus_180` | -0.158 | -0.175 | 0.158 | -0.343 | +0.218 |
