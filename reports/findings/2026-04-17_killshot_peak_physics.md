# KILL_SHOT Peak Physics
Date: 2026-04-17
Source: `training/output/isolated/KILL_SHOT.pkl`  (4,411 total, 2,691 with peak>$3, 2,043 analyzable)

## Time to peak
- Median: 160s (2.7 min)
- p25: 90s, p75: 340s, p90: 565s
- Mean bars-to-peak: 50.2 (= 251s)

## Peak-detection signal fire rates (n=2043)
| Signal | Fires at peak | Rate |
|---|---|---|
| 1m velocity flips sign against trade (±3 bars) | 68 | 3.3% |
| 1m acceleration flips against trade | 5 | 0.2% |
| Volume fades >20% across peak | 241 | 11.8% |
| 1m wick_ratio > 0.3 at peak (jump >30%) | 139 | 6.8% |
| 1m p_at_center > 0.6 at peak | 946 | 46.3% |

## Feature values: before peak / AT peak / after peak
(±3 bars = ±15s window)

| Feature | Before | AT PEAK | After | Δ(after-before) |
|---|---|---|---|---|
| 1m_z_se | +0.015 | +0.013 | +0.001 | -0.014 |
| 1m_velocity | +0.000 | +0.000 | -0.250 | -0.250 |
| 1m_acceleration | +0.000 | +0.000 | +0.000 | +0.000 |
| 1m_variance_ratio | +0.386 | +0.391 | +0.397 | +0.012 |
| 1m_vol_rel | +0.832 | +0.850 | +0.908 | +0.076 |
| 1m_dmi_diff | -0.827 | -0.889 | -1.142 | -0.315 |
| 1m_wick_ratio | +0.526 | +0.500 | +0.476 | -0.050 |
| 1m_p_at_center | +0.564 | +0.569 | +0.556 | -0.007 |
| 1m_reversion_prob | +0.973 | +0.973 | +0.972 | -0.001 |
| 1m_bar_range | +28.000 | +29.000 | +31.000 | +3.000 |
| 5m_velocity | +0.000 | +0.000 | +0.000 | +0.000 |
| 5m_acceleration | +0.000 | +0.000 | +0.000 | +0.000 |
| 5m_wick_ratio | +0.864 | +0.857 | +0.854 | -0.010 |
| 15m_wick_ratio | +0.862 | +0.862 | +0.860 | -0.002 |
| 1h_z_se | +0.048 | +0.048 | +0.048 | +0.000 |
| 1h_velocity | +2.250 | +2.250 | +2.250 | +0.000 |

## Largest feature swings across peak (Cohen d)
Effect size = (median_after - median_before) / pooled_std. Ranks which signals move most when peak passes.

| Feature | Before→After | Cohen d |
|---|---|---|
| 1m_wick_ratio | +0.526 → +0.476 | -0.19 |
| 1m_vol_rel | +0.832 → +0.908 | +0.09 |
| 1m_bar_range | +28.000 → +31.000 | +0.08 |
| 5m_wick_ratio | +0.864 → +0.854 | -0.04 |
| 1m_variance_ratio | +0.386 → +0.397 | +0.04 |
| 1m_p_at_center | +0.564 → +0.556 | -0.03 |
| 1m_velocity | +0.000 → -0.250 | -0.03 |
| 1m_dmi_diff | -0.827 → -1.142 | -0.03 |
| 1m_z_se | +0.015 → +0.001 | -0.01 |
| 15m_wick_ratio | +0.862 → +0.860 | -0.01 |
| 1m_reversion_prob | +0.973 → +0.972 | -0.01 |
| 1m_acceleration | +0.000 → +0.000 | +0.00 |
| 5m_velocity | +0.000 → +0.000 | +0.00 |
| 5m_acceleration | +0.000 → +0.000 | +0.00 |
| 1h_z_se | +0.048 → +0.048 | +0.00 |
| 1h_velocity | +2.250 → +2.250 | +0.00 |

## Exit rule candidates (back-test on this cohort)
Each rule scans bars AFTER entry; exits at first trigger (5s resolution). Compared to the no-rule baseline (final close, $/trade) and peak.

| Rule | Fires | Total $ | $/trade | Avg bars to exit |
|---|---|---|---|---|
| (baseline — natural exit) | — | $+31,233 | $+11.61 | — |
| 1m velocity flips against trade | 2415 | $+18,392 | $+6.83 | 11.4 |
| 1m acceleration negative (against) | 1865 | $+26,598 | $+9.88 | 35.8 |
| p_at_center > 0.5 after MFE > 1pt | 2603 | $+20,685 | $+7.69 | 13.2 |
| reversion_prob > 0.6 after MFE > 1pt | 2641 | $+19,948 | $+7.41 | 8.0 |
| wick against trade (1m_wick_ratio > 0.4) | 2606 | $+18,496 | $+6.87 | 10.8 |
| MFE gave back 50% from running peak (trail) | 2524 | $+9,137 | $+3.40 | 12.3 |
| Fixed target $5 | 2498 | $+24,744 | $+9.20 | 13.4 |
| Fixed target $7 | 2273 | $+27,824 | $+10.34 | 17.4 |
| Fixed target $10 | 1906 | $+30,190 | $+11.22 | 23.0 |
