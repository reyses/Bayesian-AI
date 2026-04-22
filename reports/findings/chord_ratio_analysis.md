# Two-chord analysis — noise vs trend change

**Price chord** = |close[t] − close[t−W]| (net price displacement).
**Regression chord** = |fit[t] − fit[t−W]| (regression-mean displacement).
**Ratio** = reg_chord / price_chord. Near 0 → noise; near 1 → trend.

Zigzag threshold: $15.0. Windows: [10, 20, 60, 180] 1m bars.

IS pivot events: 36,375 | OOS: 10,708

## Chord features Cohen d (UP vs DOWN next)

| Feature | d_IS | d_OOS | Walk-forward |
|---|---:|---:|---|
| `price_net_10` | +0.044 | +0.033 | — |
| `price_net_20` | +0.031 | +0.030 | — |
| `efficiency_10` | +0.026 | +0.052 | — |
| `price_net_60` | +0.022 | +0.020 | — |
| `reg_to_path_10` | -0.021 | +0.008 | — |
| `efficiency_20` | +0.017 | +0.042 | — |
| `reg_to_path_20` | -0.015 | +0.002 | — |
| `price_path_10` | +0.014 | -0.001 | — |
| `price_net_180` | +0.013 | +0.007 | — |
| `reg_chord_10` | +0.013 | +0.008 | — |
| `efficiency_60` | +0.012 | +0.018 | — |
| `price_path_20` | +0.008 | +0.001 | — |
| `reg_to_path_60` | -0.007 | -0.002 | — |
| `reg_chord_20` | +0.004 | +0.003 | — |
| `efficiency_180` | +0.004 | +0.001 | — |
| `price_path_60` | +0.003 | +0.002 | — |
| `reg_chord_180` | +0.001 | +0.002 | — |
| `price_path_180` | +0.001 | +0.001 | — |
| `reg_chord_60` | +0.001 | +0.001 | — |
| `reg_to_path_180` | -0.000 | +0.001 | — |

## IS: stratified by reg_to_path_10

| Ratio bucket | N | UP% | DOWN% | Avg leg $ |
|---|---:|---:|---:|---:|
| VERY_NOISE (<0.05) | 3,194 | 51.1% | 48.9% | $54 |
| NOISE (0.05-0.15) | 6,025 | 49.4% | 50.6% | $53 |
| MIXED (0.15-0.30) | 8,530 | 50.1% | 49.9% | $55 |
| TREND (0.30-0.50) | 9,386 | 50.6% | 49.4% | $54 |
| STRONG_TREND (>0.50) | 9,240 | 49.3% | 50.7% | $55 |

## OOS: stratified by reg_to_path_10

| Ratio bucket | N | UP% | DOWN% | Avg leg $ |
|---|---:|---:|---:|---:|
| VERY_NOISE (<0.05) | 962 | 48.0% | 52.0% | $55 |
| NOISE (0.05-0.15) | 1,831 | 50.5% | 49.5% | $55 |
| MIXED (0.15-0.30) | 2,552 | 49.4% | 50.6% | $55 |
| TREND (0.30-0.50) | 2,719 | 50.3% | 49.7% | $53 |
| STRONG_TREND (>0.50) | 2,644 | 50.6% | 49.4% | $55 |

## reg_to_path_10 distribution (IS)

| Percentile | ratio |
|---|---:|
| p5 | 0.029 |
| p10 | 0.057 |
| p25 | 0.148 |
| p50 | 0.308 |
| p75 | 0.503 |
| p90 | 0.697 |
| p95 | 0.814 |
