# V1 -> V2 threshold calibration table

Generated 2026-05-04 06:47 UTC
Sample: 41205 bars across 5 days

## How to read

- `v1_fire_rate_pct`: fraction of V1 bars where the threshold fires (CURRENT BEHAVIOR)
- `v2_rate_pct_if_literal`: fraction of V2 bars where the SAME literal threshold fires (BROKEN if very low)
- `v2_rate_change_pct`: how much fire rate changes if threshold is kept literal
- `v2_threshold_matched`: V2 threshold that matches V1 fire rate (USE THIS)

## z_se thresholds

concept  tf  v1_threshold  v1_fire_rate_pct  v2_rate_pct_if_literal  v2_rate_change_pct  v2_threshold_matched  v2_rate_pct_at_matched
 |z_se|  1m           1.0            44.135                  40.701              -7.781                 0.929                  44.121
 |z_se|  1m           1.4            25.778                  22.390             -13.143                 1.317                  25.776
 |z_se|  1m           1.5            21.747                  18.466             -15.088                 1.415                  21.740
 |z_se|  1m           2.0             7.645                   5.189             -32.127                 1.871                   7.623
 |z_se|  5m           1.0            39.828                  33.086             -16.928                 0.895                  39.784
 |z_se|  5m           1.4            21.898                  12.799             -41.549                 1.180                  21.803
 |z_se|  5m           1.5            17.262                   8.283             -52.017                 1.295                  17.260
 |z_se|  5m           2.0             6.829                   0.808             -88.166                 1.568                   6.739
 |z_se| 15m           1.0            37.537                  34.226              -8.819                 0.960                  37.515
 |z_se| 15m           1.4            21.985                  17.374             -20.974                 1.236                  21.840
 |z_se| 15m           1.5            19.223                  14.290             -25.666                 1.344                  19.054
 |z_se| 15m           2.0             8.038                   4.330             -46.135                 1.718                   7.904
 |z_se|  1h           1.0            50.788                  40.383             -20.485                 0.870                  50.431
 |z_se|  1h           1.4            23.565                  21.315              -9.547                 1.396                  23.021
 |z_se|  1h           1.5            23.565                  19.966             -15.273                 1.396                  23.021
 |z_se|  1h           2.0            12.224                   6.526             -46.615                 1.759                  11.409

## velocity thresholds

   concept  tf  v1_threshold  v1_fire_rate_pct  v2_rate_pct_if_literal  v2_rate_change_pct  v2_threshold_matched  v2_rate_pct_at_matched
|velocity|  1m          10.0             8.475                   0.116             -98.625                 2.467                   8.382
|velocity|  1m          30.0             0.641                   0.000            -100.000                 7.634                   0.641
|velocity|  1m          50.0             0.116                   0.000            -100.000                 9.951                   0.116
|velocity|  1m         100.0             0.000                   0.000               0.000               100.000                   0.000
|velocity|  5m          10.0            27.730                   3.203             -88.447                 3.083                  27.628
|velocity|  5m          30.0             4.791                   0.000            -100.000                 8.444                   4.652
|velocity|  5m          50.0             1.451                   0.000            -100.000                17.750                   1.311
|velocity|  5m         100.0             0.146                   0.000            -100.000                24.419                   0.146
|velocity| 15m          10.0            50.533                   9.606             -80.991                 3.208                  50.494
|velocity| 15m          30.0            11.159                   0.000            -100.000                 9.771                  10.914
|velocity| 15m          50.0             5.664                   0.000            -100.000                12.417                   5.237
|velocity| 15m         100.0             2.621                   0.000            -100.000                16.542                   2.616
|velocity|  1h          10.0            69.159                  45.249             -34.572                 5.333                  67.560
|velocity|  1h          30.0            24.021                   2.580             -89.260                13.396                  23.116
|velocity|  1h          50.0             9.921                   0.000            -100.000                24.521                   9.416
|velocity|  1h         100.0             1.747                   0.000            -100.000                30.312                   1.425

## Mapping to nightmare_blended.py constants

| Constant | V1 value | TFs used | V2 calibrated values |
|---|---:|---|---|
| `ROCHE` | 2.0 | 1m | 1m=1.87 |
| `H1_Z_MIN` | 1.0 | 1h | 1h=0.87 |
| `H1_AGAINST_Z_MIN` | 1.5 | 1h | 1h=1.40 |
| `EXHAUST_Z_MIN` | 1.4 | 1m | 1m=1.32 |
| `MTF_Z_MIN` | 1.4 | multiple | 1m=1.32, 5m=1.18, 15m=1.24 |
| `MTF_1M_VEL_ALIVE` | 10.0 | 1m | 1m=2.47 |
| `MTF_5M_VEL_MIN` | 30.0 | 5m | 5m=8.44 |
| `VELOCITY_THRESHOLD` | 50.0 | 1m | 1m=9.95 |
| `FREIGHT_TRAIN_THRESHOLD` | 100.0 | 1m | 1m=100.00 |

## Untouchable thresholds (EXACT match — no retune)

- `VR_ENTRY = 1.0`, `FREIGHT_TRAIN_VR_MAX = 0.85`, `MTF_VR_MIN = 0.58`, `REGIME_VR_MAX = 0.35`, `REGIME_FLIP_VR_BAIL = 0.30`, `ABSORB_VR_BAIL = 0.65`, `RIDE_VR_TRENDING = 1.0`, `VR_CONFIRMING = 0.8` — variance_ratio is EXACT
- `WICK_5M_MIN = 0.83`, `WICK_15M_MIN = 0.77` — wick_ratio is EXACT
- `MTF_VOL_MIN = 2.0`, `ABSORB_VOL_PERSIST_MAX = 1.5` — vol_rel is EXACT
