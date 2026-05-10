# Stepped HDBSCAN per phrase shape

_Generated 2026-05-10T01:07:51.808539_

Split: IS    min_cluster_size: 30    min_samples: 10
Features (16): `['slope_15m__mean', 'slope_15m__std', 'z_close_15m__mean', 'z_close_15m__std', 'sigma_rank_15m__mean', 'sigma_rank_15m__std', 'slope_5m__mean', 'slope_5m__std', 'z_close_5m__mean', 'z_close_5m__std', 'sigma_rank_5m__mean', 'sigma_rank_5m__std', 'r2adj_5m__mean', 'r2adj_5m__std', 'length_min', 'peak_abs_z']`

## Overall: clusters per shape

```
           shape  n_total  n_clusters  n_noise  pct_noise
EXPONENTIAL_DOWN      365           3      117       32.1
  EXPONENTIAL_UP      362           3      118       32.6
       LINEAR_UP      336           4      151       44.9
     LINEAR_DOWN      303           2      161       53.1
           NOISE      198           3       19        9.6
  LOGARITHMIC_UP      121           0      121      100.0
       STEP_DOWN      114           0      114      100.0
LOGARITHMIC_DOWN      105           0      105      100.0
         STEP_UP       94           0       94      100.0
```

## How to read this

- Per shape, UMAP reduces chord-fingerprint features to 2D, then HDBSCAN
  finds natural density clusters.
- `cluster_id = -1` is HDBSCAN NOISE (idiosyncratic phrases that do not
  fit any cluster).
- The NUMBER of clusters per shape tells us the conditioning depth the
  data supports for that shape.
- High noise % means most phrases of that shape are idiosyncratic and
  the Bayesian table must rely on shape-level prior alone.

## EXPONENTIAL_UP  (n=362, 3 clusters, 118 NOISE = 32.6%)

```
 cluster_id   n  pct  ride_pnl_pts_mean  ride_pnl_pts_med  fade_pnl_pts_mean  fade_pnl_pts_med  max_mfe_ride_pts_mean  max_mfe_ride_pts_med  max_mae_ride_pts_mean  max_mae_ride_pts_med  resolved_as_cascade_pct  extended_60m_pct
         -1 118 32.6               84.2              25.2              -84.2             -25.2                  122.0                  50.6                  -22.3                 -13.5                     15.3              65.3
          0  98 27.1               63.3              50.4              -63.3             -50.4                  102.4                  87.4                  -22.1                 -12.9                     52.0              93.9
          1  95 26.2                5.2               3.2               -5.2              -3.2                   26.1                  18.2                  -21.2                 -16.5                      1.1              53.7
          2  51 14.1               20.2              18.2              -20.2             -18.2                   40.5                  37.0                  -11.9                  -7.5                      0.0              74.5
```

Chart: `reports/findings/segments/stepped_hdbscan\hdbscan_EXPONENTIAL_UP.png`

## NOISE  (n=198, 3 clusters, 19 NOISE = 9.6%)

```
 cluster_id  n  pct  ride_pnl_pts_mean  ride_pnl_pts_med  fade_pnl_pts_mean  fade_pnl_pts_med  max_mfe_ride_pts_mean  max_mfe_ride_pts_med  max_mae_ride_pts_mean  max_mae_ride_pts_med  resolved_as_cascade_pct  extended_60m_pct
         -1 19  9.6               10.7               0.0              -10.7               0.0                   54.9                  33.2                  -21.7                 -12.8                      5.3              31.6
          0 38 19.2                0.0               0.0                0.0               0.0                  112.3                  96.2                  -33.1                 -19.1                     42.1             100.0
          1 94 47.5                1.1               0.0               -1.1               0.0                   31.9                  21.6                  -25.7                 -18.6                     13.8              40.4
          2 47 23.7                1.0               0.0               -1.0               0.0                   16.4                  13.2                 -122.1                 -62.0                     40.4              91.5
```

Chart: `reports/findings/segments/stepped_hdbscan\hdbscan_NOISE.png`

## EXPONENTIAL_DOWN  (n=365, 3 clusters, 117 NOISE = 32.1%)

```
 cluster_id   n  pct  ride_pnl_pts_mean  ride_pnl_pts_med  fade_pnl_pts_mean  fade_pnl_pts_med  max_mfe_ride_pts_mean  max_mfe_ride_pts_med  max_mae_ride_pts_mean  max_mae_ride_pts_med  resolved_as_cascade_pct  extended_60m_pct
         -1 117 32.1               36.3              15.8              -36.3             -15.8                   76.9                  36.0                  -18.4                 -12.0                     23.9              76.1
          0 122 33.4                6.2               4.2               -6.2              -4.2                   26.4                  18.0                  -17.5                 -11.9                      1.6              40.2
          1  47 12.9               40.1              29.0              -40.1             -29.0                   77.4                  55.5                  -12.4                  -5.0                     38.3              80.9
          2  79 21.6              135.8              91.5             -135.8             -91.5                  231.5                 159.5                  -26.3                 -16.8                     57.0             100.0
```

Chart: `reports/findings/segments/stepped_hdbscan\hdbscan_EXPONENTIAL_DOWN.png`

## LINEAR_UP  (n=336, 4 clusters, 151 NOISE = 44.9%)

```
 cluster_id   n  pct  ride_pnl_pts_mean  ride_pnl_pts_med  fade_pnl_pts_mean  fade_pnl_pts_med  max_mfe_ride_pts_mean  max_mfe_ride_pts_med  max_mae_ride_pts_mean  max_mae_ride_pts_med  resolved_as_cascade_pct  extended_60m_pct
         -1 151 44.9              102.1              85.2             -102.1             -85.2                  166.6                 139.0                  -17.7                 -11.8                     39.7              92.7
          0  32  9.5               35.9              38.9              -35.9             -38.9                   66.0                  63.6                  -28.0                 -13.0                      6.2             100.0
          1  50 14.9                1.8              -0.1               -1.8               0.1                   26.3                  22.8                  -21.8                 -16.0                      0.0              78.0
          2  63 18.8               47.2              47.8              -47.2             -47.8                   81.8                  82.0                   -9.8                  -3.8                     38.1              88.9
          3  40 11.9               16.4              17.2              -16.4             -17.2                   56.8                  50.9                  -19.1                 -13.4                     45.0              97.5
```

Chart: `reports/findings/segments/stepped_hdbscan\hdbscan_LINEAR_UP.png`

## LINEAR_DOWN  (n=303, 2 clusters, 161 NOISE = 53.1%)

```
 cluster_id   n  pct  ride_pnl_pts_mean  ride_pnl_pts_med  fade_pnl_pts_mean  fade_pnl_pts_med  max_mfe_ride_pts_mean  max_mfe_ride_pts_med  max_mae_ride_pts_mean  max_mae_ride_pts_med  resolved_as_cascade_pct  extended_60m_pct
         -1 161 53.1               39.2              16.2              -39.2             -16.2                   90.9                  54.8                  -15.7                  -8.0                     34.2              89.4
          0  40 13.2              236.7             185.9             -236.7            -185.9                  367.6                 332.2                  -28.5                 -13.4                     62.5             100.0
          1 102 33.7               42.1              31.5              -42.1             -31.5                   91.3                  76.2                  -12.2                  -8.2                     52.0              97.1
```

Chart: `reports/findings/segments/stepped_hdbscan\hdbscan_LINEAR_DOWN.png`

## LOGARITHMIC_UP  (n=121, 0 clusters, 121 NOISE = 100.0%)

```
 cluster_id   n   pct  ride_pnl_pts_mean  ride_pnl_pts_med  fade_pnl_pts_mean  fade_pnl_pts_med  max_mfe_ride_pts_mean  max_mfe_ride_pts_med  max_mae_ride_pts_mean  max_mae_ride_pts_med  resolved_as_cascade_pct  extended_60m_pct
         -1 121 100.0               77.9              34.0              -77.9             -34.0                  122.4                  77.0                  -16.6                  -8.5                     33.1              88.4
```

Chart: `reports/findings/segments/stepped_hdbscan\hdbscan_LOGARITHMIC_UP.png`

## STEP_DOWN  (n=114, 0 clusters, 114 NOISE = 100.0%)

```
 cluster_id   n   pct  ride_pnl_pts_mean  ride_pnl_pts_med  fade_pnl_pts_mean  fade_pnl_pts_med  max_mfe_ride_pts_mean  max_mfe_ride_pts_med  max_mae_ride_pts_mean  max_mae_ride_pts_med  resolved_as_cascade_pct  extended_60m_pct
         -1 114 100.0                7.7               6.2               -7.7              -6.2                   35.2                  25.0                  -13.0                  -7.4                      1.8              11.4
```

Chart: `reports/findings/segments/stepped_hdbscan\hdbscan_STEP_DOWN.png`

## LOGARITHMIC_DOWN  (n=105, 0 clusters, 105 NOISE = 100.0%)

```
 cluster_id   n   pct  ride_pnl_pts_mean  ride_pnl_pts_med  fade_pnl_pts_mean  fade_pnl_pts_med  max_mfe_ride_pts_mean  max_mfe_ride_pts_med  max_mae_ride_pts_mean  max_mae_ride_pts_med  resolved_as_cascade_pct  extended_60m_pct
         -1 105 100.0               58.8              15.5              -58.8             -15.5                  114.2                  73.8                  -17.5                  -8.8                     41.9              83.8
```

Chart: `reports/findings/segments/stepped_hdbscan\hdbscan_LOGARITHMIC_DOWN.png`

## STEP_UP  (n=94, 0 clusters, 94 NOISE = 100.0%)

```
 cluster_id  n   pct  ride_pnl_pts_mean  ride_pnl_pts_med  fade_pnl_pts_mean  fade_pnl_pts_med  max_mfe_ride_pts_mean  max_mfe_ride_pts_med  max_mae_ride_pts_mean  max_mae_ride_pts_med  resolved_as_cascade_pct  extended_60m_pct
         -1 94 100.0               10.1               5.9              -10.1              -5.9                   34.7                  24.1                  -15.9                 -10.0                      3.2              11.7
```

Chart: `reports/findings/segments/stepped_hdbscan\hdbscan_STEP_UP.png`
