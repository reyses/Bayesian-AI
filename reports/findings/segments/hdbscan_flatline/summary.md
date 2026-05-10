# Stepped HDBSCAN per phrase shape

_Generated 2026-05-10T01:29:36.562922_

Split: IS    min_cluster_size: 50    min_samples: 15
Features (8): `['slope_pts_per_min', 'mean_sigma', 'sigma_rank_mid', 'r2adj', 'length_min', 'peak_abs_z', 'net_move_pts', 'tod_start_hour_utc']`

## Overall: clusters per shape

```
   shape  n_total  n_clusters  n_noise  pct_noise
FLATLINE     1169           8      262       22.4
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

## FLATLINE  (n=1169, 8 clusters, 262 NOISE = 22.4%)

```
 cluster_id   n  pct  ride_pnl_pts_mean  ride_pnl_pts_med  fade_pnl_pts_mean  fade_pnl_pts_med  max_mfe_ride_pts_mean  max_mfe_ride_pts_med  max_mae_ride_pts_mean  max_mae_ride_pts_med  resolved_as_cascade_pct  extended_60m_pct
         -1 262 22.4               12.1               5.8              -12.1              -5.8                   43.5                  32.0                  -23.6                 -13.4                      6.1              48.5
          0  81  6.9               44.8              41.0              -44.8             -41.0                   71.0                  57.2                  -19.3                 -11.5                      3.7              23.5
          1 153 13.1                7.1               9.8               -7.1              -9.8                   22.6                  19.5                  -12.4                  -9.2                      0.0              24.8
          2 106  9.1                2.1               2.6               -2.1              -2.6                   17.2                  15.2                  -12.3                  -7.8                      0.0              50.9
          3  97  8.3                7.9               5.0               -7.9              -5.0                   54.4                  47.2                  -18.3                 -11.0                     42.3              67.0
          4  93  8.0                1.3               0.8               -1.3              -0.8                   16.1                  13.8                  -13.6                 -11.5                      0.0              21.5
          5 102  8.7               16.8              22.0              -16.8             -22.0                   71.2                  63.6                  -39.5                 -21.5                      2.0              29.4
          6 115  9.8                3.2               1.2               -3.2              -1.2                   25.9                  17.2                  -19.7                 -16.2                      1.7              60.0
          7 160 13.7                6.5               5.9               -6.5              -5.9                   29.3                  24.4                  -14.9                 -11.0                      0.0              45.0
```

Chart: `reports/findings/segments/hdbscan_flatline\hdbscan_FLATLINE.png`
