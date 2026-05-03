# V2 features: per-feature × per-regime × per-quantile — 2026-05-03 17:33 UTC

**TFs:** ['5s', '1m', '5m', '15m', '1h', '4h']  **Quantiles:** 5  **Split:** IS

## Top 30 (concept_tf) by max signal_strength across regimes

regime_2d             UP_SMOOTH  UP_CHOPPY  DOWN_SMOOTH  DOWN_CHOPPY  FLAT_SMOOTH  FLAT_CHOPPY  max_strength
concept_tf                                                                                                  
price_sigma_w_5s          41.86      35.76        32.48        16.36         4.34         4.22         41.86
bar_range_1h              41.11      33.58        29.38        24.63         4.39         6.65         41.11
bar_range_1m              40.61      38.43        38.15        20.28         6.36         3.52         40.61
bar_range_5m              38.02      39.77        36.51        19.80         3.00         4.41         39.77
vol_velocity_w_1h         38.52      21.09        35.20        22.91         3.77         5.64         38.52
price_sigma_w_5m          32.81      33.24        37.41        17.30         3.34         5.55         37.41
price_accel_w_1h          37.38      14.38        19.87        16.24         6.87        11.90         37.38
vol_mean_w_5m             36.69      32.26        37.17        25.13         5.11         4.05         37.17
vol_mean_w_15m            26.44      27.64        36.98        27.77         2.76         8.17         36.98
vol_mean_w_5s             30.96      30.72        36.22        18.90         4.31         2.60         36.22
price_sigma_w_1m          35.95      34.76        35.10        16.88         3.06         5.76         35.95
price_velocity_w_1h       13.43      21.70         8.31        35.53        15.69        26.60         35.53
bar_range_5s              27.20      33.35        35.53        18.56         4.71         5.60         35.53
price_velocity_1b_1h      35.32      17.49        20.81        23.24         9.38        13.77         35.32
bar_range_15m             31.58      34.92        31.71        19.24         5.10         6.27         34.92
vol_mean_w_1m             34.42      32.68        33.50        22.78         3.18         3.03         34.42
body_1h                   33.96      12.85        21.12        22.40         9.08        12.47         33.96
price_velocity_w_4h       33.71      25.01        19.40        16.79         5.72         6.24         33.71
vol_accel_1b_1h           15.29      19.16        32.96        14.83         7.32         9.04         32.96
vol_velocity_1b_1h        26.88      18.63        31.76        24.72         9.21        10.80         31.76

## Top 30 shape changers

          concept  tf  n_distinct_shapes                                                                                  shape_list  max_strength
    price_sigma_w  1m                  4 noisy,monotonic_decreasing,noisy,inverted_u_shape,monotonic_increasing,monotonic_increasing     35.949014
      vol_sigma_w  5m                  4                inverted_u_shape,monotonic_decreasing,noisy,noisy,monotonic_increasing,noisy     31.370401
    price_sigma_w  5s                  3                           noisy,monotonic_decreasing,noisy,noisy,noisy,monotonic_increasing     41.857413
        bar_range  1m                  3                           noisy,monotonic_decreasing,noisy,noisy,noisy,monotonic_increasing     40.614552
        bar_range  5m                  3            noisy,monotonic_decreasing,monotonic_increasing,noisy,noisy,monotonic_increasing     39.772989
   vol_velocity_w  1h                  3                                        noisy,monotonic_decreasing,u_shape,noisy,noisy,noisy     38.520194
       vol_mean_w  5m                  3            monotonic_decreasing,monotonic_decreasing,monotonic_increasing,noisy,noisy,noisy     37.169346
        bar_range  5s                  3                           monotonic_decreasing,noisy,noisy,noisy,monotonic_increasing,noisy     35.525529
price_velocity_1b  1h                  3                                 inverted_u_shape,inverted_u_shape,noisy,noisy,noisy,u_shape     35.322483
        bar_range 15m                  3                           noisy,monotonic_decreasing,noisy,noisy,noisy,monotonic_increasing     34.919481
  vol_velocity_1b  1h                  3                                            inverted_u_shape,noisy,noisy,noisy,u_shape,noisy     31.755131
      vol_sigma_w 15m                  3                           noisy,monotonic_decreasing,noisy,noisy,monotonic_increasing,noisy     30.863085
             body 15m                  3                                            noisy,inverted_u_shape,noisy,noisy,u_shape,noisy     30.029655
price_velocity_1b 15m                  3                                            noisy,inverted_u_shape,noisy,noisy,u_shape,noisy     29.555020
     vol_accel_1b 15m                  3                                            noisy,inverted_u_shape,noisy,noisy,u_shape,noisy     29.170409
 price_velocity_w  5m                  3                                        noisy,inverted_u_shape,u_shape,noisy,u_shape,u_shape     26.781416
           vwap_w 15m                  3                               noisy,inverted_u_shape,noisy,noisy,noisy,monotonic_decreasing     25.655886
     price_mean_w  5s                  3                               noisy,noisy,noisy,inverted_u_shape,noisy,monotonic_decreasing     25.207325
   price_accel_1b  1m                  3                                 inverted_u_shape,inverted_u_shape,noisy,noisy,noisy,u_shape     24.984008
           vwap_w  5s                  3                               noisy,noisy,noisy,inverted_u_shape,noisy,monotonic_decreasing     24.892431
     price_mean_w  1m                  3                               noisy,noisy,noisy,inverted_u_shape,noisy,monotonic_decreasing     24.593316
           vwap_w  1m                  3                               noisy,noisy,noisy,inverted_u_shape,noisy,monotonic_decreasing     24.593074
     price_mean_w  5m                  3                    noisy,inverted_u_shape,noisy,inverted_u_shape,noisy,monotonic_decreasing     24.468404
           vwap_w  1h                  3                    noisy,inverted_u_shape,noisy,inverted_u_shape,noisy,monotonic_decreasing     24.438088
           vwap_w  5m                  3                    noisy,inverted_u_shape,noisy,inverted_u_shape,noisy,monotonic_decreasing     24.427682
      vol_accel_w  4h                  3                                            noisy,inverted_u_shape,noisy,noisy,u_shape,noisy     23.930301
   price_accel_1b  4h                  3                                            noisy,inverted_u_shape,noisy,noisy,u_shape,noisy     23.919842
   vol_velocity_w  1m                  3                                 inverted_u_shape,inverted_u_shape,noisy,noisy,u_shape,noisy     23.812509
  vol_velocity_1b  5m                  3                                            noisy,inverted_u_shape,noisy,noisy,u_shape,noisy     23.491016
      vol_sigma_w  1h                  3                                            noisy,inverted_u_shape,noisy,noisy,u_shape,noisy     23.156271
