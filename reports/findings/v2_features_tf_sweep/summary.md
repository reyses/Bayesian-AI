# V2 features TF-sweep EDA — 2026-05-03 16:50 UTC

**Concepts:** 23  **TFs:** ['5s', '15s', '1m', '5m', '15m', '1h', '4h', '1D']  **Split:** IS

## Concepts that FLIP sign across TFs (regime relationship character changes)

        concept  tfs_present  sign_flips  max_abs_d  monotone_increasing  monotone_decreasing      d_5s      d_5m      d_1h      d_1D
vol_velocity_1b            8           5   0.085864                False                False -0.003066 -0.001678  0.008817  0.085864
    vol_accel_w            8           5   0.081040                False                False -0.013224 -0.001359  0.040152  0.035148
 price_accel_1b            8           4   0.157886                False                False  0.035678 -0.002085  0.016085  0.099045
 vol_velocity_w            8           3   0.427984                False                False  0.006652 -0.004998 -0.020301 -0.205667
   vol_accel_1b            8           2   0.118408                False                False -0.011024  0.000332  0.008886 -0.118408
  price_sigma_w            8           2   0.202429                False                False -0.193359 -0.202429 -0.130470 -0.000091
      bar_range            8           1   0.235308                False                False -0.182817 -0.235308 -0.221976  0.177397
  price_accel_w            8           1   0.800973                False                False -0.000367  0.001939  0.293458  0.546531
     vol_mean_w            8           1   0.435300                False                False -0.121117 -0.223133 -0.435300  0.072076

## Concepts that hold consistently across TFs

          concept  tfs_present  sign_flips  max_abs_d  monotone_increasing  monotone_decreasing      d_5s      d_5m      d_1h      d_1D
 price_velocity_w            8           0   1.250112                False                False  0.089615  0.485702  1.250112  0.314050
price_velocity_1b            8           0   0.897687                False                False  0.054314  0.168323  0.515205  0.385925
             body            8           0   0.873125                False                False  0.052622  0.170126  0.512882  0.367982
      vol_sigma_w            8           0   0.411942                False                False -0.064175 -0.135478 -0.411942 -0.011676
           vwap_w            8           0   0.261356                False                 True  0.261356  0.254964  0.143932  0.024408
     price_mean_w            8           0   0.261308                False                False  0.261308  0.252778  0.128006  0.026619

## Cohen-d UP_SMOOTH vs DOWN_SMOOTH pivot

tf                   5s   15s    1m    5m   15m    1h    4h    1D
concept                                                          
bar_range         -0.18 -0.20 -0.19 -0.24 -0.22 -0.22 -0.18  0.18
body               0.05  0.03  0.11  0.17  0.29  0.51  0.87  0.37
price_accel_1b     0.04 -0.03  0.04 -0.00  0.00  0.02  0.16  0.10
price_accel_w     -0.00 -0.02  0.00  0.00  0.02  0.29  0.80  0.55
price_mean_w       0.26  0.26  0.26  0.25  0.22  0.13  0.02  0.03
price_sigma_w     -0.19 -0.20 -0.19 -0.20 -0.15 -0.13  0.05 -0.00
price_velocity_1b  0.05  0.04  0.11  0.17  0.29  0.52  0.90  0.39
price_velocity_w   0.09  0.14  0.29  0.49  0.86  1.25  0.82  0.31
vol_accel_1b      -0.01  0.01  0.00  0.00  0.00  0.01 -0.04 -0.12
vol_accel_w       -0.01  0.00  0.00 -0.00 -0.00  0.04 -0.08  0.04
vol_mean_w        -0.12 -0.18 -0.21 -0.22 -0.26 -0.44 -0.15  0.07
vol_sigma_w       -0.06 -0.11 -0.16 -0.14 -0.16 -0.41 -0.05 -0.01
vol_velocity_1b   -0.00  0.01  0.02 -0.00 -0.00  0.01 -0.01  0.09
vol_velocity_w     0.01  0.00 -0.00 -0.00  0.00 -0.02 -0.43 -0.21
vwap_w             0.26  0.26  0.26  0.25  0.23  0.14  0.04  0.02

## Cohen-d SMOOTH vs CHOPPY pivot

tf                   5s   15s    1m    5m   15m    1h    4h    1D
concept                                                          
bar_range         -0.12 -0.13 -0.13 -0.14 -0.14 -0.15 -0.14 -0.18
body               0.02 -0.00  0.01  0.00  0.01  0.02  0.03  0.09
price_accel_1b     0.01 -0.01  0.01  0.00  0.00 -0.00 -0.01 -0.02
price_accel_w      0.02  0.01 -0.00  0.00  0.00 -0.00 -0.00  0.01
price_mean_w      -0.03 -0.03 -0.03 -0.03 -0.03 -0.03 -0.05 -0.06
price_sigma_w     -0.13 -0.14 -0.11 -0.10 -0.07  0.09 -0.02  0.05
price_velocity_1b  0.01  0.00  0.01  0.01  0.01  0.02  0.03  0.07
price_velocity_w   0.01  0.00  0.01  0.02  0.03  0.10  0.19  0.20
vol_accel_1b      -0.00 -0.00 -0.01  0.00  0.00  0.00  0.03 -0.23
vol_accel_w       -0.01 -0.01 -0.00  0.00  0.01 -0.03  0.00 -0.42
vol_mean_w        -0.08 -0.11 -0.13 -0.13 -0.13 -0.04  0.01  0.03
vol_sigma_w       -0.04 -0.07 -0.10 -0.11 -0.10 -0.05 -0.02 -0.15
vol_velocity_1b   -0.00 -0.00  0.01  0.00  0.00 -0.03 -0.05 -0.38
vol_velocity_w    -0.00 -0.01  0.00  0.00 -0.04 -0.10 -0.10 -0.30
vwap_w            -0.03 -0.03 -0.03 -0.03 -0.03 -0.03 -0.06 -0.06
