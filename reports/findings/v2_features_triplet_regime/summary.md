# V2 features triplet regime-stratified (Layer C1) - 2026-05-03 17:59 UTC

**Triplet design**: anchor (leading feature) x companion x companion. Each feature binned into 3 regime-local quantiles. 27 cells per (triplet, regime). Min cell n = 80.

**Lift threshold**: |cell_mean - regime_baseline| >= 10.0 ticks

**Anchors used**: [('bar_range', '1h'), ('vol_mean_w', '1h'), ('hurst_w', '1h'), ('hurst_w', '15m'), ('hurst_w', '5m'), ('hurst_w', '1m')]

**Companions used**: [('price_velocity_w', '5m'), ('price_sigma_w', '5m'), ('swing_noise_w', '1m'), ('bar_range', '5m'), ('body', '5m'), ('vol_velocity_w', '15m'), ('vol_mean_w', '5m'), ('reversion_prob_w', '15m'), ('z_se_w', '15m'), ('price_velocity_1b', '15m')]

## Regime baselines

- **UP_SMOOTH**: +19.84 ticks
- **UP_CHOPPY**: +10.99 ticks
- **DOWN_SMOOTH**: -21.82 ticks
- **DOWN_CHOPPY**: -13.60 ticks
- **FLAT_SMOOTH**: -0.03 ticks
- **FLAT_CHOPPY**: -0.29 ticks

## Top 50 high-lift triplet cells

anchor_concept anchor_tf        x_concept x_tf         y_concept y_tf   regime_2d  q_anchor  q_x  q_y   n   cell_mean  regime_mean       lift  win_rate   cell_std                                              triplet_id  abs_lift
       hurst_w       15m    price_sigma_w   5m    vol_velocity_w  15m   UP_SMOOTH         1    2    1  96  119.744792    19.840890  99.903902  0.645833 334.354047         hurst_w_15m|price_sigma_w_5m|vol_velocity_w_15m 99.903902
     bar_range        1h price_velocity_w   5m    vol_velocity_w  15m   UP_SMOOTH         2    2    1 123  114.922764    19.840890  95.081874  0.723577 310.402776     bar_range_1h|price_velocity_w_5m|vol_velocity_w_15m 95.081874
    vol_mean_w        1h    price_sigma_w   5m    vol_velocity_w  15m   UP_SMOOTH         0    2    1 120  114.166667    19.840890  94.325777  0.741667 297.401070       vol_mean_w_1h|price_sigma_w_5m|vol_velocity_w_15m 94.325777
       hurst_w       15m price_velocity_w   5m        vol_mean_w   5m   UP_SMOOTH         1    2    0 111  113.907658    19.840890  94.066768  0.594595 324.523633           hurst_w_15m|price_velocity_w_5m|vol_mean_w_5m 94.066768
       hurst_w        1h price_velocity_w   5m  reversion_prob_w  15m   UP_SMOOTH         2    2    1 211  113.613744    19.840890  93.772854  0.710900 310.101676     hurst_w_1h|price_velocity_w_5m|reversion_prob_w_15m 93.772854
       hurst_w        5m    price_sigma_w   5m    vol_velocity_w  15m   UP_SMOOTH         2    2    1  93  109.846774    19.840890  90.005884  0.623656 329.427517          hurst_w_5m|price_sigma_w_5m|vol_velocity_w_15m 90.005884
       hurst_w        1h    price_sigma_w   5m  reversion_prob_w  15m   UP_SMOOTH         2    2    1 227  109.838106    19.840890  89.997216  0.660793 300.982525        hurst_w_1h|price_sigma_w_5m|reversion_prob_w_15m 89.997216
     bar_range        1h        bar_range   5m    vol_velocity_w  15m DOWN_SMOOTH         2    2    1  82 -104.881098   -21.819873 -83.061224  0.268293 139.657331            bar_range_1h|bar_range_5m|vol_velocity_w_15m 83.061224
     bar_range        1h    swing_noise_w   1m    vol_velocity_w  15m DOWN_SMOOTH         2    2    1  81 -103.098765   -21.819873 -81.278892  0.296296 143.747313        bar_range_1h|swing_noise_w_1m|vol_velocity_w_15m 81.278892
       hurst_w        1h    swing_noise_w   1m  reversion_prob_w  15m   UP_SMOOTH         2    2    1 230   98.242391    19.840890  78.401501  0.665217 287.291751        hurst_w_1h|swing_noise_w_1m|reversion_prob_w_15m 78.401501
       hurst_w        1h        bar_range   5m    vol_velocity_w  15m DOWN_SMOOTH         2    2    1  88 -100.079545   -21.819873 -78.259672  0.193182 138.700571              hurst_w_1h|bar_range_5m|vol_velocity_w_15m 78.259672
       hurst_w        1h    price_sigma_w   5m    vol_velocity_w  15m   UP_SMOOTH         2    2    1 137   97.682482    19.840890  77.841592  0.700730 281.916213          hurst_w_1h|price_sigma_w_5m|vol_velocity_w_15m 77.841592
       hurst_w        1h price_velocity_w   5m        vol_mean_w   5m   UP_SMOOTH         2    2    0 144   95.147569    19.840890  75.306679  0.729167 286.604809            hurst_w_1h|price_velocity_w_5m|vol_mean_w_5m 75.306679
    vol_mean_w        1h   vol_velocity_w  15m price_velocity_1b  15m   UP_CHOPPY         2    2    0 117   82.955128    10.986748  71.968380  0.675214 170.798725  vol_mean_w_1h|vol_velocity_w_15m|price_velocity_1b_15m 71.968380
     bar_range        1h        bar_range   5m    vol_velocity_w  15m   UP_SMOOTH         2    1    1 114   91.269737    19.840890  71.428847  0.780702 274.597193            bar_range_1h|bar_range_5m|vol_velocity_w_15m 71.428847
       hurst_w       15m       vol_mean_w   5m            z_se_w  15m   UP_SMOOTH         2    2    2 200   90.931250    19.840890  71.090360  0.725000 265.074675                    hurst_w_15m|vol_mean_w_5m|z_se_w_15m 71.090360
    vol_mean_w        1h        bar_range   5m    vol_velocity_w  15m DOWN_SMOOTH         0    2    1  82  -92.451220   -21.819873 -70.631346  0.231707 130.338711           vol_mean_w_1h|bar_range_5m|vol_velocity_w_15m 70.631346
    vol_mean_w        1h    swing_noise_w   1m    vol_velocity_w  15m   UP_SMOOTH         1    2    0  88  -50.093750    19.840890 -69.934640  0.318182  96.095007       vol_mean_w_1h|swing_noise_w_1m|vol_velocity_w_15m 69.934640
       hurst_w        1h reversion_prob_w  15m price_velocity_1b  15m   UP_SMOOTH         2    1    2 226   88.544248    19.840890  68.703358  0.615044 279.685002   hurst_w_1h|reversion_prob_w_15m|price_velocity_1b_15m 68.703358
       hurst_w       15m price_velocity_w   5m    vol_velocity_w  15m   UP_SMOOTH         1    2    1 149   86.805369    19.840890  66.964479  0.630872 284.220453      hurst_w_15m|price_velocity_w_5m|vol_velocity_w_15m 66.964479
    vol_mean_w        1h    swing_noise_w   1m            z_se_w  15m DOWN_SMOOTH         0    2    1  88  -88.366477   -21.819873 -66.546604  0.272727 122.398175               vol_mean_w_1h|swing_noise_w_1m|z_se_w_15m 66.546604
       hurst_w        1h    price_sigma_w   5m            z_se_w  15m   UP_SMOOTH         2    2    2 390   86.153205    19.840890  66.312315  0.656410 250.155499                  hurst_w_1h|price_sigma_w_5m|z_se_w_15m 66.312315
       hurst_w        1h        bar_range   5m            z_se_w  15m   UP_SMOOTH         2    2    2 310   85.815323    19.840890  65.974433  0.703226 240.488419                      hurst_w_1h|bar_range_5m|z_se_w_15m 65.974433
    vol_mean_w        1h    swing_noise_w   1m  reversion_prob_w  15m DOWN_SMOOTH         0    2    2  88  -87.457386   -21.819873 -65.637513  0.295455 123.063040     vol_mean_w_1h|swing_noise_w_1m|reversion_prob_w_15m 65.637513
       hurst_w       15m    price_sigma_w   5m              body   5m   UP_SMOOTH         1    2    1 108   84.981481    19.840890  65.140591  0.703704 245.778813                    hurst_w_15m|price_sigma_w_5m|body_5m 65.140591
       hurst_w        1h    swing_noise_w   1m            z_se_w  15m   UP_SMOOTH         2    2    2 371   84.665094    19.840890  64.824204  0.681941 245.480178                  hurst_w_1h|swing_noise_w_1m|z_se_w_15m 64.824204
     bar_range        1h    swing_noise_w   1m    vol_velocity_w  15m   UP_SMOOTH         2    1    1 104   83.980769    19.840890  64.139879  0.663462 273.194611        bar_range_1h|swing_noise_w_1m|vol_velocity_w_15m 64.139879
       hurst_w        5m price_velocity_w   5m        vol_mean_w   5m   UP_SMOOTH         2    2    0 126   83.571429    19.840890  63.730539  0.682540 281.571669            hurst_w_5m|price_velocity_w_5m|vol_mean_w_5m 63.730539
    vol_mean_w        1h        bar_range   5m price_velocity_1b  15m   UP_CHOPPY         2    2    0 228   74.607456    10.986748  63.620708  0.679825 156.479536        vol_mean_w_1h|bar_range_5m|price_velocity_1b_15m 63.620708
    vol_mean_w        1h       vol_mean_w   5m price_velocity_1b  15m   UP_CHOPPY         2    2    0 204   74.485294    10.986748  63.498546  0.735294 157.261712       vol_mean_w_1h|vol_mean_w_5m|price_velocity_1b_15m 63.498546
    vol_mean_w        1h    price_sigma_w   5m price_velocity_1b  15m   UP_CHOPPY         2    2    0 215   74.019767    10.986748  63.033019  0.641860 160.233168    vol_mean_w_1h|price_sigma_w_5m|price_velocity_1b_15m 63.033019
       hurst_w        1h       vol_mean_w   5m            z_se_w  15m   UP_SMOOTH         2    2    2 298   82.691275    19.840890  62.850385  0.624161 246.543228                     hurst_w_1h|vol_mean_w_5m|z_se_w_15m 62.850385
       hurst_w        1h    price_sigma_w   5m    vol_velocity_w  15m   UP_SMOOTH         2    2    0 187   82.663102    19.840890  62.822212  0.684492 259.132438          hurst_w_1h|price_sigma_w_5m|vol_velocity_w_15m 62.822212
    vol_mean_w        1h    swing_noise_w   1m price_velocity_1b  15m   UP_CHOPPY         2    2    0 232   73.679957    10.986748  62.693209  0.676724 154.879113    vol_mean_w_1h|swing_noise_w_1m|price_velocity_1b_15m 62.693209
     bar_range        1h reversion_prob_w  15m            z_se_w  15m   UP_SMOOTH         2    1    2 366   82.493852    19.840890  62.652962  0.713115 241.836466            bar_range_1h|reversion_prob_w_15m|z_se_w_15m 62.652962
       hurst_w        1m        bar_range   5m    vol_velocity_w  15m DOWN_SMOOTH         2    2    0 152  -84.384868   -21.819873 -62.564995  0.296053 186.286150              hurst_w_1m|bar_range_5m|vol_velocity_w_15m 62.564995
     bar_range        1h reversion_prob_w  15m price_velocity_1b  15m   UP_CHOPPY         2    0    0 141   73.340426    10.986748  62.353677  0.808511 129.379589 bar_range_1h|reversion_prob_w_15m|price_velocity_1b_15m 62.353677
       hurst_w        1m price_velocity_w   5m        vol_mean_w   5m   UP_CHOPPY         1    0    2 147   73.142857    10.986748  62.156109  0.802721 131.181708            hurst_w_1m|price_velocity_w_5m|vol_mean_w_5m 62.156109
       hurst_w        1m    swing_noise_w   1m    vol_velocity_w  15m DOWN_SMOOTH         2    2    0 156  -83.719551   -21.819873 -61.899678  0.294872 192.575949          hurst_w_1m|swing_noise_w_1m|vol_velocity_w_15m 61.899678
       hurst_w       15m       vol_mean_w   5m  reversion_prob_w  15m   UP_SMOOTH         2    2    1 197   81.054569    19.840890  61.213679  0.670051 269.611311          hurst_w_15m|vol_mean_w_5m|reversion_prob_w_15m 61.213679
       hurst_w        1h price_velocity_w   5m     price_sigma_w   5m   UP_SMOOTH         2    2    2 480   80.840104    19.840890  60.999214  0.639583 246.850320         hurst_w_1h|price_velocity_w_5m|price_sigma_w_5m 60.999214
    vol_mean_w        1h    price_sigma_w   5m            z_se_w  15m   UP_SMOOTH         0    2    1 102   79.818627    19.840890  59.977737  0.676471 217.005156               vol_mean_w_1h|price_sigma_w_5m|z_se_w_15m 59.977737
       hurst_w        5m    price_sigma_w   5m  reversion_prob_w  15m DOWN_SMOOTH         0    2    1 121  -81.510331   -21.819873 -59.690457  0.289256 175.811127        hurst_w_5m|price_sigma_w_5m|reversion_prob_w_15m 59.690457
     bar_range        1h    price_sigma_w   5m    vol_velocity_w  15m DOWN_SMOOTH         2    2    1  82  -81.283537   -21.819873 -59.463663  0.304878 121.765604        bar_range_1h|price_sigma_w_5m|vol_velocity_w_15m 59.463663
    vol_mean_w        1h price_velocity_w   5m        vol_mean_w   5m   UP_SMOOTH         0    2    0 169   79.075444    19.840890  59.234554  0.686391 267.453594         vol_mean_w_1h|price_velocity_w_5m|vol_mean_w_5m 59.234554
       hurst_w        1h price_velocity_w   5m         bar_range   5m   UP_SMOOTH         2    2    2 389   79.035347    19.840890  59.194457  0.670951 236.921783             hurst_w_1h|price_velocity_w_5m|bar_range_5m 59.194457
    vol_mean_w        1h    price_sigma_w   5m  reversion_prob_w  15m   UP_SMOOTH         0    2    2 110   78.979545    19.840890  59.138655  0.700000 208.954127     vol_mean_w_1h|price_sigma_w_5m|reversion_prob_w_15m 59.138655
    vol_mean_w        1h   vol_velocity_w  15m            z_se_w  15m   UP_CHOPPY         2    2    0  90   69.497222    10.986748  58.510474  0.677778 150.426900             vol_mean_w_1h|vol_velocity_w_15m|z_se_w_15m 58.510474
    vol_mean_w        1h       vol_mean_w   5m  reversion_prob_w  15m DOWN_SMOOTH         0    2    2  81  -80.120370   -21.819873 -58.300497  0.283951 101.359224        vol_mean_w_1h|vol_mean_w_5m|reversion_prob_w_15m 58.300497
    vol_mean_w        1h    price_sigma_w   5m    vol_velocity_w  15m   UP_SMOOTH         1    2    0  85  -38.305882    19.840890 -58.146772  0.352941  93.006331       vol_mean_w_1h|price_sigma_w_5m|vol_velocity_w_15m 58.146772
