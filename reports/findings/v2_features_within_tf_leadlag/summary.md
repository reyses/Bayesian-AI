# V2 features within-TF lead-lag (Layer D5) - 2026-05-03 19:59 UTC

**Pairs**: 253  **TFs**: 5  **Regimes**: 6  **Shifts**: [-12, -8, -4, -2, -1, 0, 1, 2, 4, 8, 12]

## Role distribution

- X_leads_Y: 2635 (34.7%)
- contemporaneous: 2481 (32.7%)
- X_lags_Y: 2474 (32.6%)

## Top 50 by |peak corr|

 tf                c1       c2   regime_2d  peak_shift  peak_corr  contemp_corr            role  abs_peak
 5s      price_mean_w   vwap_w FLAT_SMOOTH           0   1.000000      1.000000 contemporaneous  1.000000
 5s      price_mean_w   vwap_w DOWN_CHOPPY           0   1.000000      1.000000 contemporaneous  1.000000
 5s      price_mean_w   vwap_w   UP_SMOOTH           0   1.000000      1.000000 contemporaneous  1.000000
 5s      price_mean_w   vwap_w DOWN_SMOOTH           0   1.000000      1.000000 contemporaneous  1.000000
 5s      price_mean_w   vwap_w FLAT_CHOPPY           0   1.000000      1.000000 contemporaneous  1.000000
 5s      price_mean_w   vwap_w   UP_CHOPPY           0   1.000000      1.000000 contemporaneous  1.000000
 1m      price_mean_w   vwap_w DOWN_CHOPPY           0   0.999997      0.999997 contemporaneous  0.999997
 1m      price_mean_w   vwap_w FLAT_CHOPPY           0   0.999997      0.999997 contemporaneous  0.999997
 1m      price_mean_w   vwap_w   UP_CHOPPY           0   0.999995      0.999995 contemporaneous  0.999995
 1m      price_mean_w   vwap_w   UP_SMOOTH           0   0.999995      0.999995 contemporaneous  0.999995
 1m      price_mean_w   vwap_w FLAT_SMOOTH           0   0.999993      0.999993 contemporaneous  0.999993
 5m      price_mean_w   vwap_w DOWN_CHOPPY          -1   0.999992      0.999991        X_lags_Y  0.999992
 5m      price_mean_w   vwap_w FLAT_CHOPPY           0   0.999990      0.999990 contemporaneous  0.999990
 5m      price_mean_w   vwap_w   UP_SMOOTH          -1   0.999988      0.999983        X_lags_Y  0.999988
 5m      price_mean_w   vwap_w   UP_CHOPPY          -1   0.999987      0.999986        X_lags_Y  0.999987
 1m      price_mean_w   vwap_w DOWN_SMOOTH           0   0.999987      0.999987 contemporaneous  0.999987
 5m      price_mean_w   vwap_w DOWN_SMOOTH          -1   0.999983      0.999975        X_lags_Y  0.999983
 5m      price_mean_w   vwap_w FLAT_SMOOTH          -2   0.999982      0.999970        X_lags_Y  0.999982
15m      price_mean_w   vwap_w DOWN_CHOPPY          -2   0.999965      0.999962        X_lags_Y  0.999965
15m      price_mean_w   vwap_w FLAT_CHOPPY           0   0.999947      0.999947 contemporaneous  0.999947
15m      price_mean_w   vwap_w   UP_CHOPPY          -4   0.999936      0.999930        X_lags_Y  0.999936
15m      price_mean_w   vwap_w   UP_SMOOTH          -2   0.999935      0.999925        X_lags_Y  0.999935
15m      price_mean_w   vwap_w FLAT_SMOOTH          -4   0.999929      0.999892        X_lags_Y  0.999929
15m      price_mean_w   vwap_w DOWN_SMOOTH           0   0.999904      0.999904 contemporaneous  0.999904
 1h      price_mean_w   vwap_w DOWN_CHOPPY         -12   0.999700      0.999686        X_lags_Y  0.999700
 1h      price_mean_w   vwap_w   UP_CHOPPY           0   0.999629      0.999629 contemporaneous  0.999629
 1h      price_mean_w   vwap_w FLAT_CHOPPY           0   0.999580      0.999580 contemporaneous  0.999580
 1h      price_mean_w   vwap_w   UP_SMOOTH           0   0.999468      0.999468 contemporaneous  0.999468
 1h      price_mean_w   vwap_w FLAT_SMOOTH         -12   0.999434      0.999333        X_lags_Y  0.999434
15m price_velocity_1b     body   UP_CHOPPY           0   0.999370      0.999370 contemporaneous  0.999370
 5m price_velocity_1b     body   UP_CHOPPY           0   0.999161      0.999161 contemporaneous  0.999161
 1h      price_mean_w   vwap_w DOWN_SMOOTH           0   0.998969      0.998969 contemporaneous  0.998969
 1m price_velocity_1b     body   UP_CHOPPY           0   0.998865      0.998865 contemporaneous  0.998865
 1h price_velocity_1b     body   UP_CHOPPY           0   0.998844      0.998844 contemporaneous  0.998844
 1m price_velocity_1b     body DOWN_SMOOTH           0   0.997848      0.997848 contemporaneous  0.997848
 1m price_velocity_1b     body DOWN_CHOPPY           0   0.997771      0.997771 contemporaneous  0.997771
15m price_velocity_1b     body FLAT_CHOPPY           0   0.997583      0.997583 contemporaneous  0.997583
 1h price_velocity_1b     body FLAT_CHOPPY           0   0.997510      0.997510 contemporaneous  0.997510
 1m price_velocity_1b     body   UP_SMOOTH           0   0.997207      0.997207 contemporaneous  0.997207
 5m price_velocity_1b     body FLAT_CHOPPY           0   0.997097      0.997097 contemporaneous  0.997097
 1m price_velocity_1b     body FLAT_CHOPPY           0   0.996876      0.996876 contemporaneous  0.996876
 1m price_velocity_1b     body FLAT_SMOOTH           0   0.995465      0.995465 contemporaneous  0.995465
 5m price_velocity_1b     body DOWN_CHOPPY           0   0.989338      0.989338 contemporaneous  0.989338
15m price_velocity_1b     body DOWN_CHOPPY           0   0.988847      0.988847 contemporaneous  0.988847
 1h price_velocity_1b     body DOWN_CHOPPY           0   0.988601      0.988601 contemporaneous  0.988601
 1h price_velocity_1b     body   UP_SMOOTH           0   0.980218      0.980218 contemporaneous  0.980218
 1h         SE_high_w SE_low_w   UP_SMOOTH          12   0.975016      0.939404       X_leads_Y  0.975016
 1m         SE_high_w SE_low_w   UP_CHOPPY           0   0.974146      0.974146 contemporaneous  0.974146
15m price_velocity_1b     body   UP_SMOOTH           0   0.972322      0.972322 contemporaneous  0.972322
 5m price_velocity_1b     body DOWN_SMOOTH           0   0.969783      0.969783 contemporaneous  0.969783

## Genuine leaders (peak_shift > 0)

 tf                c1               c2   regime_2d  peak_shift  peak_corr  contemp_corr      role  abs_peak
 1h            z_se_w         z_high_w DOWN_SMOOTH          12   0.859931      0.736956 X_leads_Y  0.859931
 1h            z_se_w          z_low_w   UP_SMOOTH          12   0.855409      0.774467 X_leads_Y  0.855409
15m         bar_range        SE_high_w   UP_CHOPPY          12   0.826090      0.735718 X_leads_Y  0.826090
 1h            z_se_w         z_high_w DOWN_CHOPPY          12   0.809313      0.722887 X_leads_Y  0.809313
15m         bar_range    price_sigma_w   UP_CHOPPY          12   0.809192      0.728279 X_leads_Y  0.809192
15m         bar_range         SE_low_w   UP_CHOPPY          12   0.798789      0.707108 X_leads_Y  0.798789
 5m         bar_range         SE_low_w   UP_SMOOTH           2   0.794490      0.678606 X_leads_Y  0.794490
 5m         bar_range    price_sigma_w   UP_SMOOTH           2   0.789889      0.713101 X_leads_Y  0.789889
 1m         bar_range         SE_low_w   UP_SMOOTH           1   0.780948      0.664254 X_leads_Y  0.780948
 1m         bar_range    price_sigma_w   UP_SMOOTH           1   0.752184      0.650869 X_leads_Y  0.752184
 1h     price_sigma_w        SE_high_w DOWN_CHOPPY          12   0.749108      0.672168 X_leads_Y  0.749108
 5m         bar_range        SE_high_w DOWN_SMOOTH           4   0.748104      0.674847 X_leads_Y  0.748104
15m         bar_range        SE_high_w DOWN_SMOOTH           8   0.728897      0.630932 X_leads_Y  0.728897
15m         bar_range    price_sigma_w   UP_SMOOTH          12   0.718482      0.594040 X_leads_Y  0.718482
15m         bar_range    price_sigma_w DOWN_SMOOTH          12   0.715614      0.643865 X_leads_Y  0.715614
15m         bar_range    price_sigma_w FLAT_CHOPPY          12   0.704819      0.610446 X_leads_Y  0.704819
15m         bar_range         SE_low_w   UP_SMOOTH           8   0.704815      0.551312 X_leads_Y  0.704815
15m         bar_range       vol_mean_w DOWN_CHOPPY          12   0.702004      0.589671 X_leads_Y  0.702004
 5m         bar_range        SE_high_w DOWN_CHOPPY           1   0.690763      0.615409 X_leads_Y  0.690763
15m         bar_range         SE_low_w FLAT_CHOPPY           8   0.688907      0.612736 X_leads_Y  0.688907
 5m     price_sigma_w    swing_noise_w FLAT_SMOOTH          12   0.682187      0.605930 X_leads_Y  0.682187
15m         bar_range        SE_high_w FLAT_CHOPPY           8   0.674952      0.576720 X_leads_Y  0.674952
15m         bar_range        SE_high_w DOWN_CHOPPY           8   0.670658      0.592413 X_leads_Y  0.670658
 5m         SE_high_w    swing_noise_w FLAT_SMOOTH          12   0.663751      0.602654 X_leads_Y  0.663751
15m         bar_range       vol_mean_w FLAT_SMOOTH          12   0.658842      0.581637 X_leads_Y  0.658842
15m         bar_range    price_sigma_w DOWN_CHOPPY          12   0.656541      0.588297 X_leads_Y  0.656541
15m         bar_range       vol_mean_w DOWN_SMOOTH          12   0.641679      0.550645 X_leads_Y  0.641679
 1m        vol_mean_w         SE_low_w FLAT_SMOOTH           4   0.625056      0.460929 X_leads_Y  0.625056
15m         bar_range       vol_mean_w   UP_CHOPPY          12   0.620529      0.503455 X_leads_Y  0.620529
15m         bar_range       vol_mean_w FLAT_CHOPPY          12   0.619928      0.550318 X_leads_Y  0.619928
 1h         bar_range    price_sigma_w DOWN_SMOOTH          12   0.619623      0.551825 X_leads_Y  0.619623
 1m         bar_range    swing_noise_w FLAT_SMOOTH           4   0.618089      0.552392 X_leads_Y  0.618089
 1m        vol_mean_w        SE_high_w FLAT_SMOOTH           4   0.607877      0.447185 X_leads_Y  0.607877
15m         bar_range    swing_noise_w FLAT_CHOPPY          12   0.602374      0.532372 X_leads_Y  0.602374
 1m         bar_range        SE_high_w FLAT_SMOOTH           4   0.594872      0.460009 X_leads_Y  0.594872
 1m         bar_range         SE_low_w FLAT_SMOOTH           4   0.594103      0.460476 X_leads_Y  0.594103
15m         bar_range    swing_noise_w DOWN_CHOPPY          12   0.580215      0.510504 X_leads_Y  0.580215
 5m         bar_range       vol_mean_w   UP_SMOOTH           4   0.579393      0.517218 X_leads_Y  0.579393
 5m         bar_range    swing_noise_w FLAT_SMOOTH          12   0.566918      0.473866 X_leads_Y  0.566918
 1h price_velocity_1b         z_high_w DOWN_CHOPPY          12   0.565518      0.273422 X_leads_Y  0.565518
 1h         bar_range        SE_high_w DOWN_SMOOTH          12   0.563792      0.432681 X_leads_Y  0.563792
 1h              body         z_high_w DOWN_CHOPPY          12   0.563144      0.262926 X_leads_Y  0.563144
 5m    price_accel_1b    price_accel_w FLAT_SMOOTH           8   0.561012      0.499770 X_leads_Y  0.561012
 1h         bar_range        SE_high_w   UP_SMOOTH          12   0.560688      0.492602 X_leads_Y  0.560688
 5m       vol_sigma_w    swing_noise_w DOWN_CHOPPY          12   0.560258      0.456511 X_leads_Y  0.560258
 5m         bar_range        SE_high_w FLAT_SMOOTH           8   0.558816      0.407371 X_leads_Y  0.558816
 5m       vol_sigma_w    swing_noise_w DOWN_SMOOTH          12   0.554079      0.428903 X_leads_Y  0.554079
15m         bar_range price_velocity_w   UP_SMOOTH          12   0.553971      0.462450 X_leads_Y  0.553971
 1m         bar_range    price_sigma_w FLAT_SMOOTH           4   0.550427      0.444701 X_leads_Y  0.550427
15m         bar_range    swing_noise_w   UP_SMOOTH          12   0.548846      0.495881 X_leads_Y  0.548846
