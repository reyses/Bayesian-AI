**DEPRECATED — LOOKAHEAD ARTIFACT, DO NOT QUOTE**

# V2 features within-day persistence (Step #2) - 2026-05-03 17:43 UTC

**Concepts:** ['price_sigma_w', 'bar_range', 'vol_mean_w', 'vol_velocity_w', 'price_velocity_w', 'body', 'price_velocity_1b', 'swing_noise_w']

**TFs:** ['5s', '1m', '5m', '15m', '1h']

**Tracking Q4 of 5 | max lag 20**

## Top 30 most-persistent (concept, tf, regime)

         concept  tf   regime_2d  p_lag_1  p_lag_5  p_lag_10  half_life_k half_life_str  asymptote
   swing_noise_w  1h DOWN_SMOOTH 0.998924 0.994451  0.988453           21           >20   0.975124
   swing_noise_w  1h DOWN_CHOPPY 0.998325 0.991453  0.982456           21           >20   0.963504
   swing_noise_w  1h   UP_CHOPPY 0.997543 0.987593  0.974874           21           >20   0.948454
   swing_noise_w  1h FLAT_CHOPPY 0.996733 0.983448  0.966339           21           >20   0.934978
   swing_noise_w  1h FLAT_SMOOTH 0.995842 0.978858  0.956803           21           >20   0.909707
   price_sigma_w  1h   UP_SMOOTH 0.994410 0.971326  0.940741           21           >20   0.879365
   swing_noise_w  1h   UP_SMOOTH 0.993882 0.968901  0.936486           21           >20   0.885714
   price_sigma_w  1h FLAT_SMOOTH 0.992292 0.960651  0.919181           21           >20   0.836782
      vol_mean_w  1h FLAT_CHOPPY 0.991794 0.958655  0.916504           21           >20   0.831961
   price_sigma_w  1h FLAT_CHOPPY 0.991530 0.957087  0.912725           21           >20   0.830607
   price_sigma_w  1h DOWN_SMOOTH 0.991416 0.956332  0.910714           21           >20   0.830189
      vol_mean_w  1h   UP_CHOPPY 0.991150 0.955527  0.910486           21           >20   0.827225
   price_sigma_w  1h   UP_CHOPPY 0.991139 0.955243  0.909326           21           >20   0.826316
   swing_noise_w 15m FLAT_CHOPPY 0.990728 0.953783  0.910153           21           >20   0.830857
      vol_mean_w  1h DOWN_SMOOTH 0.990354 0.951140  0.900662           21           >20   0.803653
      vol_mean_w  1h FLAT_SMOOTH 0.990266 0.950930  0.900835           21           >20   0.804301
   price_sigma_w  1h DOWN_CHOPPY 0.989967 0.949153  0.896552           21           >20   0.811594
      vol_mean_w  1h DOWN_CHOPPY 0.989967 0.949153  0.896552           21           >20   0.785714
   swing_noise_w 15m FLAT_SMOOTH 0.989945 0.953181  0.911111           21           >20   0.836286
      vol_mean_w  1h   UP_SMOOTH 0.989547 0.946846  0.891383           21           >20   0.780286
price_velocity_w  1h   UP_CHOPPY 0.988608 0.942455  0.883420           21           >20   0.794737
   swing_noise_w 15m DOWN_SMOOTH 0.988421 0.943255  0.896061           21           >20   0.818078
   swing_noise_w 15m   UP_SMOOTH 0.988186 0.941999  0.889124           21           >20   0.801028
   swing_noise_w 15m DOWN_CHOPPY 0.987692 0.937695  0.873418           21           >20   0.778689
   swing_noise_w 15m   UP_CHOPPY 0.987516 0.936948  0.877395           21           >20   0.765400
      vol_mean_w 15m DOWN_CHOPPY 0.984975 0.924370  0.853242           21           >20   0.706714
      vol_mean_w 15m DOWN_SMOOTH 0.982833 0.920086  0.844809           21           >20   0.701577
  vol_velocity_w  1h FLAT_SMOOTH 0.982564 0.911734  0.820675           21           >20   0.687366
price_velocity_w  1h   UP_SMOOTH 0.982554 0.911032  0.817518           21           >20   0.740654
      vol_mean_w 15m   UP_SMOOTH 0.982505 0.913441  0.834160           21           >20   0.677211

## Run-position forward returns (front-loaded vs distributed)

run_pos                               bar_1    bar_17p      bar_2    bar_3_4    bar_5_8   bar_9_16  delta_b1_to_b5_8     abs_b1
concept          tf  regime_2d                                                                                                 
vol_mean_w       5m  DOWN_SMOOTH -68.441667 -52.833193        NaN -56.632653 -23.980114  -0.706294         44.461553  68.441667
                 1m  DOWN_SMOOTH -54.979167 -41.956316 -61.681373 -63.431250 -40.202500 -13.730539         14.776667  54.979167
price_sigma_w    15m UP_SMOOTH    53.316327  50.155183  47.382653  38.335106  29.193314  21.702166        -24.123013  53.316327
vol_velocity_w   15m DOWN_SMOOTH -48.635802 -22.150134 -45.154321 -53.382479 -53.317073 -16.738994         -4.681271  48.635802
swing_noise_w    5m  UP_SMOOTH    47.994318  46.722387  39.709302  37.530864  38.198630  33.748175         -9.795688  47.994318
vol_mean_w       5s  DOWN_SMOOTH -46.911602 -55.909639 -55.977528 -50.932927 -39.692460 -12.760606          7.219142  46.911602
swing_noise_w    1m  DOWN_SMOOTH -45.252717 -73.140984 -39.089506 -31.808824 -27.123563 -14.847315         18.129154  45.252717
bar_range        5s  DOWN_SMOOTH -44.317568 -74.594891 -62.611940 -46.454327 -21.885204  -1.147619         22.432363  44.317568
price_sigma_w    5m  DOWN_SMOOTH -43.147849 -76.576042 -49.482143 -43.734375 -29.681818  -7.785920         13.466031  43.147849
bar_range        15m DOWN_SMOOTH -43.025701 -68.418624 -38.766355 -31.473333  -9.585821  -1.937956         33.439880  43.025701
                 1h  DOWN_SMOOTH -42.893939 -52.050521 -46.204545 -38.378788 -26.301923   6.096250         16.592016  42.893939
price_velocity_w 15m UP_SMOOTH    37.689103  37.465946  31.134615  27.281022  35.748821  30.428689         -1.940282  37.689103
swing_noise_w    5s  DOWN_SMOOTH -37.090909 -73.471198 -50.544872 -51.868534 -36.317982   1.881250          0.772927  37.090909
vol_velocity_w   1h  UP_SMOOTH    36.975806  39.426955  39.314516  40.608871  33.304435  32.201136         -3.671371  36.975806
                 15m UP_SMOOTH    36.088362  33.988851  33.204741  31.541667  37.646373  64.509073          1.558011  36.088362
price_velocity_w 5m  DOWN_SMOOTH -35.116935        NaN -33.568910 -37.774390 -39.616812 -39.868902         -4.499877  35.116935
price_sigma_w    5s  DOWN_SMOOTH -34.614734 -38.333333 -58.612319 -68.075630 -30.497475 -13.167763          4.117260  34.614734
vol_velocity_w   1m  DOWN_SMOOTH -33.792746        NaN -35.039024 -52.574427        NaN        NaN               NaN  33.792746
bar_range        1m  DOWN_SMOOTH -33.709302 -73.306098 -39.234375 -49.330189 -53.178241 -18.878319        -19.468938  33.709302
                 5m  DOWN_SMOOTH -32.860902 -73.762168 -46.758197 -59.198113 -36.406250   7.239837         -3.545348  32.860902
