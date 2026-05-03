# Layer C1 triplet OOS validation - 2026-05-03 18:03 UTC

**Method:** for each top-K IS triplet cell per regime,
reuse IS regime-local quantile boundaries on OOS data,
measure cell-mean / win-rate / n on OOS, compute lift vs OOS regime baseline.

**Survival rule:** sign(OOS lift) == sign(IS lift) AND |OOS lift| >= 0.3 * |IS lift| AND n_oos >= 20.

## Survival summary

- Total IS top cells: **120**
- With OOS data: **114**
- **Survive OOS: 31 (25.8%)**
- Sign flips: 49
- Magnitude too low: 34
- No OOS data: 6

## OOS regime baselines

- UP_SMOOTH: +18.31
- UP_CHOPPY: +12.73
- DOWN_SMOOTH: -23.87
- DOWN_CHOPPY: -16.55
- FLAT_SMOOTH: -0.43
- FLAT_CHOPPY: -0.98

## Top 30 IS cells with OOS validation

anchor_concept anchor_tf        x_concept x_tf         y_concept y_tf   regime_2d  cell  is_n     is_mean    is_lift    is_wr  oos_n   oos_mean   oos_lift   oos_wr   oos_ci_lo  oos_ci_hi  oos_baseline  survives      reason  abs_is_lift
       hurst_w       15m    price_sigma_w   5m    vol_velocity_w  15m   UP_SMOOTH 1,2,1    96  119.744792  99.903902 0.645833     84  14.982143  -3.331225 0.642857    3.663244  26.443676     18.313368     False   sign_flip    99.903902
     bar_range        1h price_velocity_w   5m    vol_velocity_w  15m   UP_SMOOTH 2,2,1   123  114.922764  95.081874 0.723577     90  11.397222  -6.916146 0.566667    1.213542  23.066736     18.313368     False   sign_flip    95.081874
    vol_mean_w        1h    price_sigma_w   5m    vol_velocity_w  15m   UP_SMOOTH 0,2,1   120  114.166667  94.325777 0.741667     73  17.630137  -0.683231 0.575342    8.717808  27.011216     18.313368     False   sign_flip    94.325777
       hurst_w       15m price_velocity_w   5m        vol_mean_w   5m   UP_SMOOTH 1,2,0   111  113.907658  94.066768 0.594595     94  21.159574   2.846206 0.680851   13.190559  29.637101     18.313368     False mag_too_low    94.066768
       hurst_w        1h price_velocity_w   5m  reversion_prob_w  15m   UP_SMOOTH 2,2,1   211  113.613744  93.772854 0.710900    120  21.960417   3.647049 0.658333   11.275469  34.345521     18.313368     False mag_too_low    93.772854
       hurst_w        5m    price_sigma_w   5m    vol_velocity_w  15m   UP_SMOOTH 2,2,1    93  109.846774  90.005884 0.623656     92   7.728261 -10.585107 0.521739    0.844837  14.666033     18.313368     False   sign_flip    90.005884
       hurst_w        1h    price_sigma_w   5m  reversion_prob_w  15m   UP_SMOOTH 2,2,1   227  109.838106  89.997216 0.660793    116  22.650862   4.337494 0.560345   10.419504  36.337554     18.313368     False mag_too_low    89.997216
     bar_range        1h        bar_range   5m    vol_velocity_w  15m DOWN_SMOOTH 2,2,1    82 -104.881098 -83.061224 0.268293     70 -12.135714  11.730811 0.485714  -29.086250   3.866429    -23.866525     False   sign_flip    83.061224
     bar_range        1h    swing_noise_w   1m    vol_velocity_w  15m DOWN_SMOOTH 2,2,1    81 -103.098765 -81.278892 0.296296     69 -36.181159 -12.314634 0.275362  -51.777899 -21.402174    -23.866525     False mag_too_low    81.278892
       hurst_w        1h    swing_noise_w   1m  reversion_prob_w  15m   UP_SMOOTH 2,2,1   230   98.242391  78.401501 0.665217    127  21.285433   2.972065 0.566929    9.852264  33.803199     18.313368     False mag_too_low    78.401501
       hurst_w        1h        bar_range   5m    vol_velocity_w  15m DOWN_SMOOTH 2,2,1    88 -100.079545 -78.259672 0.193182     57 -13.355263  10.511262 0.491228  -33.653618   5.292434    -23.866525     False   sign_flip    78.259672
       hurst_w        1h    price_sigma_w   5m    vol_velocity_w  15m   UP_SMOOTH 2,2,1   137   97.682482  77.841592 0.700730     75  19.510000   1.196632 0.560000    8.139750  31.414750     18.313368     False mag_too_low    77.841592
       hurst_w        1h price_velocity_w   5m        vol_mean_w   5m   UP_SMOOTH 2,2,0   144   95.147569  75.306679 0.729167     44  35.465909  17.152541 0.681818   19.033949  53.108239     18.313368     False mag_too_low    75.306679
    vol_mean_w        1h   vol_velocity_w  15m price_velocity_1b  15m   UP_CHOPPY 2,2,0   117   82.955128  71.968380 0.675214     48  -0.338542 -13.068388 0.666667  -25.425781  21.146484     12.729846     False   sign_flip    71.968380
     bar_range        1h        bar_range   5m    vol_velocity_w  15m   UP_SMOOTH 2,1,1   114   91.269737  71.428847 0.780702     68  14.452206  -3.861162 0.529412    1.450643  29.309835     18.313368     False   sign_flip    71.428847
       hurst_w       15m       vol_mean_w   5m            z_se_w  15m   UP_SMOOTH 2,2,2   200   90.931250  71.090360 0.725000     57  23.521930   5.208562 0.701754   12.596491  34.017763     18.313368     False mag_too_low    71.090360
    vol_mean_w        1h        bar_range   5m    vol_velocity_w  15m DOWN_SMOOTH 0,2,1    82  -92.451220 -70.631346 0.231707     88 -29.772727  -5.906202 0.409091  -47.053622 -13.541903    -23.866525     False mag_too_low    70.631346
    vol_mean_w        1h    swing_noise_w   1m    vol_velocity_w  15m   UP_SMOOTH 1,2,0    88  -50.093750 -69.934640 0.318182     63   9.055556  -9.257813 0.507937   -0.234524  18.619444     18.313368     False mag_too_low    69.934640
       hurst_w        1h reversion_prob_w  15m price_velocity_1b  15m   UP_SMOOTH 2,1,2   226   88.544248  68.703358 0.615044     91  27.554945   9.241577 0.637363   14.194162  42.388393     18.313368     False mag_too_low    68.703358
       hurst_w       15m price_velocity_w   5m    vol_velocity_w  15m   UP_SMOOTH 1,2,1   149   86.805369  66.964479 0.630872    139   8.052158 -10.261210 0.589928   -0.915737  17.150180     18.313368     False   sign_flip    66.964479
    vol_mean_w        1h    swing_noise_w   1m            z_se_w  15m DOWN_SMOOTH 0,2,1    88  -88.366477 -66.546604 0.272727     37 -70.756757 -46.890231 0.270270 -103.687331 -39.519932    -23.866525      True    survives    66.546604
       hurst_w        1h    price_sigma_w   5m            z_se_w  15m   UP_SMOOTH 2,2,2   390   86.153205  66.312315 0.656410    164  36.379573  18.066205 0.646341   25.288872  48.047790     18.313368     False mag_too_low    66.312315
       hurst_w        1h        bar_range   5m            z_se_w  15m   UP_SMOOTH 2,2,2   310   85.815323  65.974433 0.703226    145  38.827586  20.514218 0.696552   27.506724  51.273017     18.313368      True    survives    65.974433
    vol_mean_w        1h    swing_noise_w   1m  reversion_prob_w  15m DOWN_SMOOTH 0,2,2    88  -87.457386 -65.637513 0.295455     37 -70.756757 -46.890231 0.270270 -103.687331 -39.519932    -23.866525      True    survives    65.637513
       hurst_w       15m    price_sigma_w   5m              body   5m   UP_SMOOTH 1,2,1   108   84.981481  65.140591 0.703704     62  27.846774   9.533406 0.661290   11.374294  49.022782     18.313368     False mag_too_low    65.140591
       hurst_w        1h    swing_noise_w   1m            z_se_w  15m   UP_SMOOTH 2,2,2   371   84.665094  64.824204 0.681941    155  37.890323  19.576955 0.664516   26.073347  51.009839     18.313368      True    survives    64.824204
     bar_range        1h    swing_noise_w   1m    vol_velocity_w  15m   UP_SMOOTH 2,1,1   104   83.980769  64.139879 0.663462     61   0.430328 -17.883040 0.491803   -9.398361  10.789754     18.313368     False   sign_flip    64.139879
    vol_mean_w        1h        bar_range   5m price_velocity_1b  15m   UP_CHOPPY 2,2,0   228   74.607456  63.620708 0.679825     83  20.225904   7.496058 0.662651   -1.739985  42.656325     12.729846     False mag_too_low    63.620708
    vol_mean_w        1h       vol_mean_w   5m price_velocity_1b  15m   UP_CHOPPY 2,2,0   204   74.485294  63.498546 0.735294     81  24.089506  11.359660 0.691358    3.745062  46.895370     12.729846     False mag_too_low    63.498546
    vol_mean_w        1h    price_sigma_w   5m price_velocity_1b  15m   UP_CHOPPY 2,2,0   215   74.019767  63.033019 0.641860     75  26.863333  14.133487 0.680000    5.532917  50.236667     12.729846     False mag_too_low    63.033019

## Surviving cells

anchor_concept anchor_tf        x_concept x_tf         y_concept y_tf   regime_2d  cell  is_n    is_mean    is_lift    is_wr  oos_n   oos_mean   oos_lift   oos_wr   oos_ci_lo  oos_ci_hi  oos_baseline  survives   reason
    vol_mean_w        1h    swing_noise_w   1m            z_se_w  15m DOWN_SMOOTH 0,2,1    88 -88.366477 -66.546604 0.272727     37 -70.756757 -46.890231 0.270270 -103.687331 -39.519932    -23.866525      True survives
       hurst_w        1h        bar_range   5m            z_se_w  15m   UP_SMOOTH 2,2,2   310  85.815323  65.974433 0.703226    145  38.827586  20.514218 0.696552   27.506724  51.273017     18.313368      True survives
    vol_mean_w        1h    swing_noise_w   1m  reversion_prob_w  15m DOWN_SMOOTH 0,2,2    88 -87.457386 -65.637513 0.295455     37 -70.756757 -46.890231 0.270270 -103.687331 -39.519932    -23.866525      True survives
       hurst_w        1h    swing_noise_w   1m            z_se_w  15m   UP_SMOOTH 2,2,2   371  84.665094  64.824204 0.681941    155  37.890323  19.576955 0.664516   26.073347  51.009839     18.313368      True survives
       hurst_w        5m    price_sigma_w   5m  reversion_prob_w  15m DOWN_SMOOTH 0,2,1   121 -81.510331 -59.690457 0.289256     54 -56.907407 -33.040882 0.240741  -74.656366 -39.184491    -23.866525      True survives
       hurst_w       15m   vol_velocity_w  15m        vol_mean_w   5m DOWN_SMOOTH 0,0,2   183 -78.550546 -56.730673 0.366120     21 -57.821429 -33.954903 0.190476  -77.502976 -37.332738    -23.866525      True survives
       hurst_w       15m       vol_mean_w   5m            z_se_w  15m DOWN_SMOOTH 0,2,0   197 -77.907360 -56.087487 0.319797     25 -50.230000 -26.363475 0.200000  -98.307250  -6.005000    -23.866525      True survives
       hurst_w       15m       vol_mean_w   5m  reversion_prob_w  15m DOWN_SMOOTH 0,2,1   170 -77.579412 -55.759538 0.323529     31 -57.870968 -34.004442 0.096774  -70.818548 -44.255444    -23.866525      True survives
       hurst_w       15m    swing_noise_w   1m            z_se_w  15m DOWN_SMOOTH 0,2,1   161 -77.527950 -55.708077 0.329193     24 -65.937500 -42.070975 0.125000  -85.406250 -46.842969    -23.866525      True survives
       hurst_w       15m    swing_noise_w   1m  reversion_prob_w  15m DOWN_SMOOTH 0,2,2   161 -77.527950 -55.708077 0.329193     24 -65.937500 -42.070975 0.125000  -85.406250 -46.842969    -23.866525      True survives
       hurst_w        5m        bar_range   5m  reversion_prob_w  15m DOWN_SMOOTH 0,2,1   129 -77.032946 -55.213072 0.341085     57 -51.811404 -27.944878 0.368421  -76.239145 -29.124671    -23.866525      True survives
    vol_mean_w        1h   vol_velocity_w  15m        vol_mean_w   5m DOWN_CHOPPY 2,2,2   117 -68.070513 -54.473542 0.307692     65 -57.100000 -40.548198 0.338462  -81.929519 -30.429712    -16.551802      True survives
    vol_mean_w        1h    swing_noise_w   1m    vol_velocity_w  15m DOWN_CHOPPY 2,2,2   125 -54.138000 -40.541029 0.368000     86 -47.098837 -30.547035 0.360465  -67.081613 -27.273837    -16.551802      True survives
    vol_mean_w        1h        bar_range   5m    vol_velocity_w  15m DOWN_CHOPPY 2,2,2   123 -51.691057 -38.094086 0.365854     82 -49.960366 -33.408564 0.341463  -70.704726 -28.876296    -16.551802      True survives
       hurst_w        1h        bar_range   5m    vol_velocity_w  15m FLAT_SMOOTH 0,2,0   337  22.114985  22.145736 0.649852     23   9.663043  10.091562 0.521739   -5.239402  24.815761     -0.428519      True survives
       hurst_w       15m   vol_velocity_w  15m        vol_mean_w   5m FLAT_CHOPPY 1,1,2    86  21.276163  21.565255 0.558140     28   5.589286   6.565222 0.428571  -13.468527  25.761607     -0.975937      True survives
    vol_mean_w        1h    swing_noise_w   1m price_velocity_1b  15m FLAT_CHOPPY 0,2,1   148 -21.775338 -21.486246 0.405405     61 -16.303279 -15.327342 0.475410  -39.862295   7.057480     -0.975937      True survives
       hurst_w        1h       vol_mean_w   5m price_velocity_1b  15m FLAT_CHOPPY 2,2,1   254 -19.612205 -19.323113 0.429134    200 -11.221250 -10.245313 0.460000  -21.319187  -1.829688     -0.975937      True survives
       hurst_w       15m price_velocity_w   5m        vol_mean_w   5m FLAT_SMOOTH 0,1,2   157 -19.052548 -19.021797 0.445860     40 -12.812500 -12.383981 0.325000  -26.325156   0.384219     -0.428519      True survives
       hurst_w        1h    swing_noise_w   1m    vol_velocity_w  15m FLAT_SMOOTH 0,2,0   356  18.960674  18.991425 0.617978     22  10.443182  10.871700 0.545455   -4.694886  26.170455     -0.428519      True survives
       hurst_w        1h price_velocity_w   5m    vol_velocity_w  15m FLAT_CHOPPY 0,0,2   755  18.565232  18.854324 0.637086    346   9.130780  10.106717 0.528902    0.578721  17.755943     -0.975937      True survives
       hurst_w       15m   vol_velocity_w  15m            z_se_w  15m FLAT_SMOOTH 0,2,2   327 -18.325688 -18.294937 0.366972     53  -7.867925  -7.439406 0.603774  -26.481840   9.873349     -0.428519      True survives
     bar_range        1h price_velocity_w   5m            z_se_w  15m FLAT_CHOPPY 2,0,2   219  17.996575  18.285667 0.639269    164   6.057927   7.033864 0.567073   -4.015549  16.333918     -0.975937      True survives
       hurst_w        1m price_velocity_w   5m price_velocity_1b  15m FLAT_SMOOTH 1,2,0   138 -17.873188 -17.842437 0.427536     26 -15.894231 -15.465712 0.500000  -39.252885   4.740385     -0.428519      True survives
       hurst_w       15m    price_sigma_w   5m    vol_velocity_w  15m FLAT_SMOOTH 1,2,0   379  17.750000  17.780751 0.614776     40  10.306250  10.734769 0.600000    1.999375  19.700156     -0.428519      True survives
       hurst_w        1h    price_sigma_w   5m    vol_velocity_w  15m FLAT_SMOOTH 0,2,0   357  16.997199  17.027950 0.599440     21  11.976190  12.404709 0.619048   -3.666667  27.309821     -0.428519      True survives
       hurst_w       15m    swing_noise_w   1m    vol_velocity_w  15m FLAT_SMOOTH 1,2,0   402  16.893035  16.923786 0.624378     33  14.492424  14.920943 0.606061    3.951515  26.531629     -0.428519      True survives
       hurst_w       15m price_velocity_w   5m    vol_velocity_w  15m FLAT_SMOOTH 0,2,2   354 -16.877825 -16.847074 0.398305     53 -14.679245 -14.250727 0.547170  -32.416863   3.737382     -0.428519      True survives
       hurst_w       15m        bar_range   5m    vol_velocity_w  15m FLAT_SMOOTH 1,2,0   414  16.535024  16.565775 0.623188     40  16.712500  17.141019 0.675000    5.643594  27.559062     -0.428519      True survives
       hurst_w        5m price_velocity_w   5m    vol_velocity_w  15m FLAT_SMOOTH 0,2,2   392 -16.339923 -16.309173 0.413265     69 -19.677536 -19.249018 0.362319  -31.101812  -8.564855     -0.428519      True survives
       hurst_w       15m price_velocity_w   5m    vol_velocity_w  15m FLAT_SMOOTH 1,0,0   334  16.196856  16.227607 0.598802     29  12.801724  13.230243 0.586207    1.369181  24.820474     -0.428519      True survives
