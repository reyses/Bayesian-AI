# V2 features: pairwise interaction × regime — 2026-05-03 17:24 UTC

**Base TF:** `5m`  **Split:** `IS`  **Top-K:** 12  **Quantiles:** 3

## Top 30 pairs by corr(X,Y) range across regimes

                                              pair  n_regimes  min_corr_xy  max_corr_xy  corr_xy_range  sign_flip  n_pos_regimes  n_neg_regimes
        L2_1h_price_velocity_w__x__L1_1D_bar_range          6    -0.476748     0.662490       1.139238       True              3              3
      L2_1D_price_velocity_w__x__L2_1D_vol_sigma_w          6    -0.812259     0.179503       0.991762       True              2              4
      L2_1h_price_velocity_w__x__L2_1D_vol_sigma_w          6    -0.428248     0.436965       0.865213       True              2              4
      L2_4h_price_velocity_w__x__L2_1D_vol_sigma_w          6    -0.724563     0.122746       0.847309       True              2              4
        L2_4h_price_velocity_w__x__L1_1D_bar_range          6    -0.539475     0.275500       0.814975       True              2              4
 L2_1h_price_velocity_w__x__L2_1D_price_velocity_w          6    -0.175683     0.566929       0.742612       True              4              2
        L2_1D_price_velocity_w__x__L1_1D_bar_range          6    -0.609950     0.129151       0.739100       True              1              5
 L2_1h_price_velocity_w__x__L2_4h_price_velocity_w          6    -0.003706     0.730386       0.734091       True              5              1
L1_4h_price_velocity_1b__x__L2_4h_price_velocity_w          6    -0.249127     0.436993       0.686120       True              5              1
L1_4h_price_velocity_1b__x__L2_1D_price_velocity_w          6    -0.347913     0.293499       0.641413       True              4              2
             L1_4h_body__x__L2_4h_price_velocity_w          6    -0.247747     0.367201       0.614949       True              5              1
             L1_4h_body__x__L2_1D_price_velocity_w          6    -0.347106     0.230482       0.577588       True              4              2
                L3_4h_z_high_w__x__L1_1D_bar_range          6    -0.324859     0.241186       0.566045       True              4              2
                  L1_4h_body__x__L2_1D_vol_sigma_w          6    -0.243177     0.317971       0.561148       True              2              4
     L1_4h_price_velocity_1b__x__L2_1D_vol_sigma_w          6    -0.238320     0.318635       0.556954       True              2              4
         L1_4h_price_velocity_1b__x__L3_4h_z_low_w          6    -0.024635     0.530543       0.555177       True              5              1
                      L1_4h_body__x__L3_4h_z_low_w          6    -0.035860     0.513314       0.549174       True              5              1
L2_4h_price_velocity_w__x__L2_15m_price_velocity_w          6    -0.281630     0.266473       0.548103       True              4              2
L2_1D_price_velocity_w__x__L2_15m_price_velocity_w          6    -0.321086     0.224607       0.545693       True              4              2
              L3_4h_z_high_w__x__L2_1D_vol_sigma_w          6    -0.394570     0.141920       0.536490       True              4              2
         L2_1D_vol_sigma_w__x__L2_4h_price_accel_w          6    -0.200970     0.330930       0.531900       True              3              3
                 L3_4h_z_low_w__x__L1_1D_bar_range          6    -0.262636     0.260444       0.523080       True              4              2
    L2_4h_price_velocity_w__x__L2_4h_price_accel_w          6    -0.134463     0.362017       0.496480       True              5              1
    L2_1D_price_velocity_w__x__L2_4h_price_accel_w          6    -0.323282     0.165356       0.488638       True              2              4
               L2_1D_vol_sigma_w__x__L3_4h_z_low_w          6    -0.318734     0.157027       0.475760       True              2              4
           L2_4h_price_accel_w__x__L1_1D_bar_range          6    -0.121722     0.346736       0.468458       True              4              2
             L2_1D_vol_sigma_w__x__L1_1D_bar_range          6     0.270918     0.730448       0.459530      False              6              0
                L3_4h_z_se_w__x__L2_1D_vol_sigma_w          6    -0.272244     0.187188       0.459432       True              4              2
             L3_4h_z_low_w__x__L2_4h_price_accel_w          6    -0.022230     0.434153       0.456382       True              5              1
                  L3_4h_z_se_w__x__L1_1D_bar_range          6    -0.206296     0.202036       0.408332       True              4              2

## corr(X,Y) pivot — pair × regime

regime_2d                                            UP_SMOOTH  UP_CHOPPY  DOWN_SMOOTH  DOWN_CHOPPY  FLAT_SMOOTH  FLAT_CHOPPY
pair                                                                                                                         
L1_4h_body__x__L1_1D_bar_range                           0.170      0.122       -0.076       -0.090        0.028       -0.147
L1_4h_body__x__L1_4h_price_velocity_1b                   0.992      1.000        0.945        1.000        0.673        0.976
L1_4h_body__x__L2_15m_price_velocity_w                   0.407      0.336        0.331        0.254        0.322        0.268
L1_4h_body__x__L2_1D_price_velocity_w                   -0.258     -0.347        0.230        0.129        0.015        0.072
L1_4h_body__x__L2_1D_vol_sigma_w                         0.214      0.318       -0.243       -0.114       -0.123       -0.086
L1_4h_body__x__L2_4h_price_accel_w                       0.830      0.594        0.729        0.581        0.434        0.720
L1_4h_body__x__L2_4h_price_velocity_w                    0.214     -0.248        0.367        0.258        0.137        0.165
L1_4h_body__x__L3_4h_z_high_w                            0.321      0.354        0.140        0.044        0.105        0.076
L1_4h_body__x__L3_4h_z_low_w                            -0.036      0.087        0.513        0.397        0.183        0.243
L1_4h_price_velocity_1b__x__L1_1D_bar_range              0.161      0.123       -0.076       -0.090        0.014       -0.146
L1_4h_price_velocity_1b__x__L2_15m_price_velocity_w      0.411      0.336        0.301        0.255        0.223        0.259
L1_4h_price_velocity_1b__x__L2_1D_price_velocity_w      -0.248     -0.348        0.293        0.129        0.063        0.071
L1_4h_price_velocity_1b__x__L2_1D_vol_sigma_w            0.209      0.319       -0.238       -0.115       -0.146       -0.097
L1_4h_price_velocity_1b__x__L2_4h_price_accel_w          0.837      0.592        0.777        0.580        0.593        0.741
L1_4h_price_velocity_1b__x__L2_4h_price_velocity_w       0.220     -0.249        0.437        0.259        0.270        0.174
L1_4h_price_velocity_1b__x__L3_4h_z_high_w               0.330      0.353        0.192        0.043        0.304        0.095
L1_4h_price_velocity_1b__x__L3_4h_z_low_w               -0.025      0.086        0.531        0.397        0.352        0.257
L2_15m_price_velocity_w__x__L1_1D_bar_range              0.185      0.004       -0.110       -0.144        0.011       -0.106
L2_15m_price_velocity_w__x__L2_1D_vol_sigma_w            0.220      0.223       -0.178       -0.143       -0.103       -0.053
L2_15m_price_velocity_w__x__L2_4h_price_accel_w          0.369      0.223        0.251        0.167        0.134        0.194
L2_15m_price_velocity_w__x__L3_4h_z_low_w               -0.149     -0.088        0.179        0.040        0.088       -0.060
L2_1D_price_velocity_w__x__L1_1D_bar_range              -0.251     -0.610       -0.411       -0.229        0.129       -0.273
L2_1D_price_velocity_w__x__L2_15m_price_velocity_w      -0.321     -0.285        0.225        0.066        0.058        0.069
L2_1D_price_velocity_w__x__L2_1D_vol_sigma_w            -0.455     -0.812       -0.629        0.013        0.180       -0.463
L2_1D_price_velocity_w__x__L2_4h_price_accel_w          -0.289     -0.323        0.153        0.165       -0.109       -0.110
L2_1D_price_velocity_w__x__L2_4h_price_velocity_w        0.614      0.900        0.889        0.635        0.696        0.768
L2_1D_price_velocity_w__x__L3_4h_z_high_w               -0.152     -0.136        0.143       -0.039       -0.102       -0.142
L2_1D_price_velocity_w__x__L3_4h_z_low_w                -0.074     -0.094        0.168       -0.157       -0.197       -0.171
L2_1D_vol_sigma_w__x__L1_1D_bar_range                    0.670      0.730        0.271        0.654        0.320        0.650
L2_1D_vol_sigma_w__x__L2_4h_price_accel_w                0.331      0.259       -0.176        0.066       -0.201       -0.035
L2_1D_vol_sigma_w__x__L3_4h_z_low_w                     -0.002      0.142       -0.109       -0.319       -0.003        0.157
L2_1h_price_velocity_w__x__L1_1D_bar_range               0.662      0.199       -0.477       -0.356        0.081       -0.218
L2_1h_price_velocity_w__x__L1_4h_body                    0.469      0.517        0.532        0.439        0.344        0.449
L2_1h_price_velocity_w__x__L1_4h_price_velocity_1b       0.469      0.516        0.585        0.440        0.610        0.474
L2_1h_price_velocity_w__x__L2_15m_price_velocity_w       0.360      0.271        0.460        0.363        0.409        0.346
L2_1h_price_velocity_w__x__L2_1D_price_velocity_w       -0.093     -0.176        0.567        0.339        0.182        0.159
L2_1h_price_velocity_w__x__L2_1D_vol_sigma_w             0.437      0.238       -0.428       -0.351       -0.065       -0.161
L2_1h_price_velocity_w__x__L2_4h_price_accel_w           0.540      0.346        0.501        0.277        0.324        0.369
L2_1h_price_velocity_w__x__L2_4h_price_velocity_w        0.524     -0.004        0.730        0.455        0.416        0.278
L2_1h_price_velocity_w__x__L3_4h_z_high_w                0.400      0.596        0.530        0.489        0.557        0.424
L2_1h_price_velocity_w__x__L3_4h_z_low_w                 0.323      0.507        0.643        0.531        0.544        0.441
L2_1h_price_velocity_w__x__L3_4h_z_se_w                  0.404      0.632        0.623        0.575        0.590        0.533
L2_4h_price_accel_w__x__L1_1D_bar_range                  0.347      0.059       -0.053        0.062        0.019       -0.122
L2_4h_price_velocity_w__x__L1_1D_bar_range               0.275     -0.539       -0.414       -0.373        0.241       -0.035
L2_4h_price_velocity_w__x__L2_15m_price_velocity_w      -0.056     -0.282        0.266        0.114        0.083        0.026
L2_4h_price_velocity_w__x__L2_1D_vol_sigma_w             0.061     -0.725       -0.559       -0.192        0.123       -0.278
L2_4h_price_velocity_w__x__L2_4h_price_accel_w           0.289     -0.134        0.362        0.114        0.153        0.082
L2_4h_price_velocity_w__x__L3_4h_z_low_w                -0.029      0.067        0.322       -0.017       -0.022        0.020
L3_4h_z_high_w__x__L1_1D_bar_range                       0.119      0.093       -0.325       -0.316        0.241        0.114
L3_4h_z_high_w__x__L2_15m_price_velocity_w               0.034      0.056       -0.008       -0.078        0.097       -0.071
L3_4h_z_high_w__x__L2_1D_vol_sigma_w                     0.045      0.142       -0.054       -0.395        0.084        0.141
L3_4h_z_high_w__x__L2_4h_price_accel_w                   0.248      0.178        0.167       -0.029        0.050        0.040
L3_4h_z_high_w__x__L2_4h_price_velocity_w               -0.010      0.026        0.259        0.014        0.046        0.014
L3_4h_z_high_w__x__L3_4h_z_low_w                         0.739      0.802        0.756        0.753        0.851        0.678
L3_4h_z_low_w__x__L1_1D_bar_range                        0.159      0.260       -0.263       -0.095        0.206        0.163
L3_4h_z_low_w__x__L2_4h_price_accel_w                   -0.022      0.038        0.434        0.280        0.149        0.148
L3_4h_z_se_w__x__L1_1D_bar_range                         0.105      0.202       -0.206       -0.119        0.176        0.122
L3_4h_z_se_w__x__L1_4h_body                              0.379      0.451        0.589        0.533        0.415        0.490
L3_4h_z_se_w__x__L1_4h_price_velocity_1b                 0.387      0.450        0.587        0.533        0.509        0.498
L3_4h_z_se_w__x__L2_15m_price_velocity_w                 0.035      0.092        0.191        0.078        0.178        0.069
L3_4h_z_se_w__x__L2_1D_price_velocity_w                 -0.171     -0.184        0.091       -0.118       -0.168       -0.172
L3_4h_z_se_w__x__L2_1D_vol_sigma_w                       0.031      0.187       -0.057       -0.272        0.032        0.118
L3_4h_z_se_w__x__L2_4h_price_accel_w                     0.297      0.265        0.488        0.361        0.216        0.325
L3_4h_z_se_w__x__L2_4h_price_velocity_w                 -0.009      0.012        0.268        0.039        0.020        0.024
L3_4h_z_se_w__x__L3_4h_z_high_w                          0.962      0.946        0.778        0.758        0.872        0.768
L3_4h_z_se_w__x__L3_4h_z_low_w                           0.748      0.833        0.943        0.957        0.903        0.853

## interaction_rms pivot — pair × regime

regime_2d                                            UP_SMOOTH  UP_CHOPPY  DOWN_SMOOTH  DOWN_CHOPPY  FLAT_SMOOTH  FLAT_CHOPPY
pair                                                                                                                         
L1_4h_body__x__L1_1D_bar_range                          11.043      6.277        6.001        7.149        2.807        4.896
L1_4h_body__x__L1_4h_price_velocity_1b                  10.337     10.113        7.926        3.642        4.602       10.596
L1_4h_body__x__L2_15m_price_velocity_w                   5.739     11.940        6.728        9.270        1.724        3.644
L1_4h_body__x__L2_1D_price_velocity_w                    8.237      7.457        6.109        6.623        1.472        4.390
L1_4h_body__x__L2_1D_vol_sigma_w                         7.587      5.036        6.984        9.440        0.867        3.542
L1_4h_body__x__L2_4h_price_accel_w                       7.838      8.487       10.857        5.506        2.742        9.521
L1_4h_body__x__L2_4h_price_velocity_w                    3.201      5.413        5.819        7.635        1.492        2.948
L1_4h_body__x__L3_4h_z_high_w                           13.033      4.989        6.764        8.448        2.109        2.168
L1_4h_body__x__L3_4h_z_low_w                             4.328      2.407        9.776        6.170        2.626        3.573
L1_4h_price_velocity_1b__x__L1_1D_bar_range             11.348      6.566        5.956        7.149        2.863        4.841
L1_4h_price_velocity_1b__x__L2_15m_price_velocity_w      5.654     12.132        6.016        9.270        2.377        3.660
L1_4h_price_velocity_1b__x__L2_1D_price_velocity_w       8.200      6.543        6.672        6.623        1.203        4.272
L1_4h_price_velocity_1b__x__L2_1D_vol_sigma_w            7.313      4.335        6.997        9.440        1.518        3.697
L1_4h_price_velocity_1b__x__L2_4h_price_accel_w          8.273      8.774       10.927        5.506        3.130        8.901
L1_4h_price_velocity_1b__x__L2_4h_price_velocity_w       3.519      5.532        6.030        7.635        1.412        2.927
L1_4h_price_velocity_1b__x__L3_4h_z_high_w              13.761      5.484        7.000        8.448        2.327        2.370
L1_4h_price_velocity_1b__x__L3_4h_z_low_w                5.431      1.998        9.779        6.170        3.004        3.716
L2_15m_price_velocity_w__x__L1_1D_bar_range              3.913      2.906        2.118       13.678        3.476        1.083
L2_15m_price_velocity_w__x__L2_1D_vol_sigma_w            3.611      7.352        5.217        8.322        0.856        1.038
L2_15m_price_velocity_w__x__L2_4h_price_accel_w          5.728      8.670        5.555        4.029        2.864        2.537
L2_15m_price_velocity_w__x__L3_4h_z_low_w                6.773      5.249        3.170        4.281        1.972        4.178
L2_1D_price_velocity_w__x__L1_1D_bar_range               6.411      8.119        4.775        6.595        3.556        2.444
L2_1D_price_velocity_w__x__L2_15m_price_velocity_w       3.695      9.222        4.217        7.629        2.902        0.873
L2_1D_price_velocity_w__x__L2_1D_vol_sigma_w             7.100     15.981       11.938        4.007        4.064        2.053
L2_1D_price_velocity_w__x__L2_4h_price_accel_w           5.083      4.397        5.922        5.268        0.726        2.529
L2_1D_price_velocity_w__x__L2_4h_price_velocity_w        8.160      9.537        6.948        6.763        4.249        3.450
L2_1D_price_velocity_w__x__L3_4h_z_high_w                6.851      4.346        3.178        9.427        1.817        3.741
L2_1D_price_velocity_w__x__L3_4h_z_low_w                 8.097      4.647        6.622        6.978        4.796        4.583
L2_1D_vol_sigma_w__x__L1_1D_bar_range                    6.596      7.529        8.261        4.362        2.182        4.825
L2_1D_vol_sigma_w__x__L2_4h_price_accel_w                5.388      5.232        5.538        4.475        3.335        2.289
L2_1D_vol_sigma_w__x__L3_4h_z_low_w                     12.219      3.753        4.956        8.438        1.431        4.563
L2_1h_price_velocity_w__x__L1_1D_bar_range               9.441      4.525        3.352        7.586        4.052        2.769
L2_1h_price_velocity_w__x__L1_4h_body                    6.625      5.072        4.687       10.693        4.773        4.899
L2_1h_price_velocity_w__x__L1_4h_price_velocity_1b       6.577      4.903        4.583       10.693        5.056        5.013
L2_1h_price_velocity_w__x__L2_15m_price_velocity_w       4.130      3.204        4.540        4.483        4.084        3.671
L2_1h_price_velocity_w__x__L2_1D_price_velocity_w        7.362      8.170        6.946        7.053        2.917        4.561
L2_1h_price_velocity_w__x__L2_1D_vol_sigma_w             3.958      6.257        7.959        9.455        2.895        2.347
L2_1h_price_velocity_w__x__L2_4h_price_accel_w           6.308      4.808        9.546        5.240        2.519        4.251
L2_1h_price_velocity_w__x__L2_4h_price_velocity_w        5.667      4.552        8.069        7.424        4.255        3.512
L2_1h_price_velocity_w__x__L3_4h_z_high_w               38.811      5.082        8.098        6.922        1.759        2.417
L2_1h_price_velocity_w__x__L3_4h_z_low_w                17.017      6.387        8.531        8.576        3.030        6.290
L2_1h_price_velocity_w__x__L3_4h_z_se_w                 44.432      3.526        9.766       10.384        2.830        6.625
L2_4h_price_accel_w__x__L1_1D_bar_range                  8.317      5.175        7.248        5.079        3.053        2.752
L2_4h_price_velocity_w__x__L1_1D_bar_range               8.100      4.815        3.953        5.588        3.379        2.146
L2_4h_price_velocity_w__x__L2_15m_price_velocity_w       5.776      5.515        5.726        9.799        2.652        1.349
L2_4h_price_velocity_w__x__L2_1D_vol_sigma_w             7.406      7.852       11.844        5.050        2.198        2.783
L2_4h_price_velocity_w__x__L2_4h_price_accel_w           4.437      5.388       10.572        6.260        2.407        2.326
L2_4h_price_velocity_w__x__L3_4h_z_low_w                 8.120      8.753        7.196        6.285        2.555        2.941
L3_4h_z_high_w__x__L1_1D_bar_range                      12.589      2.190        9.437        5.491        2.778        2.846
L3_4h_z_high_w__x__L2_15m_price_velocity_w               8.173      6.352        2.186        7.414        3.215        3.412
L3_4h_z_high_w__x__L2_1D_vol_sigma_w                    12.791      5.631        3.428        5.294        1.397        3.839
L3_4h_z_high_w__x__L2_4h_price_accel_w                  11.394      3.685        3.812       13.134        1.459        4.319
L3_4h_z_high_w__x__L2_4h_price_velocity_w                4.693      5.527        4.587        9.520        0.958        2.068
L3_4h_z_high_w__x__L3_4h_z_low_w                         8.199      9.419       12.710       26.339        4.889        7.568
L3_4h_z_low_w__x__L1_1D_bar_range                       12.382      4.982        7.853        5.466        4.813        4.919
L3_4h_z_low_w__x__L2_4h_price_accel_w                    6.013      4.358        7.410        5.778        2.842        4.193
L3_4h_z_se_w__x__L1_1D_bar_range                        13.452      3.945        6.412        5.381        4.557        5.479
L3_4h_z_se_w__x__L1_4h_body                             25.994      7.651       18.128        6.222        2.943        4.608
L3_4h_z_se_w__x__L1_4h_price_velocity_1b                20.901      8.354       18.152        6.222        3.772        4.585
L3_4h_z_se_w__x__L2_15m_price_velocity_w                 9.145      4.342        4.509        5.457        2.683        4.279
L3_4h_z_se_w__x__L2_1D_price_velocity_w                  9.715      4.115        8.947        8.265        3.562        3.085
L3_4h_z_se_w__x__L2_1D_vol_sigma_w                      13.579      4.860        7.120        5.894        1.100        4.928
L3_4h_z_se_w__x__L2_4h_price_accel_w                    15.965      4.266        7.735        9.612        2.638        4.840
L3_4h_z_se_w__x__L2_4h_price_velocity_w                  6.554      5.609       11.041        6.589        1.438        1.887
L3_4h_z_se_w__x__L3_4h_z_high_w                          7.004      4.129       13.444       21.737        6.010        7.221
L3_4h_z_se_w__x__L3_4h_z_low_w                           9.031      9.016        6.341       11.474        4.660        9.069
