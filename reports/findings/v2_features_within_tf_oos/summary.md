# Within-TF feature×feature OOS validation (Layer D9) - 2026-05-03 20:13 UTC

## D7 confirmed regime-flip pairs OOS

- IS confirmed: 83
- Evaluated on OOS: 83
- **Survive OOS**: 79 (95.2%)

 tf                c1               c2  regime_min  regime_max  is_r_min  is_r_max  oos_r_min  oos_r_max  n_oos_min  n_oos_max  sign_min_ok  sign_max_ok  regime_diff_holds  survives  min_is_mag
 1h  price_velocity_w    price_sigma_w DOWN_SMOOTH   UP_SMOOTH -0.837553  0.887296  -0.823328   0.517805       1782       2592         True         True               True      True    0.837553
15m  price_velocity_w    price_sigma_w DOWN_SMOOTH   UP_SMOOTH -0.807789  0.846109  -0.784334   0.764453       1782       2592         True         True               True      True    0.807789
 1h  price_velocity_w    swing_noise_w DOWN_SMOOTH   UP_SMOOTH -0.751025  0.766465  -0.564159   0.331492       1782       2592         True         True               True      True    0.751025
15m  price_velocity_w         SE_low_w DOWN_SMOOTH   UP_SMOOTH -0.734075  0.760505  -0.654305   0.576676       1782       2592         True         True               True      True    0.734075
15m  price_velocity_w        SE_high_w DOWN_SMOOTH   UP_SMOOTH -0.698828  0.787214  -0.606862   0.606814       1782       2592         True         True               True      True    0.698828
 5m  price_velocity_w    price_sigma_w DOWN_SMOOTH   UP_SMOOTH -0.624167  0.703391  -0.623973   0.605811       1782       2592         True         True               True      True    0.624167
 1h price_velocity_1b        bar_range DOWN_SMOOTH   UP_SMOOTH -0.599734  0.696008  -0.588547   0.449205       1782       2592         True         True               True      True    0.599734
 1h         bar_range             body DOWN_SMOOTH   UP_SMOOTH -0.590246  0.712466  -0.577500   0.517898       1782       2592         True         True               True      True    0.590246
15m  price_velocity_w    swing_noise_w DOWN_SMOOTH   UP_SMOOTH -0.564644  0.615484  -0.375587   0.477235       1782       2592         True         True               True      True    0.564644
 1h  price_velocity_w       vol_mean_w DOWN_SMOOTH   UP_SMOOTH -0.682212  0.548084  -0.498215   0.161967       1782       2592         True         True               True      True    0.548084
 5m  price_velocity_w         SE_low_w DOWN_SMOOTH   UP_SMOOTH -0.531145  0.566646  -0.443497   0.466684       1782       2592         True         True               True      True    0.531145
 1m  price_velocity_w    price_sigma_w DOWN_SMOOTH   UP_SMOOTH -0.498496  0.534832  -0.402804   0.458103       1782       2592         True         True               True      True    0.498496
15m  price_velocity_w       vol_mean_w DOWN_SMOOTH   UP_SMOOTH -0.527726  0.485101  -0.367827   0.315041       1782       2592         True         True               True      True    0.485101
 5m  price_velocity_w        SE_high_w DOWN_SMOOTH   UP_SMOOTH -0.482172  0.629295  -0.433697   0.505970       1782       2592         True         True               True      True    0.482172
15m         bar_range price_velocity_w DOWN_SMOOTH   UP_SMOOTH -0.551930  0.462450  -0.426874   0.203481       1782       2592         True         True               True      True    0.462450
15m  price_velocity_w      vol_sigma_w DOWN_SMOOTH   UP_SMOOTH -0.475026  0.442120  -0.495313   0.291728       1782       2592         True         True               True      True    0.442120
 1h  price_velocity_w      vol_sigma_w DOWN_CHOPPY   UP_CHOPPY -0.527317  0.441616  -0.454623   0.467451       1332       1104         True         True               True      True    0.441616
 1m  price_velocity_w         SE_low_w DOWN_SMOOTH   UP_SMOOTH -0.441609  0.432006  -0.327732   0.324367       1782       2592         True         True               True      True    0.432006
 1h price_velocity_1b reversion_prob_w   UP_SMOOTH DOWN_SMOOTH -0.421956  0.531059  -0.510932   0.634090       2592       1782         True         True               True      True    0.421956
 1m  price_velocity_w        SE_high_w DOWN_SMOOTH   UP_SMOOTH -0.419550  0.449870  -0.298643   0.356515       1782       2592         True         True               True      True    0.419550
15m price_velocity_1b        bar_range DOWN_SMOOTH   UP_SMOOTH -0.416720  0.518903  -0.433436   0.354741       1782       2592         True         True               True      True    0.416720
 1h         bar_range    price_accel_w DOWN_SMOOTH   UP_SMOOTH -0.410146  0.582952  -0.500818   0.318340       1782       2592         True         True               True      True    0.410146
 1h              body reversion_prob_w   UP_SMOOTH DOWN_SMOOTH -0.401644  0.534669  -0.361175   0.635281       2592       1782         True         True               True      True    0.401644
 5m         bar_range price_velocity_w DOWN_SMOOTH   UP_SMOOTH -0.397068  0.447926  -0.374635   0.196009       1782       2592         True         True               True      True    0.397068
 1m  price_velocity_w    swing_noise_w DOWN_SMOOTH   UP_SMOOTH -0.376883  0.454814  -0.366465   0.377151       1782       2592         True         True               True      True    0.376883
15m         bar_range             body DOWN_SMOOTH   UP_SMOOTH -0.374845  0.530222  -0.405342   0.394113       1782       2592         True         True               True      True    0.374845
 5m  price_velocity_w    swing_noise_w DOWN_SMOOTH   UP_SMOOTH -0.371849  0.508506  -0.378236   0.380697       1782       2592         True         True               True      True    0.371849
 1h    vol_velocity_w          z_low_w DOWN_SMOOTH   UP_SMOOTH -0.357565  0.358129  -0.210553  -0.084690       1782       2592         True        False              False     False    0.357565
 1h     price_accel_w reversion_prob_w   UP_SMOOTH DOWN_SMOOTH -0.352744  0.430004  -0.311039   0.557190       2592       1782         True         True               True      True    0.352744
 1h    vol_velocity_w           z_se_w DOWN_SMOOTH   UP_SMOOTH -0.383077  0.347017  -0.217331   0.016430       1782       2592         True        False               True     False    0.347017
 1h            z_se_w reversion_prob_w   UP_SMOOTH DOWN_SMOOTH -0.343267  0.474929  -0.375412   0.529222       2592       1782         True         True               True      True    0.343267
 1h    vol_velocity_w         z_high_w DOWN_SMOOTH   UP_SMOOTH -0.403196  0.338354  -0.249728   0.017031       1782       2592         True        False               True     False    0.338354
 5m price_velocity_1b        bar_range DOWN_SMOOTH   UP_SMOOTH -0.320301  0.333793  -0.305633   0.237714       1782       2592         True         True               True      True    0.320301
 5m  price_velocity_w       vol_mean_w DOWN_SMOOTH   UP_SMOOTH -0.341435  0.317185  -0.293764   0.227173       1782       2592         True         True               True      True    0.317185
 5m  price_velocity_w      vol_sigma_w DOWN_SMOOTH   UP_SMOOTH -0.379281  0.313256  -0.320423   0.215958       1782       2592         True         True               True      True    0.313256
 1m         bar_range price_velocity_w DOWN_SMOOTH   UP_SMOOTH -0.298093  0.306731  -0.250595   0.156923       1782       2592         True         True               True      True    0.298093
 1h         bar_range           z_se_w DOWN_SMOOTH   UP_SMOOTH -0.382062  0.282932  -0.348179   0.218474       1782       2592         True         True               True      True    0.282932
 1h    price_accel_1b  vol_velocity_1b DOWN_SMOOTH   UP_CHOPPY -0.272237  0.268770  -0.247569   0.170089       1782       1104         True         True               True      True    0.268770
 5m         bar_range             body DOWN_SMOOTH   UP_SMOOTH -0.266117  0.335002  -0.266480   0.247081       1782       2592         True         True               True      True    0.266117
15m price_velocity_1b reversion_prob_w   UP_SMOOTH DOWN_SMOOTH -0.258547  0.301988  -0.343116   0.352935       2592       1782         True         True               True      True    0.258547
 1h    price_accel_1b reversion_prob_w   UP_SMOOTH DOWN_SMOOTH -0.254767  0.327531  -0.265589   0.429853       2592       1782         True         True               True      True    0.254767
15m         bar_range    price_accel_w DOWN_SMOOTH   UP_SMOOTH -0.243685  0.339230  -0.259600   0.225467       1782       2592         True         True               True      True    0.243685
 1h              body   vol_velocity_w DOWN_SMOOTH   UP_SMOOTH -0.348035  0.238882  -0.261464   0.189381       1782       2592         True         True               True      True    0.238882
 1h price_velocity_1b   vol_velocity_w DOWN_SMOOTH   UP_SMOOTH -0.319056  0.238370  -0.254320   0.177280       1782       2592         True         True               True      True    0.238370
 1m  price_velocity_w      vol_sigma_w DOWN_SMOOTH   UP_SMOOTH -0.278574  0.237099  -0.193010   0.195468       1782       2592         True         True               True      True    0.237099
 1h         bar_range         z_high_w DOWN_SMOOTH   UP_SMOOTH -0.236237  0.318589  -0.214828   0.300895       1782       2592         True         True               True      True    0.236237
 1h     price_accel_w   vol_velocity_w DOWN_SMOOTH   UP_SMOOTH -0.334655  0.224638  -0.325433   0.157635       1782       2592         True         True               True      True    0.224638
 1m  price_velocity_w       vol_mean_w DOWN_SMOOTH   UP_SMOOTH -0.241749  0.222528  -0.211796   0.185505       1782       2592         True         True               True      True    0.222528
15m              body reversion_prob_w   UP_SMOOTH DOWN_SMOOTH -0.220650  0.274404  -0.260474   0.303332       2592       1782         True         True               True      True    0.220650
15m     price_accel_w   vol_velocity_w DOWN_SMOOTH   UP_SMOOTH -0.234658  0.217105  -0.235172   0.227571       1782       2592         True         True               True      True    0.217105

## D8 top-100 contextualizers OOS

- Evaluated: 100
- **Survive OOS** (lift >= 0.5, sign match): 100 (100.0%)

 tf                 X                Y                 Z  is_r_min  is_r_max  is_lift  oos_r_min  oos_r_max  oos_lift  sign_match  oos_sign_flip  survives
 5m         bar_range             body            z_se_w -0.899195  0.894703 1.793898  -0.864121   0.858715  1.722836        True           True      True
15m         bar_range             body            z_se_w -0.874810  0.883985 1.758795  -0.837023   0.814943  1.651966        True           True      True
15m         bar_range             body price_velocity_1b -0.866816  0.881205 1.748021  -0.811527   0.809482  1.621009        True           True      True
 1m         bar_range             body price_velocity_1b -0.867023  0.870521 1.737544  -0.811914   0.842258  1.654172        True           True      True
 1m price_velocity_1b        bar_range              body -0.864683  0.866004 1.730686  -0.795845   0.834488  1.630333        True           True      True
 1h         bar_range             body            z_se_w -0.861118  0.862405 1.723523  -0.836201   0.815975  1.652175        True           True      True
 5m            z_se_w reversion_prob_w price_velocity_1b -0.858507  0.864553 1.723059  -0.863743   0.858960  1.722703        True           True      True
 5s            z_se_w reversion_prob_w price_velocity_1b -0.855838  0.865954 1.721792  -0.855659   0.866461  1.722119        True           True      True
 5s            z_se_w reversion_prob_w          z_high_w -0.800405  0.914104 1.714509  -0.797590   0.898580  1.696170        True           True      True
15m price_velocity_1b        bar_range              body -0.836260  0.872291 1.708551  -0.825449   0.718594  1.544043        True           True      True
 5m         bar_range             body price_velocity_1b -0.861629  0.843173 1.704802  -0.812175   0.828948  1.641123        True           True      True
 1h         bar_range             body price_velocity_1b -0.874191  0.827644 1.701835  -0.805085   0.826617  1.631703        True           True      True
 5m            z_se_w reversion_prob_w              body -0.848029  0.853570 1.701599  -0.846200   0.857590  1.703790        True           True      True
 5s            z_se_w reversion_prob_w           z_low_w -0.919324  0.777838 1.697162  -0.924163   0.756619  1.680782        True           True      True
 5s         bar_range             body            z_se_w -0.850738  0.836382 1.687120  -0.842174   0.852043  1.694217        True           True      True
 5m price_velocity_1b        bar_range            z_se_w -0.886041  0.796268 1.682308  -0.853443   0.851549  1.704993        True           True      True
 1m         bar_range             body            z_se_w -0.826988  0.845157 1.672145  -0.768285   0.742061  1.510347        True           True      True
 1m price_velocity_1b        bar_range            z_se_w -0.827039  0.842260 1.669299  -0.767039   0.736653  1.503692        True           True      True
 1h         bar_range             body     price_accel_w -0.825687  0.840567 1.666254  -0.815262   0.791503  1.606765        True           True      True
 1h price_velocity_1b        bar_range              body -0.830762  0.829434 1.660196  -0.825894   0.707650  1.533544        True           True      True
 5s         bar_range             body price_velocity_1b -0.835027  0.811903 1.646930  -0.833552   0.852925  1.686477        True           True      True
15m price_velocity_1b        bar_range            z_se_w -0.864600  0.768197 1.632797  -0.852621   0.773694  1.626315        True           True      True
 5m            z_se_w reversion_prob_w          z_high_w -0.709930  0.904890 1.614820  -0.727346   0.889620  1.616967        True           True      True
 5m            z_se_w reversion_prob_w           z_low_w -0.905103  0.706634 1.611737  -0.911036   0.715468  1.626504        True           True      True
 5s            z_se_w reversion_prob_w              body -0.789603  0.818023 1.607626  -0.799636   0.811501  1.611137        True           True      True
 1h price_velocity_1b        bar_range            z_se_w -0.844617  0.754437 1.599054  -0.852093   0.761834  1.613927        True           True      True
 5m price_velocity_1b        bar_range              body -0.824407  0.738074 1.562481  -0.802054   0.776924  1.578979        True           True      True
 5s price_velocity_1b        bar_range            z_se_w -0.796623  0.758269 1.554892  -0.737994   0.734308  1.472302        True           True      True
 1h price_velocity_1b        bar_range     price_accel_w -0.796736  0.747831 1.544567  -0.829585   0.746202  1.575787        True           True      True
 1h            z_se_w reversion_prob_w              body -0.762684  0.781831 1.544515  -0.732171   0.746731  1.478902        True           True      True
 1h            z_se_w reversion_prob_w price_velocity_1b -0.759063  0.784499 1.543561  -0.777820   0.746198  1.524019        True           True      True
15m            z_se_w reversion_prob_w price_velocity_1b -0.770474  0.772836 1.543310  -0.773385   0.759126  1.532510        True           True      True
15m            z_se_w reversion_prob_w              body -0.770082  0.764910 1.534992  -0.750889   0.756326  1.507214        True           True      True
15m         bar_range             body     price_accel_w -0.776009  0.758231 1.534240  -0.772356   0.727070  1.499426        True           True      True
15m            z_se_w reversion_prob_w           z_low_w -0.856336  0.668344 1.524680  -0.854652   0.678142  1.532794        True           True      True
 1m            z_se_w reversion_prob_w           z_low_w -0.812366  0.707384 1.519750  -0.820104   0.705136  1.525240        True           True      True
 1m            z_se_w reversion_prob_w          z_high_w -0.724201  0.790676 1.514877  -0.729017   0.825766  1.554784        True           True      True
 1h         bar_range             body    price_accel_1b -0.790680  0.721878 1.512558  -0.714977   0.531657  1.246633        True           True      True
 5s price_velocity_1b        bar_range              body -0.773530  0.721556 1.495086  -0.759557   0.771133  1.530690        True           True      True
 1h            z_se_w reversion_prob_w           z_low_w -0.851659  0.641039 1.492697  -0.838817   0.606027  1.444844        True           True      True
15m            z_se_w reversion_prob_w          z_high_w -0.646612  0.842999 1.489611  -0.738077   0.855703  1.593780        True           True      True
 1h   vol_velocity_1b   vol_velocity_w       vol_accel_w -0.673971  0.793503 1.467474  -0.690303   0.787236  1.477539        True           True      True
 1m         bar_range             body     price_accel_w -0.739343  0.715680 1.455024  -0.650646   0.679952  1.330598        True           True      True
 1m price_velocity_1b        bar_range     price_accel_w -0.738757  0.713662 1.452420  -0.644963   0.674499  1.319462        True           True      True
 5m         bar_range             body     price_accel_w -0.736653  0.708625 1.445278  -0.671817   0.677050  1.348867        True           True      True
 5s   vol_velocity_1b      vol_sigma_w      vol_accel_1b -0.523478  0.921385 1.444863  -0.500885   0.731329  1.232214        True           True      True
15m price_velocity_1b        bar_range     price_accel_w -0.769677  0.674558 1.444235  -0.790796   0.697206  1.488002        True           True      True
 1h price_velocity_1b        bar_range    price_accel_1b -0.780752  0.648567 1.429319  -0.734614   0.494025  1.228639        True           True      True
 5s   vol_velocity_1b      vol_sigma_w       vol_accel_w -0.466200  0.942371 1.408571  -0.497557   0.788201  1.285757        True           True      True
15m         bar_range             body    price_accel_1b -0.708968  0.680862 1.389829  -0.675541   0.626957  1.302498        True           True      True
