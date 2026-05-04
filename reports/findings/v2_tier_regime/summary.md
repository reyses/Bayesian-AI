# 9-tier regime analysis - 2026-05-04 01:03 UTC

## Per-tier baseline + per-regime $/day

regime_2d       UP_SMOOTH  UP_CHOPPY  DOWN_SMOOTH  DOWN_CHOPPY  FLAT_SMOOTH  FLAT_CHOPPY
tier                                                                                    
CASCADE             -2.36      -9.45       -11.13        -5.17       -15.47        24.60
KILL_SHOT           -6.41     -48.20        14.34       -10.50        13.54        -0.30
FREIGHT_TRAIN     -690.50    1104.50       -78.50         0.00         0.00        81.92
FADE_AGAINST        74.25     -21.57        63.69        22.08       -53.39       -16.54
RIDE_AGAINST       119.89     -20.26       135.64       -45.63       -86.43       -79.62
FADE_CALM           84.32     120.57        83.34        70.24       -75.46      -101.92
MTF_BREAKOUT         4.30     -25.30        50.65       -38.87       -19.03        18.91
MTF_EXHAUSTION      -0.37      13.09        13.53         4.23         0.86        12.09
BASE_NMP             8.58     128.00       -65.45      -132.79        49.92        29.49

## Best regime gate per tier

          tier                                                    gate  delta_per_day   kept_pct  delta_ci_lo  delta_ci_hi
     FADE_CALM KEEP_TOP_4: UP_SMOOTH+DOWN_SMOOTH+UP_CHOPPY+DOWN_CHOPPY      52.905797  41.326808     9.587754    96.192826
  RIDE_AGAINST                       KEEP_TOP_2: UP_SMOOTH+DOWN_SMOOTH      51.078171  26.419922    14.684845    90.479425
      BASE_NMP                                     NO_SIGNIFICANT_GATE       0.000000 100.000000          NaN          NaN
       CASCADE                                     NO_SIGNIFICANT_GATE       0.000000 100.000000          NaN          NaN
  FADE_AGAINST                                     NO_SIGNIFICANT_GATE       0.000000 100.000000          NaN          NaN
     KILL_SHOT                                     NO_SIGNIFICANT_GATE       0.000000 100.000000          NaN          NaN
  MTF_BREAKOUT                                     NO_SIGNIFICANT_GATE       0.000000 100.000000          NaN          NaN
MTF_EXHAUSTION                                     NO_SIGNIFICANT_GATE       0.000000 100.000000          NaN          NaN

## All significant gate recommendations

        tier                                                    gate  kept_n_trades  kept_pct_trades  kept_mean_per_day  baseline_mean_per_day  delta_per_day  delta_ci_lo  delta_ci_hi  delta_significant
   FADE_CALM KEEP_TOP_4: UP_SMOOTH+DOWN_SMOOTH+UP_CHOPPY+DOWN_CHOPPY           9880        41.326808          37.689855             -15.215942      52.905797     9.587754    96.192826               True
RIDE_AGAINST                       KEEP_TOP_2: UP_SMOOTH+DOWN_SMOOTH          10429        26.419922          37.884956             -13.193215      51.078171    14.684845    90.479425               True
RIDE_AGAINST             KEEP_TOP_3: UP_SMOOTH+DOWN_SMOOTH+UP_CHOPPY          13659        34.602523          36.510324             -13.193215      49.703540    13.521534    88.146755               True
   FADE_CALM             KEEP_TOP_3: UP_SMOOTH+DOWN_SMOOTH+UP_CHOPPY           8267        34.579830          33.821739             -15.215942      49.037681     3.790399    95.110435               True
RIDE_AGAINST KEEP_TOP_4: UP_SMOOTH+DOWN_SMOOTH+UP_CHOPPY+DOWN_CHOPPY          16323        41.351269          33.952802             -13.193215      47.146018    12.433186    82.845796               True
