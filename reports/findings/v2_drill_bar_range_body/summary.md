# Drill: bar_range x body joint-extreme regime detector - 2026-05-03 22:43 UTC

**Trigger**: bar_range Q4 + sign(body)

**Daily prediction rule**: |n_bigpos - n_bigneg| >= 3 margin = UP or DOWN; else ABSTAIN

## Cell -> regime distribution

### TF=5s

regime_2d  UP_SMOOTH  UP_CHOPPY  DOWN_SMOOTH  DOWN_CHOPPY  FLAT_SMOOTH  FLAT_CHOPPY
cell                                                                               
BIG+NEG         12.6        6.8         13.9          8.2         11.4         47.1
BIG+POS         13.2        7.3         13.5          7.7         11.4         46.9
OTHER           17.6        8.1          9.7          5.7         19.3         39.6

### TF=1m

regime_2d  UP_SMOOTH  UP_CHOPPY  DOWN_SMOOTH  DOWN_CHOPPY  FLAT_SMOOTH  FLAT_CHOPPY
cell                                                                               
BIG+NEG         11.1        6.8         15.7          8.8         10.4         47.1
BIG+POS         13.4        7.2         12.8          8.9         10.0         47.8
OTHER           17.7        8.1          9.6          5.6         19.5         39.6

### TF=5m

regime_2d  UP_SMOOTH  UP_CHOPPY  DOWN_SMOOTH  DOWN_CHOPPY  FLAT_SMOOTH  FLAT_CHOPPY
cell                                                                               
BIG+NEG         10.1        6.5         16.5          9.3         10.5         47.1
BIG+POS         13.7        7.6         12.3          8.6          9.8         48.1
OTHER           17.8        8.0          9.5          5.5         19.5         39.6

### TF=15m

regime_2d  UP_SMOOTH  UP_CHOPPY  DOWN_SMOOTH  DOWN_CHOPPY  FLAT_SMOOTH  FLAT_CHOPPY
cell                                                                               
BIG+NEG          8.7        5.3         17.7         10.3         10.0         48.0
BIG+POS         15.5        7.6         10.9          8.0         10.2         47.9
OTHER           17.7        8.2          9.6          5.5         19.5         39.5

### TF=1h

regime_2d  UP_SMOOTH  UP_CHOPPY  DOWN_SMOOTH  DOWN_CHOPPY  FLAT_SMOOTH  FLAT_CHOPPY
cell                                                                               
BIG+NEG          6.3        4.7         19.6         10.5          9.5         49.6
BIG+POS         19.5       10.6          9.1          5.2          8.0         47.7
OTHER           17.6        7.9          9.5          5.8         19.8         39.4

## Daily classifier performance

 tf  n_days  n_up_pred  up_precision  up_lift_vs_baserate  n_down_pred  down_precision  down_lift_vs_baserate  coverage  abstain_pct
 5s     348        110      0.272727             0.025601          120        0.208333               0.027299  0.660920     0.339080
 1m     348         88      0.340909             0.093783          118        0.322034               0.140999  0.591954     0.408046
 5m     348        102      0.470588             0.223462          123        0.357724               0.176689  0.646552     0.353448
15m     348        136      0.485294             0.238168          150        0.366667               0.185632  0.821839     0.178161
 1h     348        105      0.542857             0.295731          125        0.416000               0.234966  0.660920     0.339080
