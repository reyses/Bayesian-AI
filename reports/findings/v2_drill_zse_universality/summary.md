# Drill: z_se_w universality - 2026-05-03 22:33 UTC

## A. z_se_w quintile x regime distribution

If z_se_w is a regime proxy, extreme quintiles concentrate in one regime. If uniform, z_se_w carries info beyond regime label.

### TF=5s

regime_2d  UP_SMOOTH  UP_CHOPPY  DOWN_SMOOTH  DOWN_CHOPPY  FLAT_SMOOTH  FLAT_CHOPPY
q_z_se                                                                             
0               15.4        8.1         10.0          6.4         21.0         39.0
1               15.8        7.9          9.8          6.6         20.7         39.3
2               14.9        8.4          9.8          6.5         21.2         39.2
3               14.9        8.5         10.2          6.0         21.2         39.1
4               15.2        8.9          9.9          6.2         21.2         38.5

### TF=1m

regime_2d  UP_SMOOTH  UP_CHOPPY  DOWN_SMOOTH  DOWN_CHOPPY  FLAT_SMOOTH  FLAT_CHOPPY
q_z_se                                                                             
0               15.0        8.3          9.9          6.3         21.5         38.9
1               14.9        8.6          9.9          6.4         20.2         40.0
2               14.9        8.0         10.1          6.4         21.4         39.2
3               15.8        8.4         10.0          6.0         21.0         38.8
4               15.6        8.5          9.7          6.7         21.2         38.3

### TF=5m

regime_2d  UP_SMOOTH  UP_CHOPPY  DOWN_SMOOTH  DOWN_CHOPPY  FLAT_SMOOTH  FLAT_CHOPPY
q_z_se                                                                             
0               14.8        8.3         10.4          6.2         20.9         39.4
1               15.4        8.6          9.6          6.6         20.9         38.9
2               15.2        8.5          9.9          6.6         20.9         38.8
3               15.1        8.1          9.9          6.1         21.2         39.6
4               15.5        8.4          9.8          6.4         21.4         38.5

### TF=15m

regime_2d  UP_SMOOTH  UP_CHOPPY  DOWN_SMOOTH  DOWN_CHOPPY  FLAT_SMOOTH  FLAT_CHOPPY
q_z_se                                                                             
0               14.2        7.8         10.2          7.0         20.4         40.3
1               16.0        9.5          9.4          6.1         20.9         38.1
2               16.8        8.4          9.3          6.0         20.7         38.8
3               13.4        7.9         11.5          6.6         21.7         38.9
4               15.7        8.3          9.2          6.2         21.4         39.1

### TF=1h

regime_2d  UP_SMOOTH  UP_CHOPPY  DOWN_SMOOTH  DOWN_CHOPPY  FLAT_SMOOTH  FLAT_CHOPPY
q_z_se                                                                             
0               13.4        9.0         11.4          7.2         21.8         37.3
1               17.4        7.8          9.6          5.7         22.0         37.3
2               14.0        7.3          9.6          6.7         22.3         40.1
3               13.2        7.9         11.4          8.1         19.1         40.2
4               18.4        9.9          7.8          4.2         19.0         40.8

## B. z_se_w as price contextualizer

Top-20 features whose corr(X, fwd_ret) is most modified by z_se_w bin:

 tf                 X     r_min     r_max     lift  n_bins  sign_flip
 1h     price_accel_w -0.005942  0.128502 0.134444       5      False
 1h    price_accel_1b -0.106483  0.025673 0.132157       5      False
 1h  price_velocity_w -0.062700  0.065251 0.127952       5       True
 1h         bar_range -0.037772  0.086989 0.124761       5      False
15m         bar_range -0.059024  0.062103 0.121126       5       True
15m          SE_low_w -0.046097  0.073538 0.119635       5      False
 1h price_velocity_1b -0.046544  0.065289 0.111833       5      False
15m    vol_velocity_w -0.064504  0.046635 0.111139       5      False
 5s          SE_low_w -0.028031  0.082720 0.110751       5      False
 1h              body -0.052086  0.057257 0.109343       5       True
15m    price_accel_1b -0.064125  0.044672 0.108797       5      False
 5s         SE_high_w -0.018413  0.089907 0.108320       5      False
 5s     swing_noise_w -0.018451  0.088393 0.106844       5      False
 1h      vol_accel_1b -0.075433  0.031371 0.106804       5      False
 1h       vol_sigma_w -0.054304  0.050290 0.104594       5       True
 5s     price_sigma_w -0.012989  0.088394 0.101384       5      False
 1m    price_accel_1b -0.007416  0.093157 0.100574       5      False
 1h        vol_mean_w -0.045748  0.053709 0.099458       5      False
15m            vwap_w -0.055805  0.042297 0.098102       5      False
15m      price_mean_w -0.055777  0.042226 0.098004       5      False
 1h           hurst_w -0.029211  0.066009 0.095220       5      False
15m     price_sigma_w -0.008552  0.086162 0.094714       5      False
 1h     swing_noise_w -0.028084  0.065519 0.093604       5      False
15m           z_low_w -0.064287  0.023914 0.088201       5      False
15m         SE_high_w -0.010672  0.073683 0.084355       5      False
 5s   vol_velocity_1b -0.055562  0.028020 0.083582       5      False
15m     price_accel_w -0.040613  0.042734 0.083347       5      False
15m     swing_noise_w -0.017827  0.065029 0.082856       5      False
 1h          SE_low_w -0.022658  0.059500 0.082158       5      False
 1h  reversion_prob_w -0.085235 -0.005207 0.080027       5      False

## C. Peer comparison

        Z  mean_lift  max_lift  n_triplets
   z_se_w   0.279508  1.793898        1155
  z_low_w   0.223265  1.697162        1155
 z_high_w   0.215273  1.714509        1155
 SE_low_w   0.140941  0.808244        1155
SE_high_w   0.137451  0.775968        1155
  hurst_w   0.091319  0.868309        1155
