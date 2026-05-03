# V2 features within-TF drift (Layer D7) - 2026-05-03 20:02 UTC

Half-1 IS dates: 2025-01-01 -> 2025-05-11  (104 days)

Half-2 IS dates: 2025-05-12 -> 2025-09-17  (104 days)

Sign flips between halves (within regime): 182/7590 (2.4%)

## Top sign-flip drifters

 tf                c1               c2   regime_2d      r_h1      r_h2     delta  abs_delta  sign_flip  n_h1  n_h2  abs_max
 1h  price_velocity_w        SE_high_w FLAT_SMOOTH -0.199445  0.890147  1.089593   1.089593       True  5562  4176 0.890147
 1h  price_velocity_w    price_sigma_w FLAT_SMOOTH -0.353213  0.885445  1.238658   1.238658       True  5562  4176 0.885445
15m  price_velocity_w        SE_high_w FLAT_SMOOTH -0.284468  0.881327  1.165795   1.165795       True  5679  4176 0.881327
 1h  price_velocity_w         SE_low_w FLAT_SMOOTH -0.292307  0.867758  1.160066   1.160066       True  5562  4176 0.867758
15m  price_velocity_w         SE_low_w FLAT_SMOOTH -0.456137  0.862932  1.319070   1.319070       True  5679  4176 0.862932
15m  price_velocity_w    price_sigma_w FLAT_SMOOTH -0.383579  0.861376  1.244955   1.244955       True  5679  4176 0.861376
 5m  price_velocity_w        SE_high_w FLAT_SMOOTH -0.233491  0.809813  1.043304   1.043304       True  5708  4176 0.809813
 1h      price_mean_w    swing_noise_w DOWN_SMOOTH -0.802314  0.085944  0.888258   0.888258       True  2778  1884 0.802314
 5m  price_velocity_w    price_sigma_w FLAT_SMOOTH -0.375355  0.799870  1.175225   1.175225       True  5708  4176 0.799870
 1h            vwap_w    swing_noise_w DOWN_SMOOTH -0.796547  0.074506  0.871053   0.871053       True  2778  1884 0.796547
 5m  price_velocity_w         SE_low_w FLAT_SMOOTH -0.380359  0.782739  1.163099   1.163099       True  5708  4176 0.782739
 1h  price_velocity_w        SE_high_w DOWN_SMOOTH -0.767124  0.056526  0.823650   0.823650       True  2778  1884 0.767124
 1h        vol_mean_w         SE_low_w FLAT_SMOOTH  0.708159 -0.091525 -0.799684   0.799684       True  5574  4176 0.708159
 1h        vol_mean_w        SE_high_w FLAT_SMOOTH  0.686545 -0.115902 -0.802447   0.802447       True  5574  4176 0.686545
 1m  price_velocity_w        SE_high_w FLAT_SMOOTH -0.201344  0.676717  0.878061   0.878061       True  5714  4176 0.676717
 1m  price_velocity_w    price_sigma_w FLAT_SMOOTH -0.293186  0.661911  0.955097   0.955097       True  5714  4176 0.661911
 1m  price_velocity_w         SE_low_w FLAT_SMOOTH -0.291109  0.654934  0.946043   0.946043       True  5714  4176 0.654934
 1h      price_mean_w    price_sigma_w DOWN_SMOOTH -0.652906  0.125692  0.778598   0.778598       True  2778  1884 0.652906
 1h     price_sigma_w           vwap_w DOWN_SMOOTH -0.650314  0.103346  0.753660   0.753660       True  2778  1884 0.650314
 1h       vol_sigma_w         SE_low_w FLAT_SMOOTH  0.612701 -0.088210 -0.700911   0.700911       True  5574  4176 0.612701
 1h       vol_sigma_w        SE_high_w FLAT_SMOOTH  0.599153 -0.119305 -0.718458   0.718458       True  5574  4176 0.599153
 1h     price_sigma_w       vol_mean_w FLAT_SMOOTH  0.594776 -0.145161 -0.739937   0.739937       True  5574  4176 0.594776
15m  price_velocity_w    swing_noise_w FLAT_SMOOTH -0.213459  0.590327  0.803787   0.803787       True  5628  4176 0.590327
 1h  price_velocity_w     price_mean_w DOWN_SMOOTH  0.561857 -0.097276 -0.659133   0.659133       True  2778  1884 0.561857
 1h  price_velocity_w           vwap_w DOWN_SMOOTH  0.561521 -0.085421 -0.646942   0.646942       True  2778  1884 0.561521
 1m  price_velocity_w    swing_noise_w FLAT_SMOOTH -0.210451  0.553011  0.763462   0.763462       True  5712  4176 0.553011
 1h  price_velocity_w         SE_low_w   UP_CHOPPY -0.259633  0.525188  0.784821   0.784821       True   996  2928 0.525188
 1h     price_sigma_w      vol_sigma_w FLAT_SMOOTH  0.500000 -0.141312 -0.641312   0.641312       True  5574  4176 0.500000
 5m  price_velocity_w    swing_noise_w FLAT_SMOOTH -0.189353  0.497330  0.686682   0.686682       True  5688  4176 0.497330
 1h     price_sigma_w          hurst_w   UP_CHOPPY  0.464616 -0.116325 -0.580941   0.580941       True   996  2928 0.464616
 1h  price_velocity_w    price_sigma_w FLAT_CHOPPY -0.452003  0.268110  0.720113   0.720113       True  8736  9579 0.452003
 1h        vol_mean_w    swing_noise_w DOWN_CHOPPY  0.425699 -0.067414 -0.493113   0.493113       True  1716  1272 0.425699
 1h         SE_high_w          hurst_w   UP_CHOPPY  0.420861 -0.162869 -0.583730   0.583730       True   996  2928 0.420861
 1h  price_velocity_w         SE_low_w FLAT_CHOPPY -0.408006  0.192238  0.600244   0.600244       True  8736  9579 0.408006
 1h price_velocity_1b reversion_prob_w FLAT_SMOOTH  0.304715 -0.393799 -0.698514   0.698514       True  5574  4176 0.393799
 1h  price_velocity_w    swing_noise_w DOWN_CHOPPY -0.385929  0.178008  0.563937   0.563937       True  1716  1272 0.385929
 1h       vol_sigma_w          hurst_w DOWN_SMOOTH -0.118899  0.375019  0.493918   0.493918       True  2778  1884 0.375019
 1h  price_velocity_w        SE_high_w FLAT_CHOPPY -0.367059  0.290706  0.657765   0.657765       True  8736  9579 0.367059
 1h        vol_mean_w    swing_noise_w FLAT_SMOOTH  0.360630 -0.240009 -0.600639   0.600639       True  5358  4176 0.360630
 1h     price_accel_w reversion_prob_w FLAT_SMOOTH  0.229818 -0.358201 -0.588019   0.588019       True  5550  4176 0.358201
 1h  price_velocity_w    swing_noise_w FLAT_CHOPPY -0.351049  0.165588  0.516637   0.516637       True  8736  9579 0.351049
15m           hurst_w    swing_noise_w   UP_CHOPPY -0.327107  0.050483  0.377590   0.377590       True   996  2928 0.327107
 1h      price_mean_w      vol_sigma_w   UP_CHOPPY -0.319730  0.119910  0.439639   0.439639       True   996  2928 0.319730
 1h       vol_sigma_w    swing_noise_w FLAT_SMOOTH  0.315953 -0.227851 -0.543804   0.543804       True  5358  4176 0.315953
 1h price_velocity_1b         SE_low_w FLAT_SMOOTH -0.079571  0.311286  0.390857   0.390857       True  5574  4176 0.311286
 1h       vol_sigma_w           vwap_w   UP_CHOPPY -0.310679  0.120282  0.430961   0.430961       True   996  2928 0.310679
 1h  price_velocity_w          hurst_w   UP_CHOPPY  0.304456 -0.303817 -0.608273   0.608273       True   996  2928 0.304456
 5s  price_velocity_w        SE_high_w   UP_CHOPPY -0.291306  0.061450  0.352756   0.352756       True   996  2928 0.291306
 1h     price_sigma_w          hurst_w DOWN_CHOPPY -0.063793  0.282906  0.346699   0.346699       True  1716  1272 0.282906
15m  price_velocity_w           vwap_w DOWN_SMOOTH  0.282105 -0.089594 -0.371699   0.371699       True  2778  1884 0.282105

## Top stable high-magnitude pairs

 tf                c1     c2   regime_2d     r_h1     r_h2         delta    abs_delta  sign_flip  n_h1  n_h2  abs_mean
 5s      price_mean_w vwap_w FLAT_SMOOTH 1.000000 1.000000  5.934757e-08 5.934757e-08      False  5717  4176  1.000000
 5s      price_mean_w vwap_w DOWN_SMOOTH 1.000000 1.000000  2.390527e-08 2.390527e-08      False  2778  1884  1.000000
 5s      price_mean_w vwap_w   UP_SMOOTH 0.999999 1.000000  2.697527e-07 2.697527e-07      False  3324  3822  1.000000
 5s      price_mean_w vwap_w FLAT_CHOPPY 1.000000 0.999999 -9.475416e-08 9.475416e-08      False  8736  9579  1.000000
 5s      price_mean_w vwap_w   UP_CHOPPY 0.999999 1.000000  5.808786e-07 5.808786e-07      False   996  2928  1.000000
 5s      price_mean_w vwap_w DOWN_CHOPPY 1.000000 0.999997 -2.335962e-06 2.335962e-06      False  1716  1272  0.999998
 1m      price_mean_w vwap_w   UP_CHOPPY 0.999992 0.999995  2.761009e-06 2.761009e-06      False   996  2928  0.999993
 1m      price_mean_w vwap_w FLAT_CHOPPY 0.999992 0.999994  1.935125e-06 1.935125e-06      False  8736  9579  0.999993
 1m      price_mean_w vwap_w   UP_SMOOTH 0.999986 0.999996  1.047296e-05 1.047296e-05      False  3324  3822  0.999991
 1m      price_mean_w vwap_w DOWN_SMOOTH 0.999973 0.999996  2.343538e-05 2.343538e-05      False  2778  1884  0.999984
 1m      price_mean_w vwap_w FLAT_SMOOTH 0.999990 0.999974 -1.616126e-05 1.616126e-05      False  5714  4176  0.999982
 5m      price_mean_w vwap_w   UP_CHOPPY 0.999979 0.999979 -5.507021e-07 5.507021e-07      False   996  2928  0.999979
 5m      price_mean_w vwap_w FLAT_CHOPPY 0.999978 0.999979  5.215716e-07 5.215716e-07      False  8736  9579  0.999979
 1m      price_mean_w vwap_w DOWN_CHOPPY 0.999989 0.999963 -2.582385e-05 2.582385e-05      False  1716  1272  0.999976
 5m      price_mean_w vwap_w DOWN_SMOOTH 0.999954 0.999982  2.853064e-05 2.853064e-05      False  2778  1884  0.999968
 5m      price_mean_w vwap_w   UP_SMOOTH 0.999950 0.999983  3.289587e-05 3.289587e-05      False  3324  3822  0.999967
 5m      price_mean_w vwap_w DOWN_CHOPPY 0.999962 0.999913 -4.915876e-05 4.915876e-05      False  1716  1272  0.999938
 5m      price_mean_w vwap_w FLAT_SMOOTH 0.999971 0.999860 -1.111637e-04 1.111637e-04      False  5709  4176  0.999916
15m      price_mean_w vwap_w   UP_CHOPPY 0.999915 0.999884 -3.113453e-05 3.113453e-05      False   996  2928  0.999899
15m      price_mean_w vwap_w FLAT_CHOPPY 0.999878 0.999888  9.543244e-06 9.543244e-06      False  8736  9579  0.999883
15m      price_mean_w vwap_w DOWN_SMOOTH 0.999835 0.999890  5.461779e-05 5.461779e-05      False  2778  1884  0.999863
15m      price_mean_w vwap_w   UP_SMOOTH 0.999816 0.999876  5.927047e-05 5.927047e-05      False  3324  3822  0.999846
15m      price_mean_w vwap_w DOWN_CHOPPY 0.999837 0.999684 -1.529342e-04 1.529342e-04      False  1716  1272  0.999761
15m      price_mean_w vwap_w FLAT_SMOOTH 0.999955 0.999357 -5.977633e-04 5.977633e-04      False  5682  4176  0.999656
 1h      price_mean_w vwap_w   UP_CHOPPY 0.999304 0.999643  3.384863e-04 3.384863e-04      False   996  2928  0.999474
15m price_velocity_1b   body   UP_CHOPPY 0.999404 0.999215 -1.889045e-04 1.889045e-04      False   996  2928  0.999309
 1h      price_mean_w vwap_w FLAT_CHOPPY 0.998996 0.999185  1.885967e-04 1.885967e-04      False  8736  9579  0.999091
 5m price_velocity_1b   body   UP_CHOPPY 0.999242 0.998788 -4.543986e-04 4.543986e-04      False   996  2928  0.999015
 1h price_velocity_1b   body   UP_CHOPPY 0.998768 0.999110  3.417889e-04 3.417889e-04      False   996  2928  0.998939
 1h      price_mean_w vwap_w   UP_SMOOTH 0.998796 0.998948  1.513980e-04 1.513980e-04      False  3324  3822  0.998872
 1h      price_mean_w vwap_w DOWN_SMOOTH 0.998170 0.998806  6.356862e-04 6.356862e-04      False  2778  1884  0.998488
 1h      price_mean_w vwap_w FLAT_SMOOTH 0.999493 0.996774 -2.719103e-03 2.719103e-03      False  5574  4176  0.998133
 1h price_velocity_1b   body FLAT_CHOPPY 0.996862 0.999207  2.344929e-03 2.344929e-03      False  8736  9579  0.998034
15m price_velocity_1b   body FLAT_CHOPPY 0.997168 0.998879  1.710620e-03 1.710620e-03      False  8736  9579  0.998023
 1m price_velocity_1b   body   UP_CHOPPY 0.999328 0.996431 -2.896701e-03 2.896701e-03      False   996  2928  0.997880
 5m price_velocity_1b   body FLAT_CHOPPY 0.996745 0.998163  1.417904e-03 1.417904e-03      False  8736  9579  0.997454
 1m price_velocity_1b   body DOWN_SMOOTH 0.998039 0.996849 -1.189566e-03 1.189566e-03      False  2778  1884  0.997444
 1m price_velocity_1b   body DOWN_CHOPPY 0.998006 0.996766 -1.240550e-03 1.240550e-03      False  1716  1272  0.997386
 1h      price_mean_w vwap_w DOWN_CHOPPY 0.998699 0.995885 -2.813768e-03 2.813768e-03      False  1716  1272  0.997292
 1m price_velocity_1b   body FLAT_CHOPPY 0.998001 0.993962 -4.038911e-03 4.038911e-03      False  8736  9579  0.995982
 1m price_velocity_1b   body   UP_SMOOTH 0.998597 0.989069 -9.527935e-03 9.527935e-03      False  3324  3822  0.993833
 1m price_velocity_1b   body FLAT_SMOOTH 0.996884 0.990029 -6.854670e-03 6.854670e-03      False  5717  4176  0.993457
 5m price_velocity_1b   body DOWN_CHOPPY 0.987746 0.996211  8.464614e-03 8.464614e-03      False  1716  1272  0.991979
 1h price_velocity_1b   body DOWN_CHOPPY 0.986724 0.996932  1.020859e-02 1.020859e-02      False  1716  1272  0.991828
15m price_velocity_1b   body DOWN_CHOPPY 0.986909 0.996533  9.624177e-03 9.624177e-03      False  1716  1272  0.991721
 1h price_velocity_1b   body   UP_SMOOTH 0.978969 0.986917  7.947824e-03 7.947824e-03      False  3324  3822  0.982943
15m price_velocity_1b   body   UP_SMOOTH 0.968837 0.986052  1.721491e-02 1.721491e-02      False  3324  3822  0.977445
15m price_velocity_1b   body DOWN_SMOOTH 0.967935 0.983197  1.526204e-02 1.526204e-02      False  2778  1884  0.975566
 5m price_velocity_1b   body DOWN_SMOOTH 0.968833 0.980291  1.145766e-02 1.145766e-02      False  2778  1884  0.974562
 1h price_velocity_1b   body DOWN_SMOOTH 0.966090 0.981961  1.587064e-02 1.587064e-02      False  2778  1884  0.974026

## D2 sign-flip confirmation

Of 254 D2 regime-flip pairs, **83** (32.7%) have the regime difference confirmed in BOTH IS halves.

 tf                c1               c2  regime_min  regime_max  d2_r_min  d2_r_max    min_h1    min_h2   max_h1   max_h2  confirmed  min_mag
15m  price_velocity_w    price_sigma_w DOWN_SMOOTH   UP_SMOOTH -0.807789  0.846109 -0.812452 -0.773125 0.856755 0.771949       True 0.771949
15m  price_velocity_w         SE_low_w DOWN_SMOOTH   UP_SMOOTH -0.734075  0.760505 -0.747148 -0.612618 0.795491 0.562814       True 0.562814
15m  price_velocity_w        SE_high_w DOWN_SMOOTH   UP_SMOOTH -0.698828  0.787214 -0.713701 -0.558942 0.809975 0.624000       True 0.558942
 5m  price_velocity_w    price_sigma_w DOWN_SMOOTH   UP_SMOOTH -0.624167  0.703391 -0.635369 -0.557655 0.724478 0.616187       True 0.557655
15m  price_velocity_w    swing_noise_w DOWN_SMOOTH   UP_SMOOTH -0.564644  0.615484 -0.562960 -0.530109 0.621447 0.520260       True 0.520260
 1h price_velocity_1b        bar_range DOWN_SMOOTH   UP_SMOOTH -0.599734  0.696008 -0.620578 -0.487014 0.719206 0.582803       True 0.487014
 1h         bar_range             body DOWN_SMOOTH   UP_SMOOTH -0.590246  0.712466 -0.614425 -0.482120 0.738045 0.597500       True 0.482120
15m  price_velocity_w       vol_mean_w DOWN_SMOOTH   UP_SMOOTH -0.527726  0.485101 -0.518632 -0.576808 0.500797 0.464884       True 0.464884
15m  price_velocity_w      vol_sigma_w DOWN_SMOOTH   UP_SMOOTH -0.475026  0.442120 -0.460737 -0.570972 0.469485 0.481455       True 0.460737
 1h price_velocity_1b reversion_prob_w   UP_SMOOTH DOWN_SMOOTH -0.421956  0.531059 -0.466413 -0.441254 0.560856 0.469590       True 0.441254
 1h  price_velocity_w    price_sigma_w DOWN_SMOOTH   UP_SMOOTH -0.837553  0.887296 -0.910564 -0.434527 0.901023 0.712216       True 0.434527
 1h              body reversion_prob_w   UP_SMOOTH DOWN_SMOOTH -0.401644  0.534669 -0.436320 -0.434164 0.561852 0.472296       True 0.434164
15m price_velocity_1b        bar_range DOWN_SMOOTH   UP_SMOOTH -0.416720  0.518903 -0.422898 -0.398508 0.540157 0.444072       True 0.398508
15m         bar_range             body DOWN_SMOOTH   UP_SMOOTH -0.374845  0.530222 -0.374810 -0.388690 0.553853 0.454184       True 0.374810
 1m  price_velocity_w    price_sigma_w DOWN_SMOOTH   UP_SMOOTH -0.498496  0.534832 -0.517734 -0.374095 0.559870 0.429770       True 0.374095
 5m  price_velocity_w         SE_low_w DOWN_SMOOTH   UP_SMOOTH -0.531145  0.566646 -0.555797 -0.340931 0.596389 0.422004       True 0.340931
 1h     price_accel_w reversion_prob_w   UP_SMOOTH DOWN_SMOOTH -0.352744  0.430004 -0.388713 -0.331328 0.445917 0.393627       True 0.331328
 1h         bar_range    price_accel_w DOWN_SMOOTH   UP_SMOOTH -0.410146  0.582952 -0.430919 -0.330280 0.628499 0.427797       True 0.330280
 5m  price_velocity_w      vol_sigma_w DOWN_SMOOTH   UP_SMOOTH -0.379281  0.313256 -0.374147 -0.426114 0.326547 0.319469       True 0.319469
 5m  price_velocity_w        SE_high_w DOWN_SMOOTH   UP_SMOOTH -0.482172  0.629295 -0.505458 -0.314618 0.653791 0.507442       True 0.314618
 1h         bar_range           z_se_w DOWN_SMOOTH   UP_SMOOTH -0.382062  0.282932 -0.420370 -0.392542 0.307346 0.343211       True 0.307346
 5m  price_velocity_w       vol_mean_w DOWN_SMOOTH   UP_SMOOTH -0.341435  0.317185 -0.334032 -0.365762 0.325255 0.304685       True 0.304685
 1h            z_se_w reversion_prob_w   UP_SMOOTH DOWN_SMOOTH -0.343267  0.474929 -0.389771 -0.299614 0.462145 0.506946       True 0.299614
 1m  price_velocity_w        SE_high_w DOWN_SMOOTH   UP_SMOOTH -0.419550  0.449870 -0.439448 -0.288702 0.476775 0.323742       True 0.288702
 1h  price_velocity_w    swing_noise_w DOWN_SMOOTH   UP_SMOOTH -0.751025  0.766465 -0.826308 -0.287785 0.806105 0.300466       True 0.287785
 1m  price_velocity_w    swing_noise_w DOWN_SMOOTH   UP_SMOOTH -0.376883  0.454814 -0.391750 -0.283800 0.474888 0.376690       True 0.283800
 5m  price_velocity_w    swing_noise_w DOWN_SMOOTH   UP_SMOOTH -0.371849  0.508506 -0.381392 -0.281919 0.524314 0.428590       True 0.281919
 1h  price_velocity_w       vol_mean_w DOWN_SMOOTH   UP_SMOOTH -0.682212  0.548084 -0.738456 -0.398070 0.642671 0.279207       True 0.279207
 5m         bar_range price_velocity_w DOWN_SMOOTH   UP_SMOOTH -0.397068  0.447926 -0.391283 -0.409325 0.477134 0.278224       True 0.278224
 1h    vol_velocity_w         z_high_w DOWN_SMOOTH   UP_SMOOTH -0.403196  0.338354 -0.466865 -0.272484 0.369223 0.309211       True 0.272484
15m         bar_range price_velocity_w DOWN_SMOOTH   UP_SMOOTH -0.551930  0.462450 -0.544902 -0.545823 0.477950 0.261410       True 0.261410
15m price_velocity_1b reversion_prob_w   UP_SMOOTH DOWN_SMOOTH -0.258547  0.301988 -0.284218 -0.260218 0.327344 0.260660       True 0.260218
 5m price_velocity_1b        bar_range DOWN_SMOOTH   UP_SMOOTH -0.320301  0.333793 -0.332679 -0.257263 0.351007 0.270013       True 0.257263
 1h         bar_range         z_high_w DOWN_SMOOTH   UP_SMOOTH -0.236237  0.318589 -0.254401 -0.255001 0.326225 0.435985       True 0.254401
 1m  price_velocity_w         SE_low_w DOWN_SMOOTH   UP_SMOOTH -0.441609  0.432006 -0.460085 -0.315648 0.470452 0.252750       True 0.252750
 1h  price_velocity_w      vol_sigma_w DOWN_CHOPPY   UP_CHOPPY -0.527317  0.441616 -0.462798 -0.680056 0.569566 0.245765       True 0.245765
 5m         bar_range             body DOWN_SMOOTH   UP_SMOOTH -0.266117  0.335002 -0.271474 -0.244564 0.351375 0.282129       True 0.244564
15m         bar_range    price_accel_w DOWN_SMOOTH   UP_SMOOTH -0.243685  0.339230 -0.249418 -0.240551 0.356875 0.311418       True 0.240551
 1m  price_velocity_w      vol_sigma_w DOWN_SMOOTH   UP_SMOOTH -0.278574  0.237099 -0.269730 -0.335824 0.245530 0.240025       True 0.240025
 1h              body   vol_velocity_w DOWN_SMOOTH   UP_SMOOTH -0.348035  0.238882 -0.378859 -0.248092 0.235382 0.282142       True 0.235382
 1h price_velocity_1b   vol_velocity_w DOWN_SMOOTH   UP_SMOOTH -0.319056  0.238370 -0.338855 -0.255029 0.234431 0.284051       True 0.234431
15m              body reversion_prob_w   UP_SMOOTH DOWN_SMOOTH -0.220650  0.274404 -0.227546 -0.246792 0.294447 0.239593       True 0.227546
 1h    price_accel_1b reversion_prob_w   UP_SMOOTH DOWN_SMOOTH -0.254767  0.327531 -0.298120 -0.226909 0.355166 0.253599       True 0.226909
 1m         bar_range price_velocity_w DOWN_SMOOTH   UP_SMOOTH -0.298093  0.306731 -0.301410 -0.270389 0.320663 0.226566       True 0.226566
 1m  price_velocity_w       vol_mean_w DOWN_SMOOTH   UP_SMOOTH -0.241749  0.222528 -0.232144 -0.283028 0.224520 0.224285       True 0.224285
15m     price_accel_w   vol_velocity_w DOWN_SMOOTH   UP_SMOOTH -0.234658  0.217105 -0.225226 -0.284412 0.230310 0.207265       True 0.207265
 1h     price_accel_w   vol_velocity_w DOWN_SMOOTH   UP_SMOOTH -0.334655  0.224638 -0.375445 -0.202890 0.235915 0.208797       True 0.202890
 1h    vol_velocity_w           z_se_w DOWN_SMOOTH   UP_SMOOTH -0.383077  0.347017 -0.488253 -0.185669 0.385984 0.307756       True 0.185669
 1h    vol_velocity_w          z_low_w DOWN_SMOOTH   UP_SMOOTH -0.357565  0.358129 -0.450011 -0.184131 0.476826 0.229292       True 0.184131
15m     price_accel_w reversion_prob_w   UP_SMOOTH DOWN_SMOOTH -0.190837  0.222922 -0.218511 -0.173309 0.235397 0.212958       True 0.173309
