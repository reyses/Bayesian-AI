**DEPRECATED — LOOKAHEAD ARTIFACT, DO NOT QUOTE**

# Optimized within-TF mirror — pruned to 8 family reps

Generated 2026-05-03 22:46 UTC

**Pruned concepts**: ['bar_range', 'body', 'vwap_w', 'vol_velocity_w', 'vol_mean_w', 'z_se_w', 'hurst_w', 'reversion_prob_w']

## Optimized vs unoptimized comparison

            metric           unoptimized         optimized
 D2 sign-flip rate   577 / 1,265 = 45.6%  33 / 140 = 23.6%
 D8 sign-flip rate 4960 / 26,565 = 18.7% 112 / 840 = 13.3%
D9 D2 OOS survival       79 / 83 = 95.2%    5 / 33 = 15.2%
D9 D8 OOS survival      100 / 100 = 100%   32 / 50 = 64.0%

## D2 sign-flip pairs (pruned)

 tf                     pair     r_min    r_max   spread
 1h          bar_range__body -0.590246 0.712466 1.302712
 1h   body__reversion_prob_w -0.401644 0.534669 0.936313
15m          bar_range__body -0.374845 0.530222 0.905067
 1h z_se_w__reversion_prob_w -0.343267 0.474929 0.818196
 1h   vol_velocity_w__z_se_w -0.383077 0.347017 0.730094
 1h        bar_range__z_se_w -0.382062 0.282932 0.664995
 1m          bar_range__body -0.254720 0.366838 0.621558
 5m          bar_range__body -0.266117 0.335002 0.601119
 1h     body__vol_velocity_w -0.348035 0.238882 0.586917
15m   body__reversion_prob_w -0.220650 0.274404 0.495054
 1h          vwap_w__hurst_w -0.306043 0.150271 0.456314
15m z_se_w__reversion_prob_w -0.155297 0.219371 0.374668
 1h       bar_range__hurst_w -0.171087 0.197622 0.368709
 1h       vol_mean_w__z_se_w -0.201481 0.116948 0.318429
15m     body__vol_velocity_w -0.169179 0.143828 0.313006
 5m     body__vol_velocity_w -0.202048 0.099125 0.301173
15m        bar_range__z_se_w -0.172098 0.109247 0.281345
15m   vol_velocity_w__z_se_w -0.150487 0.128702 0.279190
 1m   body__reversion_prob_w -0.106223 0.143459 0.249682
 1h             body__vwap_w -0.142599 0.102859 0.245458
15m         body__vol_mean_w -0.094324 0.132811 0.227134
 1h         body__vol_mean_w -0.102058 0.120831 0.222889
15m          vwap_w__hurst_w -0.119164 0.086112 0.205276
 5m   body__reversion_prob_w -0.061181 0.124486 0.185668
 1m     body__vol_velocity_w -0.098798 0.071417 0.170214
 5m         body__vol_mean_w -0.071428 0.098343 0.169771
15m  vol_velocity_w__hurst_w -0.064748 0.092373 0.157121
15m       bar_range__hurst_w -0.080294 0.068094 0.148389
 5m          vwap_w__hurst_w -0.090229 0.057419 0.147649
 1m         body__vol_mean_w -0.088873 0.056796 0.145669
15m             body__vwap_w -0.083945 0.051358 0.135303
 5m   vol_velocity_w__z_se_w -0.075154 0.058213 0.133367
 1h   vwap_w__vol_velocity_w -0.059959 0.060768 0.120727

## D8 top contextualizer triplets (pruned)

 tf              X                Y                Z     r_min     r_max     lift  sign_flip
 5m      bar_range             body           z_se_w -0.899195  0.894703 1.793898       True
15m      bar_range             body           z_se_w -0.874810  0.883985 1.758795       True
 1h      bar_range             body           z_se_w -0.861118  0.862405 1.723523       True
 5m         z_se_w reversion_prob_w             body -0.848029  0.853570 1.701599       True
 5s      bar_range             body           z_se_w -0.850738  0.836382 1.687120       True
 1m      bar_range             body           z_se_w -0.826988  0.845157 1.672145       True
 5s         z_se_w reversion_prob_w             body -0.789603  0.818023 1.607626       True
 1h         z_se_w reversion_prob_w             body -0.762684  0.781831 1.544515       True
15m         z_se_w reversion_prob_w             body -0.770082  0.764910 1.534992       True
 1m         z_se_w reversion_prob_w             body -0.675219  0.670187 1.345407       True
 5s vol_velocity_w       vol_mean_w        bar_range -0.388830  0.751690 1.140520       True
 1h           body reversion_prob_w           z_se_w -0.459761  0.641104 1.100865       True
 5m           body       vol_mean_w           z_se_w -0.468839  0.447437 0.916276       True
15m           body reversion_prob_w           z_se_w -0.422819  0.488982 0.911801       True
 1h           body   vol_velocity_w           z_se_w -0.461896  0.349296 0.811192       True
 5s           body       vol_mean_w           z_se_w -0.410343  0.382037 0.792381       True
 1m           body       vol_mean_w           z_se_w -0.400746  0.372477 0.773223       True
15m           body       vol_mean_w           z_se_w -0.411427  0.348147 0.759574       True
 5m           body   vol_velocity_w           z_se_w -0.375695  0.343494 0.719188       True
15m           body   vol_velocity_w           z_se_w -0.384939  0.308236 0.693175       True
 1h           body           z_se_w reversion_prob_w  0.031076  0.719032 0.687956      False
 5m           body reversion_prob_w           z_se_w -0.332845  0.351459 0.684303       True
 1m           body reversion_prob_w           z_se_w -0.333015  0.350737 0.683752       True
 1h      bar_range             body          hurst_w -0.238434  0.422778 0.661212       True
15m vol_velocity_w       vol_mean_w        bar_range -0.698797 -0.044361 0.654436      False
 1m           body   vol_velocity_w           z_se_w -0.335188  0.295649 0.630836       True
 1h           body           vwap_w           z_se_w -0.312958  0.317567 0.630525       True
 5s           body reversion_prob_w           z_se_w -0.291997  0.329382 0.621379       True
 5m           body           vwap_w           z_se_w -0.313942  0.297249 0.611191       True
15m           body           vwap_w           z_se_w -0.300989  0.305289 0.606278       True

## Modifier influence ranking (pruned)

               Z  mean_lift  max_lift  n_triplets
          z_se_w   0.301008  1.793898         105
            body   0.213336  1.701599         105
       bar_range   0.120781  1.140520         105
reversion_prob_w   0.104675  0.687956         105
      vol_mean_w   0.093400  0.387370         105
  vol_velocity_w   0.089161  0.401690         105
         hurst_w   0.087371  0.661212         105
          vwap_w   0.082309  0.346062         105
