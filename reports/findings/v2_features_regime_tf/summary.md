# V2 features TF sweep × regime — 2026-05-03 17:13 UTC

**Base TF:** `5m`  **Split:** `IS`  **Min cell n:** 200

## Top 30 (concept, TF) by corr_fwd range across regimes

         concept  tf  n_regimes  n_pos  n_neg  has_inversion  min_corr  max_corr  corr_range
       bar_range  5m          6      2      4           True -0.157195  0.234748    0.391942
   price_sigma_w 15s          6      2      4           True -0.155351  0.222207    0.377558
       bar_range  1m          6      2      4           True -0.151870  0.212947    0.364817
       bar_range  1h          6      3      3           True -0.158113  0.198127    0.356239
       bar_range 15m          6      2      4           True -0.155493  0.185856    0.341349
       bar_range  4h          6      4      2           True -0.158241  0.181004    0.339246
   price_sigma_w  5s          6      2      4           True -0.115774  0.222795    0.338569
       bar_range 15s          6      2      4           True -0.142215  0.193394    0.335609
   price_sigma_w  1m          6      3      3           True -0.139376  0.191835    0.331211
       bar_range  5s          6      2      4           True -0.144122  0.184683    0.328805
price_velocity_w  5m          6      1      5           True -0.220530  0.105725    0.326255
price_velocity_w  4h          6      1      5           True -0.209567  0.114434    0.324001
     vol_sigma_w  5m          6      2      4           True -0.176509  0.146524    0.323033
   price_sigma_w  4h          6      4      2           True -0.133824  0.173980    0.307804
   price_sigma_w  5m          6      4      2           True -0.128713  0.174070    0.302783
      vol_mean_w 15s          6      2      4           True -0.152659  0.150021    0.302680
      vol_mean_w 15m          6      3      3           True -0.155887  0.143459    0.299346
      vol_mean_w  1m          6      2      4           True -0.142565  0.156366    0.298931
    price_mean_w  5s          6      2      4           True -0.177321  0.120743    0.298065
          vwap_w  5s          6      2      4           True -0.177320  0.120738    0.298059
          vwap_w 15s          6      2      4           True -0.177246  0.120766    0.298012
    price_mean_w 15s          6      2      4           True -0.177224  0.120784    0.298008
          vwap_w  1m          6      2      4           True -0.175599  0.121004    0.296604
    price_mean_w  1m          6      2      4           True -0.175173  0.120869    0.296042
          vwap_w  5m          6      2      4           True -0.172277  0.121053    0.293330
    price_mean_w  5m          6      2      4           True -0.171353  0.120896    0.292249
          vwap_w 15m          6      2      4           True -0.167453  0.122768    0.290222
    price_mean_w 15m          6      2      4           True -0.166646  0.123514    0.290160
          vwap_w  1h          6      3      3           True -0.159758  0.129749    0.289507
    price_mean_w  1h          6      4      2           True -0.161760  0.124124    0.285884

## Multiscale character

### Mean |price_velocity_w| per (regime, TF) — trend strength proxy

tf              5s    15s     1m     5m    15m      1h      4h      1D
regime_2d                                                             
UP_SMOOTH    0.375  0.573  1.104  3.431  6.349  16.924  22.255  86.555
UP_CHOPPY    0.429  0.671  1.285  3.570  5.420  14.670  22.064  84.331
DOWN_SMOOTH  0.476  0.737  1.459  4.404  7.446  20.434  23.445  99.937
DOWN_CHOPPY  0.442  0.683  1.280  3.692  5.628  12.124  21.136  88.501
FLAT_SMOOTH  0.320  0.493  0.975  2.853  4.438   9.580  21.369  93.465
FLAT_CHOPPY  0.426  0.675  1.293  3.810  5.847  10.588  18.845  83.288

### Mean swing_noise_w per (regime, TF) — chop strength proxy

tf               5s      15s       1m       5m       15m        1h        4h         1D
regime_2d                                                                              
UP_SMOOTH    52.465   93.088  196.767  483.718   965.007  2026.276  3984.746   9261.542
UP_CHOPPY    60.392  108.704  227.338  525.236   958.365  2340.018  4053.844  10634.520
DOWN_SMOOTH  66.600  120.057  254.483  607.316  1157.941  2263.539  4017.229   9566.578
DOWN_CHOPPY  61.782  109.823  225.487  503.778   861.740  2000.723  4280.422   9318.101
FLAT_SMOOTH  45.326   81.300  172.684  402.260   747.706  1728.782  3946.286  10469.466
FLAT_CHOPPY  61.639  109.016  226.962  530.281   938.950  1978.463  3866.800   9583.103

### Mean bar_range per (regime, TF) — variation proxy

tf              5s    15s     1m      5m     15m      1h       4h       1D
regime_2d                                                                 
UP_SMOOTH    1.597  3.134  7.834  19.229  34.046  70.708  147.833  406.208
UP_CHOPPY    2.000  3.820  9.254  22.833  40.373  80.077  162.571  364.125
DOWN_SMOOTH  2.104  4.068  9.966  25.024  43.890  89.626  179.620  350.207
DOWN_CHOPPY  1.931  3.769  9.414  23.079  39.734  80.162  153.306  421.396
FLAT_SMOOTH  1.337  2.684  6.824  17.029  29.622  59.546  113.260  274.381
FLAT_CHOPPY  1.851  3.632  9.027  22.627  39.746  81.036  158.698  390.518
