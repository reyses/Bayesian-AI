# V2 Regret Stratified by Regime

Source: training_v2/output/regret_full_nmp.pkl

Trades: 19106

## Best-action distribution per (regime, direction) cell

     regime direction    n  mean_actual  mean_best  mean_regret  mean_fade_peak  mean_flip_peak  mean_early_entry_gain  pct_same_early  pct_same_at_exit  pct_same_extended  pct_counter_early  pct_counter_at_exit  pct_counter_extended
DOWN_CHOPPY      long  651        -0.31     133.45       133.76           73.94           95.31                 186.30            0.31              0.15              45.01               0.15                 0.00                 54.38
DOWN_CHOPPY     short  558         0.36     132.83       132.47           97.86           68.92                 179.03            0.18              0.00              56.09               0.36                 0.00                 43.37
DOWN_SMOOTH      long 1090        -3.35     161.55       164.90           70.29          130.44                 224.71            0.18              0.09              40.09               0.00                 0.00                 59.63
DOWN_SMOOTH     short  899         2.93     155.72       152.79          123.40           69.31                 208.19            0.44              0.00              59.73               0.00                 0.11                 39.71
FLAT_CHOPPY      long 4264         0.61     142.73       142.11           83.64           95.35                 191.80            0.38              0.02              49.67               0.16                 0.07                 49.70
FLAT_CHOPPY     short 3657         0.00     138.71       138.71           91.36           83.41                 187.25            0.33              0.00              50.62               0.14                 0.11                 48.81
FLAT_SMOOTH      long 1729        -0.19     102.74       102.93           61.88           68.59                 137.01            0.58              0.06              50.32               0.23                 0.17                 48.64
FLAT_SMOOTH     short 1631         0.02      94.71        94.70           64.74           57.72                 131.51            0.25              0.00              50.89               0.37                 0.06                 48.44
  UP_CHOPPY      long  852         1.25     139.62       138.37           98.89           77.19                 190.59            0.00              0.00              56.81               0.23                 0.00                 42.96
  UP_CHOPPY     short  752        -0.69     151.73       152.42           83.61          110.24                 211.60            0.53              0.00              41.76               0.13                 0.13                 57.45
  UP_SMOOTH      long 1560         2.56     121.67       119.11           94.08           60.64                 162.86            0.45              0.00              58.91               0.00                 0.06                 40.58
  UP_SMOOTH     short 1463        -1.67     125.59       127.25           61.71           96.53                 172.44            0.68              0.00              41.56               0.00                 0.07                 57.69

## Per-regime regret summary

     regime    n  mean_actual  mean_best  mean_regret  mean_fade_peak  mean_flip_peak  mean_early_gain  median_capture
DOWN_SMOOTH 1989        -0.51     158.92       159.43           94.30          102.81           217.25            0.02
  UP_CHOPPY 1604         0.34     145.30       144.96           91.73           92.68           200.44            0.05
FLAT_CHOPPY 7921         0.33     140.87       140.54           87.21           89.84           189.70            0.03
DOWN_CHOPPY 1209        -0.00     133.16       133.17           84.98           83.13           182.94            0.02
  UP_SMOOTH 3023         0.52     123.57       123.05           78.42           78.01           167.50            0.03
FLAT_SMOOTH 3360        -0.09      98.84        98.93           63.27           63.31           134.34            0.02