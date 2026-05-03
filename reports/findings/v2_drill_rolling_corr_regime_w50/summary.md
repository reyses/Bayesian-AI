# Drill: rolling corr as regime classifier - 2026-05-03 23:14 UTC

**Window**: 50 5m bars (~4.2h)

**Threshold**: |corr| > 0.2 for UP/DOWN, else FLAT

**Baseline regime distribution**: {'FLAT': 0.572, 'UP': 0.248, 'DOWN': 0.179}

## Pair classification accuracy

                               pair  tf  n_days  overall_acc   up_acc  down_acc  flat_acc  cm_up_to_up  cm_up_to_down  cm_up_to_flat  cm_down_to_up  cm_down_to_down  cm_down_to_flat  cm_flat_to_up  cm_flat_to_down  cm_flat_to_flat
    price_velocity_w__price_sigma_w 15m     348     0.681034 0.767442  0.809524  0.603015           66              3             17              2               51               10             41               38              120
    price_velocity_w__price_sigma_w  5m     348     0.678161 0.662791  0.619048  0.703518           57              2             27              2               39               22             25               34              140
       price_velocity_1b__bar_range 15m     348     0.669540 0.441860  0.555556  0.804020           38              2             46              0               35               28             11               28              160
                    bar_range__body 15m     348     0.666667 0.430233  0.571429  0.798995           37              2             47              1               36               26             12               28              159
       price_velocity_1b__bar_range  1h     348     0.649425 0.732558  0.619048  0.623116           63              3             20              1               39               23             30               45              124
                    bar_range__body  1h     348     0.643678 0.709302  0.634921  0.618090           61              3             22              2               40               21             32               44              123
    price_velocity_w__price_sigma_w  1h     347     0.553314 0.651163  0.619048  0.489899           56              5             25              6               39               18             47               54               97
        price_velocity_w__SE_high_w 15m     348     0.543103 0.488372  0.492063  0.582915           42              7             37              6               31               26             41               42              116
         price_velocity_w__SE_low_w 15m     348     0.517241 0.441860  0.523810  0.547739           38              9             39              3               33               27             32               58              109
       price_velocity_w__vol_mean_w  1h     347     0.487032 0.500000  0.555556  0.459596           43              7             36              7               35               21             53               54               91
price_velocity_1b__reversion_prob_w  1h     347     0.345821 0.116279  0.047619  0.540404           10             32             44             32                3               28             57               34              107
    price_velocity_w__swing_noise_w  1h     114     0.166667 0.156250  0.642857  0.073529            5             23              4              2                9                3              6               57                5
