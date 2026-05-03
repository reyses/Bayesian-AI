# Drill: rolling corr as regime classifier - 2026-05-03 23:15 UTC

**Window**: 50 5m bars (~4.2h)

**Threshold**: |corr| > 0.2 for UP/DOWN, else FLAT

**Baseline regime distribution**: {'FLAT': 0.562, 'UP': 0.226, 'DOWN': 0.212}

## Pair classification accuracy

                               pair  tf  n_days  overall_acc   up_acc  down_acc  flat_acc  cm_up_to_up  cm_up_to_down  cm_up_to_flat  cm_down_to_up  cm_down_to_down  cm_down_to_flat  cm_flat_to_up  cm_flat_to_down  cm_flat_to_flat
                    bar_range__body  1h      71     0.746479 0.875000  0.733333     0.700           14              0              2              0               11                4              8                4               28
       price_velocity_1b__bar_range  1h      71     0.746479 0.875000  0.733333     0.700           14              0              2              0               11                4              7                5               28
                    bar_range__body 15m      71     0.732394 0.500000  0.733333     0.825            8              0              8              0               11                4              5                2               33
       price_velocity_1b__bar_range 15m      71     0.732394 0.500000  0.666667     0.850            8              0              8              0               10                5              4                2               34
    price_velocity_w__price_sigma_w 15m      71     0.718310 0.812500  0.933333     0.600           13              1              2              0               14                1              9                7               24
    price_velocity_w__price_sigma_w  5m      71     0.690141 0.750000  0.733333     0.650           12              0              4              0               11                4              7                7               26
        price_velocity_w__SE_high_w 15m      71     0.563380 0.437500  0.666667     0.575            7              1              8              0               10                5             10                7               23
         price_velocity_w__SE_low_w 15m      71     0.549296 0.312500  0.733333     0.575            5              1             10              0               11                4             10                7               23
    price_velocity_w__swing_noise_w  1h      11     0.545455 0.666667  0.666667     0.400            2              0              1              1                2                0              1                2                2
    price_velocity_w__price_sigma_w  1h      71     0.535211 0.500000  0.733333     0.475            8              0              8              1               11                3             11               10               19
       price_velocity_w__vol_mean_w  1h      71     0.464789 0.250000  0.666667     0.475            4              1             11              1               10                4             13                8               19
price_velocity_1b__reversion_prob_w  1h      71     0.338028 0.000000  0.133333     0.550            0              7              9              8                2                5             11                7               22
