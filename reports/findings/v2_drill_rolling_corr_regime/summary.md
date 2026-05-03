# Drill: rolling corr as regime classifier - 2026-05-03 22:39 UTC

**Window**: 200 5m bars (~16.7h)

**Threshold**: |corr| > 0.2 for UP/DOWN, else FLAT

**Baseline regime distribution**: {'FLAT': 0.571, 'UP': 0.247, 'DOWN': 0.182}

## Pair classification accuracy

                               pair  tf  n_days  overall_acc   up_acc  down_acc  flat_acc  cm_up_to_up  cm_up_to_down  cm_up_to_flat  cm_down_to_up  cm_down_to_down  cm_down_to_flat  cm_flat_to_up  cm_flat_to_down  cm_flat_to_flat
       price_velocity_1b__bar_range 15m     348     0.580460 0.395349  0.555556  0.668342           34              8             44              4               35               24             24               42              133
                    bar_range__body 15m     348     0.563218 0.372093  0.523810  0.658291           32             10             44              5               33               25             26               42              131
    price_velocity_w__price_sigma_w  5m     347     0.559078 0.569767  0.698413  0.510101           49             10             27              7               44               12             37               60              101
       price_velocity_1b__bar_range  1h     347     0.541787 0.581395  0.619048  0.500000           50             15             21              7               39               17             38               61               99
                    bar_range__body  1h     347     0.538905 0.616279  0.603175  0.484848           53             15             18              7               38               18             42               60               96
    price_velocity_w__price_sigma_w 15m     347     0.515850 0.662791  0.698413  0.393939           57             11             18             10               44                9             52               68               78
        price_velocity_w__SE_high_w 15m     347     0.498559 0.511628  0.587302  0.464646           44             12             30             10               37               16             54               52               92
         price_velocity_w__SE_low_w 15m     347     0.469741 0.465116  0.634921  0.419192           40             15             31              8               40               15             43               72               83
    price_velocity_w__swing_noise_w  1h     327     0.449541 0.461538  0.600000  0.396825           36             22             20             14               36               10             59               55               75
    price_velocity_w__price_sigma_w  1h     347     0.400576 0.488372  0.587302  0.303030           42             24             20             10               37               16             67               71               60
       price_velocity_w__vol_mean_w  1h     347     0.365994 0.406977  0.634921  0.262626           35             21             30             10               40               13             80               66               52
price_velocity_1b__reversion_prob_w  1h     347     0.291066 0.139535  0.126984  0.409091           12             41             33             42                8               13             78               39               81
