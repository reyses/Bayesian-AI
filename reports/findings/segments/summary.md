# Segment population census

_Generated 2026-05-10T00:42:16.767840_

- Days processed: 345 (345 succeeded, 0 failed, 0 skipped from cache)
- Motifs total:   2882
- Melodies total: 13533
- Motif TF: 15m  min_motif_min: 30.0
- Melody TF: 5m  min_melody_min: 5.0

## Motif shape distribution

```
shape_class           n_motifs   pct      mean_len   mean_slope   mean_r2adj   mean_pk_z
-----------------------------------------------------------------------------------------------
FLATLINE                 1599    55.5%      58.6m      -0.002        0.34       2.36
LINEAR_UP                 292    10.1%     260.4m      +0.375        0.43       4.30
LINEAR_DOWN               271     9.4%     246.2m      -0.432        0.37       4.89
NOISE                     200     6.9%     184.7m      +0.038        0.59      11.18
EXPONENTIAL_DOWN          182     6.3%     242.0m      -0.400        0.53       4.59
EXPONENTIAL_UP            168     5.8%     257.6m      +0.353        0.55       4.52
LOGARITHMIC_UP            103     3.6%     292.4m      +0.399        0.42       4.38
LOGARITHMIC_DOWN           66     2.3%     260.4m      -0.407        0.34       5.31
STEP_DOWN                   1     0.0%      30.0m      -0.152        0.73       9.74
```

## Motif IS vs OOS counts by shape

```
shape_class           IS        OOS       IS/d     OOS/d   sign-stable
--------------------------------------------------------------------------------
FLATLINE                 1286      313     4.64    4.60  YES
LINEAR_UP                 241       51     0.87    0.75  YES
LINEAR_DOWN               217       54     0.78    0.79  YES
NOISE                     162       38     0.58    0.56  YES
EXPONENTIAL_DOWN          142       40     0.51    0.59  YES
EXPONENTIAL_UP            142       26     0.51    0.38  YES
LOGARITHMIC_UP             84       19     0.30    0.28  YES
LOGARITHMIC_DOWN           54       12     0.19    0.18  YES
STEP_DOWN                   1        0     0.00    0.00  no
```

## Melody shape distribution

```
shape_class           n_melodies pct      mean_len   mean_slope   mean_r2adj   mean_pk_z
-----------------------------------------------------------------------------------------------
FLATLINE                 8400    62.1%      12.7m      +0.000        0.39       2.11
LINEAR_UP                1322     9.8%      62.4m      +0.806        0.42       4.06
LINEAR_DOWN              1233     9.1%      59.6m      -0.855        0.38       4.37
EXPONENTIAL_DOWN          788     5.8%      49.8m      -0.728        0.49       4.21
EXPONENTIAL_UP            703     5.2%      51.6m      +0.674        0.50       3.88
LOGARITHMIC_UP            497     3.7%      70.8m      +0.837        0.38       3.84
LOGARITHMIC_DOWN          385     2.8%      59.7m      -0.847        0.30       4.00
NOISE                     201     1.5%      55.7m      +0.116        0.58      12.30
STEP_DOWN                   2     0.0%      20.0m      -0.528       -0.00       3.52
STEP_UP                     2     0.0%      24.1m      +0.638        0.41       8.05
```

## Notes

- Motifs = 15m-CRM-anchored macro segments (>= min_motif_min)
- Melodies = 5m-CRM-anchored micro sub-segments NESTED inside motifs
- Each segment carries its segment_chord (slope, sigma_rank, r2adj, shape_class, length, peak_z, ...) computed lookahead-clean over the segment span
- Sign-stable column flags shapes whose IS rate-per-day is within 30% of OOS rate-per-day; non-stable shapes may not generalize
