**DEPRECATED — LOOKAHEAD ARTIFACT, DO NOT QUOTE**

# Layer 1b: 5m motif sub-clustering within 15m phrase clusters

_Generated 2026-05-10T01:36:36.222565_

Split: IS    phrase_mcs: 30    motif_mcs: 30
Pure Layer 1 (2D shape only). Features: `['slope_pts_per_min', 'mean_sigma', 'sigma_rank_mid', 'r2adj', 'length_min', 'peak_abs_z', 'net_move_pts', 'tod_start_hour_utc']`

## Hierarchy summary (motifs grouped by parent 15m cluster)

```
   parent_15m_cluster  n_motifs  n_5m_subclusters  n_noise
    LINEAR_DOWN_NOISE      1663                 4       47
          NOISE_NOISE       896                 2      162
       FLATLINE_NOISE       831                 7      266
 LOGARITHMIC_UP_NOISE       797                 6       54
  EXPONENTIAL_DOWN_C1       618                 3       10
         LINEAR_UP_C0       605                 3       39
    EXPONENTIAL_UP_C1       495                 4       61
         LINEAR_UP_C2       470                 4      139
LOGARITHMIC_DOWN_C-99       430                 6       79
  EXPONENTIAL_DOWN_C0       413                 2       42
          FLATLINE_C2       347                 3      108
          FLATLINE_C8       339                 3      121
          FLATLINE_C9       335                 5       35
    EXPONENTIAL_UP_C0       328                 3       77
          FLATLINE_C1       319                 4       60
      LINEAR_UP_NOISE       291                 2       35
         LINEAR_UP_C1       283                 3       31
         LINEAR_UP_C3       274                 2       16
          FLATLINE_C3       253                 3       26
 EXPONENTIAL_UP_NOISE       227                 3       18
          FLATLINE_C6       221                 2       20
          FLATLINE_C0       178                 2       15
          FLATLINE_C4       118                 0      118
          FLATLINE_C7       111                 0      111
          FLATLINE_C5        81                 0       81
```

## Per-cluster outcomes (phrase + motif levels)

```
           shape           cluster_15m      level   n  ride_mean  cascade_pct cluster_5m
        FLATLINE                 NOISE phrase_15m 336   1.163690     2.678571        NaN
        FLATLINE                    C0 phrase_15m  90  45.927778     7.777778        NaN
        FLATLINE                    C1 phrase_15m 164   8.408537     0.000000        NaN
        FLATLINE                    C2 phrase_15m 109   3.651376     0.000000        NaN
        FLATLINE                    C3 phrase_15m  82   1.817073    45.121951        NaN
        FLATLINE                    C4 phrase_15m  53   5.669811    20.754717        NaN
        FLATLINE                    C5 phrase_15m  34   0.867647     0.000000        NaN
        FLATLINE                    C6 phrase_15m 112  32.946429     1.785714        NaN
        FLATLINE                    C7 phrase_15m  51   1.730392     0.000000        NaN
        FLATLINE                    C8 phrase_15m 112   3.756696     2.678571        NaN
        FLATLINE                    C9 phrase_15m 143   7.914336     0.000000        NaN
       LINEAR_UP                 NOISE phrase_15m  40  92.825000    47.500000        NaN
       LINEAR_UP                    C0 phrase_15m  70  54.842857    37.142857        NaN
       LINEAR_UP                    C1 phrase_15m  43 151.215116    30.232558        NaN
       LINEAR_UP                    C2 phrase_15m  41  94.140244    56.097561        NaN
       LINEAR_UP                    C3 phrase_15m  47  26.462766    38.297872        NaN
     LINEAR_DOWN                 NOISE phrase_15m 217  88.357143    58.525346        NaN
  EXPONENTIAL_UP                 NOISE phrase_15m  30  76.266667    63.333333        NaN
  EXPONENTIAL_UP                    C0 phrase_15m  49 161.051020    28.571429        NaN
  EXPONENTIAL_UP                    C1 phrase_15m  63  68.000000    46.031746        NaN
EXPONENTIAL_DOWN                 NOISE phrase_15m   4 111.687500    25.000000        NaN
EXPONENTIAL_DOWN                    C0 phrase_15m  38 234.105263    81.578947        NaN
EXPONENTIAL_DOWN                    C1 phrase_15m 100  49.287500    44.000000        NaN
  LOGARITHMIC_UP                 NOISE phrase_15m  84 103.714286    44.047619        NaN
EXPONENTIAL_DOWN   EXPONENTIAL_DOWN_C0   motif_5m  42  22.130952     9.523810      NOISE
EXPONENTIAL_DOWN   EXPONENTIAL_DOWN_C0   motif_5m  72 148.461806    56.944444         C0
EXPONENTIAL_DOWN   EXPONENTIAL_DOWN_C0   motif_5m 299  16.510870     2.341137         C1
EXPONENTIAL_DOWN   EXPONENTIAL_DOWN_C1   motif_5m  10  -4.150000     0.000000      NOISE
EXPONENTIAL_DOWN   EXPONENTIAL_DOWN_C1   motif_5m 111   3.002252     3.603604         C0
EXPONENTIAL_DOWN   EXPONENTIAL_DOWN_C1   motif_5m 177  19.326271     0.564972         C1
EXPONENTIAL_DOWN   EXPONENTIAL_DOWN_C1   motif_5m 320  26.485937    16.875000         C2
  EXPONENTIAL_UP     EXPONENTIAL_UP_C0   motif_5m  77  11.967532     7.792208      NOISE
  EXPONENTIAL_UP     EXPONENTIAL_UP_C0   motif_5m 162  66.776235    17.901235         C0
  EXPONENTIAL_UP     EXPONENTIAL_UP_C0   motif_5m  50   1.245000     0.000000         C1
  EXPONENTIAL_UP     EXPONENTIAL_UP_C0   motif_5m  39  12.961538     0.000000         C2
  EXPONENTIAL_UP     EXPONENTIAL_UP_C1   motif_5m  61   5.901639     3.278689      NOISE
  EXPONENTIAL_UP     EXPONENTIAL_UP_C1   motif_5m  47   5.510638     0.000000         C0
  EXPONENTIAL_UP     EXPONENTIAL_UP_C1   motif_5m 125  43.168000    33.600000         C1
  EXPONENTIAL_UP     EXPONENTIAL_UP_C1   motif_5m 172  14.489826     1.162791         C2
  EXPONENTIAL_UP     EXPONENTIAL_UP_C1   motif_5m  90   2.430556     1.111111         C3
  EXPONENTIAL_UP  EXPONENTIAL_UP_NOISE   motif_5m  18  10.152778     5.555556      NOISE
  EXPONENTIAL_UP  EXPONENTIAL_UP_NOISE   motif_5m  42  64.619048    45.238095         C0
  EXPONENTIAL_UP  EXPONENTIAL_UP_NOISE   motif_5m  54   4.500000     0.000000         C1
  EXPONENTIAL_UP  EXPONENTIAL_UP_NOISE   motif_5m 113   8.219027     0.884956         C2
     FLATLINE_C0           FLATLINE_C0   motif_5m  15 -21.816667     0.000000      NOISE
     FLATLINE_C0           FLATLINE_C0   motif_5m  75   2.280000     0.000000         C0
     FLATLINE_C0           FLATLINE_C0   motif_5m  88  52.610795     3.409091         C1
     FLATLINE_C1           FLATLINE_C1   motif_5m  60   7.470833     0.000000      NOISE
     FLATLINE_C1           FLATLINE_C1   motif_5m  65   1.580769     0.000000         C0
     FLATLINE_C1           FLATLINE_C1   motif_5m  46  14.288043     6.521739         C1
     FLATLINE_C1           FLATLINE_C1   motif_5m  77  14.415584     1.298701         C2
     FLATLINE_C1           FLATLINE_C1   motif_5m  71  -0.982394     0.000000         C3
     FLATLINE_C2           FLATLINE_C2   motif_5m 108   4.344907     0.000000      NOISE
     FLATLINE_C2           FLATLINE_C2   motif_5m 124   5.415323     0.000000         C0
     FLATLINE_C2           FLATLINE_C2   motif_5m  35   0.250000     0.000000         C1
     FLATLINE_C2           FLATLINE_C2   motif_5m  80   5.550000     1.250000         C2
     FLATLINE_C3           FLATLINE_C3   motif_5m  26   9.240385     7.692308      NOISE
     FLATLINE_C3           FLATLINE_C3   motif_5m  91  26.467033     8.791209         C0
     FLATLINE_C3           FLATLINE_C3   motif_5m  44   0.926136     0.000000         C1
     FLATLINE_C3           FLATLINE_C3   motif_5m  92  15.005435     0.000000         C2
     FLATLINE_C4           FLATLINE_C4   motif_5m 118   8.652542     1.694915      NOISE
     FLATLINE_C5           FLATLINE_C5   motif_5m  81   4.086420     0.000000      NOISE
     FLATLINE_C6           FLATLINE_C6   motif_5m  20  23.700000     0.000000      NOISE
     FLATLINE_C6           FLATLINE_C6   motif_5m 134  14.794776     0.000000         C0
     FLATLINE_C6           FLATLINE_C6   motif_5m  67  63.671642     4.477612         C1
     FLATLINE_C7           FLATLINE_C7   motif_5m 111   3.876126     0.000000      NOISE
     FLATLINE_C8           FLATLINE_C8   motif_5m 121   6.423554     9.917355      NOISE
     FLATLINE_C8           FLATLINE_C8   motif_5m  73   0.452055     0.000000         C0
     FLATLINE_C8           FLATLINE_C8   motif_5m  93   3.161290     0.000000         C1
     FLATLINE_C8           FLATLINE_C8   motif_5m  52  19.480769     7.692308         C2
     FLATLINE_C9           FLATLINE_C9   motif_5m  35  11.800000     0.000000      NOISE
     FLATLINE_C9           FLATLINE_C9   motif_5m  37   1.668919     0.000000         C0
     FLATLINE_C9           FLATLINE_C9   motif_5m  68  17.356618     1.470588         C1
     FLATLINE_C9           FLATLINE_C9   motif_5m  32  19.976562    21.875000         C2
     FLATLINE_C9           FLATLINE_C9   motif_5m  99   7.065657     0.000000         C3
     FLATLINE_C9           FLATLINE_C9   motif_5m  64   6.140625     0.000000         C4
  FLATLINE_NOISE        FLATLINE_NOISE   motif_5m 266   6.556391     7.894737      NOISE
  FLATLINE_NOISE        FLATLINE_NOISE   motif_5m 173   6.458092     0.000000         C0
  FLATLINE_NOISE        FLATLINE_NOISE   motif_5m  47   2.340426     0.000000         C1
  FLATLINE_NOISE        FLATLINE_NOISE   motif_5m  62   1.064516     0.000000         C2
  FLATLINE_NOISE        FLATLINE_NOISE   motif_5m  55   7.004545     0.000000         C3
  FLATLINE_NOISE        FLATLINE_NOISE   motif_5m  34  17.257353     0.000000         C4
  FLATLINE_NOISE        FLATLINE_NOISE   motif_5m 103  25.317961     7.766990         C5
  FLATLINE_NOISE        FLATLINE_NOISE   motif_5m  91  17.881868     5.494505         C6
     LINEAR_DOWN     LINEAR_DOWN_NOISE   motif_5m  47  16.936170     0.000000      NOISE
     LINEAR_DOWN     LINEAR_DOWN_NOISE   motif_5m 412  63.798544    37.864078         C0
     LINEAR_DOWN     LINEAR_DOWN_NOISE   motif_5m 624  16.501603     2.403846         C1
     LINEAR_DOWN     LINEAR_DOWN_NOISE   motif_5m 309   2.495146     0.323625         C2
     LINEAR_DOWN     LINEAR_DOWN_NOISE   motif_5m 271   7.285055     0.000000         C3
       LINEAR_UP          LINEAR_UP_C0   motif_5m  39  13.429487     2.564103      NOISE
       LINEAR_UP          LINEAR_UP_C0   motif_5m 190  12.614474     2.631579         C0
       LINEAR_UP          LINEAR_UP_C0   motif_5m 226   4.245575     0.442478         C1
       LINEAR_UP          LINEAR_UP_C0   motif_5m 150  32.701667    22.666667         C2
       LINEAR_UP          LINEAR_UP_C1   motif_5m  31  27.879032     9.677419      NOISE
       LINEAR_UP          LINEAR_UP_C1   motif_5m  99  14.356061     0.000000         C0
       LINEAR_UP          LINEAR_UP_C1   motif_5m  45  48.711111    15.555556         C1
       LINEAR_UP          LINEAR_UP_C1   motif_5m 108  85.942130    17.592593         C2
       LINEAR_UP          LINEAR_UP_C2   motif_5m 139   8.838129     2.158273      NOISE
       LINEAR_UP          LINEAR_UP_C2   motif_5m 129   5.498062     1.550388         C0
       LINEAR_UP          LINEAR_UP_C2   motif_5m  89  53.691011    41.573034         C1
       LINEAR_UP          LINEAR_UP_C2   motif_5m  40  11.506250     0.000000         C2
       LINEAR_UP          LINEAR_UP_C2   motif_5m  73   7.996575     1.369863         C3
       LINEAR_UP          LINEAR_UP_C3   motif_5m  16   1.625000     0.000000      NOISE
       LINEAR_UP          LINEAR_UP_C3   motif_5m  61  35.413934    19.672131         C0
       LINEAR_UP          LINEAR_UP_C3   motif_5m 197   7.804569     3.045685         C1
       LINEAR_UP       LINEAR_UP_NOISE   motif_5m  35  14.307143     2.857143      NOISE
       LINEAR_UP       LINEAR_UP_NOISE   motif_5m 202   9.315594     1.980198         C0
       LINEAR_UP       LINEAR_UP_NOISE   motif_5m  54  85.421296    50.000000         C1
LOGARITHMIC_DOWN LOGARITHMIC_DOWN_C-99   motif_5m  79  15.392405    10.126582      NOISE
LOGARITHMIC_DOWN LOGARITHMIC_DOWN_C-99   motif_5m  34   1.529412     0.000000         C0
LOGARITHMIC_DOWN LOGARITHMIC_DOWN_C-99   motif_5m 107   6.478972     0.000000         C1
LOGARITHMIC_DOWN LOGARITHMIC_DOWN_C-99   motif_5m  63  20.396825     0.000000         C2
LOGARITHMIC_DOWN LOGARITHMIC_DOWN_C-99   motif_5m  41  36.128049     4.878049         C3
LOGARITHMIC_DOWN LOGARITHMIC_DOWN_C-99   motif_5m  42 133.125000    30.952381         C4
LOGARITHMIC_DOWN LOGARITHMIC_DOWN_C-99   motif_5m  64  21.136719    29.687500         C5
  LOGARITHMIC_UP  LOGARITHMIC_UP_NOISE   motif_5m  54  13.245370     1.851852      NOISE
  LOGARITHMIC_UP  LOGARITHMIC_UP_NOISE   motif_5m 124  22.243952     0.000000         C0
  LOGARITHMIC_UP  LOGARITHMIC_UP_NOISE   motif_5m  34   6.838235     0.000000         C1
  LOGARITHMIC_UP  LOGARITHMIC_UP_NOISE   motif_5m 152  66.970395    26.315789         C2
  LOGARITHMIC_UP  LOGARITHMIC_UP_NOISE   motif_5m 254   4.251969     1.574803         C3
  LOGARITHMIC_UP  LOGARITHMIC_UP_NOISE   motif_5m  72  11.788194     8.333333         C4
  LOGARITHMIC_UP  LOGARITHMIC_UP_NOISE   motif_5m 107   0.904206     0.000000         C5
     NOISE_NOISE           NOISE_NOISE   motif_5m 162   0.046296    16.666667      NOISE
     NOISE_NOISE           NOISE_NOISE   motif_5m  40  19.537500    10.000000         C0
     NOISE_NOISE           NOISE_NOISE   motif_5m 694  15.923991     9.221902         C1
```

## Notes

- Each row in `phrases_with_15m_clusters.csv` is one phrase with its
  15m cluster id (per shape).
- Each row in `motifs_with_5m_subclusters.csv` is one motif tagged
  with its parent 15m cluster AND its own 5m sub-cluster id.
- Bayesian-table cell key is now (shape, cluster_15m, cluster_5m).
