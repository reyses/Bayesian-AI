# Oracle-labeled segment summary

_Generated 2026-05-10T00:43:16.720461_

- Motifs labeled:   2882
- Melodies labeled: 13533

## MOTIF shape -> outcome (IS only)

```
shape_class                n   ride_$   fade_$  %cascade  %cont_seg med_mfe_ride med_mae_ride
----------------------------------------------------------------------------------------------------
FLATLINE                1286     +9.4     -9.4      5.4%      26.0%        +24.8        -12.4
LINEAR_UP                241    +79.5    -79.5     41.1%      21.6%        +84.8         -9.8
LINEAR_DOWN              217    +88.4    -88.4     58.5%      21.7%       +108.2         -8.2
NOISE                    162     +0.1     -0.1     47.5%       0.6%        +25.2        -27.1
EXPONENTIAL_UP           142   +101.9   -101.9     43.7%      17.6%        +97.4        -12.0
EXPONENTIAL_DOWN         142   +100.5   -100.5     53.5%      19.7%       +121.6        -12.0
LOGARITHMIC_UP            84   +103.7   -103.7     44.0%      17.9%       +121.2         -8.4
LOGARITHMIC_DOWN          54   +112.1   -112.1     61.1%      14.8%        +99.5         -6.8
STEP_DOWN                  1    +13.8    -13.8      0.0%       0.0%        +17.0         -0.8
```

## MELODY shape -> outcome (IS only)

```
shape_class                n   ride_$   fade_$  %cascade  %cont_seg med_mfe_ride med_mae_ride
----------------------------------------------------------------------------------------------------
FLATLINE                6771     +6.6     -6.6      0.2%       2.8%        +12.0         -4.2
LINEAR_UP               1055    +40.0    -40.0     20.2%       1.0%        +41.2         -3.2
LINEAR_DOWN             1004    +43.0    -43.0     22.9%       0.9%        +46.2         -3.2
EXPONENTIAL_DOWN         643    +40.9    -40.9     14.8%       0.8%        +40.0         -3.5
EXPONENTIAL_UP           594    +38.5    -38.5     14.0%       1.7%        +35.0         -3.2
LOGARITHMIC_UP           409    +46.1    -46.1     21.0%       1.7%        +44.2         -4.0
LOGARITHMIC_DOWN         294    +38.0    -38.0     18.7%       0.7%        +37.5         -4.5
NOISE                    168     +0.4     -0.4     25.0%       0.0%        +13.2        -20.8
STEP_UP                    2    +19.9    -19.9      0.0%       0.0%        +27.0         -1.5
STEP_DOWN                  1     +4.8     -4.8      0.0%       0.0%        +25.5         -8.8
```

## Notes

- ride_pnl_pts: signed PnL of trading WITH the segment slope direction
  (entry at start, exit at end). Positive = trading with the slope made money.
- fade_pnl_pts: -ride_pnl_pts. The segment that "gives" ride_pnl is exactly
  the segment that "takes" fade_pnl.
- %cascade: fraction with length>=60min AND peak_abs_z>=4 (the macro-event criterion).
- %cont_seg: fraction whose NEXT segment continues the same slope direction.
- max_mfe/mae_ride: peak favorable / adverse excursion DURING the segment
  if traded with the slope from segment start.
- These outcomes are the lookup values the Bayesian table will key on.