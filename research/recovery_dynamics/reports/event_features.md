# What features describe a DEATH event (wrong trade that never comes back)
train 2024 (n=10268, death 30%) -> test 2025 (n=9584, death 25%)

## OOS death-prediction AUC = **0.572**  (gap +0.072 -> **CONDITIONAL** by the signal bar)

## Feature importance (permutation, OOS AUC drop) + direction
```
          feature | importance | corr w/ death
     realized_vol |     0.0231 | -0.093 (less death)
      time_of_day |     0.0181 | -0.115 (less death)
        range_pts |     0.0104 | -0.086 (less death)
    adverse_drift |     0.0022 | -0.022 (less death)
 efficiency_ratio |    -0.0012 | -0.005 (less death)
      adverse_mom |    -0.0025 | +0.009 (more death)
        vol_accel |    -0.0029 | -0.007 (less death)
```
## Feature-GROUP importance (shuffle whole group, OOS AUC drop)
```
      VOL (realized_vol+vol_accel+range_pts  ) | AUC drop 0.0481
     TIME (time_of_day                       ) | AUC drop 0.0210
    TREND (efficiency_ratio                  ) | AUC drop 0.0008
  ADVERSE (adverse_drift+adverse_mom         ) | AUC drop -0.0020
```

## Read
Death prediction is CONDITIONAL. The top group/feature is what to READ live to decide hold-vs-cut.
If ADVERSE/TREND dominate -> a trade dies when the market trends against it (the regime read);
if VOL dominates -> death is about volatility magnitude. Either way this is the causal gauge the
fixed clock lacked. Next: turn the top features into the live two-gauge meter.
