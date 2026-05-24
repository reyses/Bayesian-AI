**DEPRECATED — LOOKAHEAD ARTIFACT, DO NOT QUOTE**

# FADE_CALM Dropout Investigation

Generated: 2026-04-29 23:23

## Timeline

- FADE_CALM total trades: 365
- First trade: 2025-01-02 04:10:00+00:00
- Last trade: 2025-12-19 14:29:55+00:00
- Days since last firing: 52

## Per-month firing frequency

```
     ym  n_trades
2025-01        37
2025-02        33
2025-03        26
2025-04        49
2025-05        32
2025-06        19
2025-07        34
2025-08        30
2025-09        18
2025-10        35
2025-11        33
2025-12        19
```

## Top discriminating features (FADE_CALM trades vs all trades)

Sorted by |delta_z| (mean difference normalized by all-trades stdev):

```
          feature   fc_mean  all_mean    fc_std   all_std     fc_p25   fc_p75     ratio   delta_z
         1m_hurst  0.677404  0.702006  0.108688  0.113164   0.592867 0.754584  0.964955 -0.217397
1m_reversion_prob  0.955733  0.930987  0.099371  0.133003   0.956042 0.990349  1.026580  0.186050
      1D_dmi_diff -2.527304 -0.821998 14.468439 12.840634  -8.706268 7.273819  3.074586 -0.132805
5m_variance_ratio  0.429466  0.474230  0.306072  0.346575   0.204330 0.565265  0.905607 -0.129161
         15m_z_se  0.069443 -0.062873  1.165518  1.252289  -0.725352 0.925381 -1.104497  0.105659
      5m_dmi_diff -0.034512 -1.270089 13.665231 14.841492  -8.782201 8.780684  0.027173  0.083252
      1m_dmi_diff -0.157593 -1.282613 13.772340 14.841074 -10.218758 8.696185  0.122869  0.075804
1m_variance_ratio  0.445457  0.465130  0.295883  0.285074   0.238702 0.557462  0.957706 -0.069008
     15m_dmi_diff -0.601642 -1.382488 13.633248 15.205284  -8.992867 8.366247  0.435188  0.051354
          1m_z_se -0.044902 -0.103058  1.027458  1.240095  -0.720565 0.704088  0.435698  0.046896
      1h_dmi_diff -2.179996 -1.930554 15.214108 15.259112 -12.108395 8.623948  1.129208 -0.016347
          5m_z_se -0.061312 -0.078809  1.233329  1.255101  -0.938219 0.891424  0.777984  0.013941
         5m_hurst  0.707089  0.705543  0.117367  0.114724   0.629259 0.788299  1.002190  0.013470
```

## Feature shift, FADE_CALM period vs post-dropout

```
          feature   fc_mean  post_mean    fc_std  post_std  shift_in_z_units
1m_reversion_prob  0.955733   0.908774  0.099371  0.149955         -0.472555
         1m_hurst  0.677404   0.712878  0.108688  0.110757          0.326379
      1D_dmi_diff -2.527304   1.581011 14.468439  7.445092          0.283950
         15m_z_se  0.069443  -0.093246  1.165518  1.248430         -0.139585
1m_variance_ratio  0.445457   0.480353  0.295883  0.264968          0.117936
      5m_dmi_diff -0.034512  -1.387338 13.665231 14.106169         -0.098998
5m_variance_ratio  0.429466   0.448145  0.306072  0.325774          0.061030
      1m_dmi_diff -0.157593  -0.889855 13.772340 14.092000         -0.053169
     15m_dmi_diff -0.601642  -1.318060 13.633248 14.804383         -0.052549
          1m_z_se -0.044902  -0.065231  1.027458  1.418547         -0.019786
```

## Inferred FADE_CALM entry envelope

```
          feature     fc_p05  fc_median    fc_p95     fc_min    fc_max
         1m_hurst   0.507620   0.676248  0.852149   0.411554  1.000000
1m_reversion_prob   0.878095   0.978190  0.997674   0.000000  0.999900
      1D_dmi_diff -31.377855  -1.450834 19.496613 -42.890160 26.780048
5m_variance_ratio   0.089215   0.354474  1.125492   0.028240  1.749757
         15m_z_se  -1.815477   0.003844  1.888667  -3.655832  3.004142
      5m_dmi_diff -22.183825   0.206700 23.426542 -43.545450 44.061650
      1m_dmi_diff -21.787038  -0.478116 23.813542 -51.238205 36.862710
1m_variance_ratio   0.140770   0.373629  1.039640   0.058593  1.943750
     15m_dmi_diff -22.419575  -1.678971 22.265752 -43.259940 41.178978
          1m_z_se  -1.775912  -0.061691  1.515228  -3.303743  2.893292
```

## Hypotheses

- If `shift_in_z_units` is large (>1) for some features, the post-dropout 
  market conditions DON'T match FADE_CALM's typical entry envelope.
- If shift is small (<0.5), the conditions ARE there but the tier still 
  doesn't fire — meaning some other condition (timing? regime?) blocks it.

## Recommended next step

Read `nn_v2/nightmare_blended.py` (or wherever FADE_CALM is defined) and 
compare the EXACT entry conditions to the post-dropout feature values 
computed here. Probably one or two specific thresholds aren't being met 
in the new regime.
