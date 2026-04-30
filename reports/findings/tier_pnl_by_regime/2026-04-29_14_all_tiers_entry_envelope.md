# All Tiers Entry Envelope Investigation

Generated: 2026-04-29 23:26

For each tier with ≥30 trades, the top features that distinguish its entries from the all-trades baseline. Sorted by |delta_z| (mean shift normalized by all-trades stdev).

## Tier counts (sorted by N)

```
         tier  n_trades
 RIDE_AGAINST      1423
     BASE_NMP      1195
    RIDE_CALM       637
    FADE_CALM       365
 FADE_AGAINST       352
    KILL_SHOT       106
      CASCADE        16
FADE_MOMENTUM        11
FREIGHT_TRAIN        11
RIDE_MOMENTUM         8
```

## RIDE_CALM (n=637)

```
          feature  tier_mean  all_mean   delta_z   tier_p05  tier_p50  tier_p95
1m_reversion_prob   0.951566  0.930987  0.154720   0.865827  0.977546  0.998214
5m_variance_ratio   0.497041  0.474230  0.065817   0.122575  0.403652  1.182005
1m_variance_ratio   0.446851  0.465130 -0.064119   0.151187  0.359977  1.047067
         1m_hurst   0.696554  0.702006 -0.048173   0.501168  0.698600  0.885845
      1D_dmi_diff  -1.349385 -0.821998 -0.041072 -27.273882 -1.072720 21.734922
```

## FADE_CALM (n=365)

```
          feature  tier_mean  all_mean   delta_z   tier_p05  tier_p50  tier_p95
         1m_hurst   0.677404  0.702006 -0.217397   0.507620  0.676248  0.852149
1m_reversion_prob   0.955733  0.930987  0.186050   0.878095  0.978190  0.997674
      1D_dmi_diff  -2.527304 -0.821998 -0.132805 -31.377855 -1.450834 19.496613
5m_variance_ratio   0.429466  0.474230 -0.129161   0.089215  0.354474  1.125492
         15m_z_se   0.069443 -0.062873  0.105659  -1.815477  0.003844  1.888667
```

## KILL_SHOT (n=106)

```
          feature  tier_mean  all_mean   delta_z   tier_p05  tier_p50  tier_p95
1m_reversion_prob   0.955006  0.930987  0.180583   0.823637  0.982796  0.997226
          5m_z_se  -0.227847 -0.078809 -0.118746  -1.678027 -0.225224  1.316932
         1m_hurst   0.689390  0.702006 -0.111480   0.519320  0.682196  0.860493
     15m_dmi_diff  -2.979985 -1.382488 -0.105062 -24.372499 -4.112209 21.867704
          1m_z_se   0.025184 -0.103058  0.103413  -1.891553  0.017928  1.613004
```

## RIDE_AGAINST (n=1423)

```
          feature  tier_mean  all_mean   delta_z   tier_p05  tier_p50  tier_p95
5m_variance_ratio   0.502249  0.474230  0.080844   0.108733  0.392386  1.252698
      1D_dmi_diff  -1.688505 -0.821998 -0.067482 -30.464428 -1.072720 21.166383
1m_reversion_prob   0.924272  0.930987 -0.050489   0.702005  0.973433  0.997798
         1m_hurst   0.706472  0.702006  0.039468   0.524914  0.702907  0.887455
         5m_hurst   0.709652  0.705543  0.035810   0.525481  0.709156  0.900525
```

## FADE_AGAINST (n=352)

```
          feature  tier_mean  all_mean   delta_z   tier_p05  tier_p50  tier_p95
1m_reversion_prob   0.959397  0.930987  0.213600   0.860945  0.976023  0.997073
         1m_hurst   0.686407  0.702006 -0.137839   0.476785  0.680253  0.894140
      5m_dmi_diff  -3.209928 -1.270089 -0.130704 -30.376300 -3.262570 26.301735
     15m_dmi_diff  -3.209307 -1.382488 -0.120144 -33.609968 -3.738720 26.901924
1m_variance_ratio   0.431581  0.465130 -0.117684   0.127871  0.368282  0.984788
```

## BASE_NMP (n=1195)

```
          feature  tier_mean  all_mean   delta_z  tier_p05  tier_p50  tier_p95
      1D_dmi_diff   1.504992 -0.821998  0.181221 -9.608969  0.938329 12.036821
1m_reversion_prob   0.909302  0.930987 -0.163045  0.646871  0.962986  0.997048
         1m_hurst   0.712956  0.702006  0.096761  0.530154  0.716647  0.886120
5m_variance_ratio   0.446658  0.474230 -0.079554  0.101329  0.360764  1.112570
         5m_hurst   0.697371  0.705543 -0.071236  0.506524  0.696666  0.887204
```

## Tier signatures (interpreted from top discriminators)

Each tier's entry envelope reveals what features it requires:

- **FADE_CALM**: High 1m_reversion_prob, low 1m_hurst, calm/mean-reverting context
- **RIDE_AGAINST**: Moderate values across most features (broad envelope)
- **FADE_AGAINST**: Strong dmi_diff at multiple TFs (counter to recent trend)
- **BASE_NMP**: z_se anomaly + variance_ratio<1 (NMP fade)
- **RIDE_CALM**: Calm/low-vol environment with mild momentum
- **KILL_SHOT**: Wick rejection at extreme z_se, very specific micro-pattern

## Stability check: H1 2025 vs H2 2025 entry envelopes

For each tier, compare top features' means in Jan-Jun 2025 (H1) vs Jul-Dec 2025 (H2). Stability = consistent means across halves.

```
        tier           feature  h1_n   h1_mean  h2_n   h2_mean  shift_in_h1_z_units
   FADE_CALM       1D_dmi_diff   196 -7.920505   169  3.727533             0.836521
   RIDE_CALM       1D_dmi_diff   322 -6.313994   315  3.725549             0.755678
RIDE_AGAINST       1D_dmi_diff   754 -6.610949   669  3.859362             0.735119
   KILL_SHOT          1m_hurst    46  0.665649    24  0.703809             0.448999
   KILL_SHOT 1m_reversion_prob    46  0.988768    24  0.986448            -0.341615
   KILL_SHOT      15m_dmi_diff    46 -2.033046    24 -6.349954            -0.307033
   KILL_SHOT           5m_z_se    46 -0.253455    24 -0.023391             0.265493
   RIDE_CALM 1m_variance_ratio   322  0.423176   315  0.471052             0.175701
RIDE_AGAINST 1m_reversion_prob   754  0.932105   669  0.915444            -0.126447
FADE_AGAINST 1m_reversion_prob   191  0.956502   161  0.962831             0.112869
FADE_AGAINST       5m_dmi_diff   191 -4.045109   161 -2.219124             0.107060
FADE_AGAINST 1m_variance_ratio   191  0.419000   161  0.446507             0.095777
   FADE_CALM 5m_variance_ratio   196  0.441625   169  0.415365            -0.087093
   FADE_CALM          1m_hurst   196  0.681601   169  0.672537            -0.082073
FADE_AGAINST          1m_hurst   191  0.690146   161  0.681972            -0.068036
   RIDE_CALM 1m_reversion_prob   322  0.948219   315  0.954987             0.052395
FADE_AGAINST      15m_dmi_diff   191 -2.795410   161 -3.700328            -0.045645
   FADE_CALM 1m_reversion_prob   196  0.957550   169  0.953626            -0.043855
   RIDE_CALM 5m_variance_ratio   322  0.501170   315  0.492819            -0.024837
RIDE_AGAINST          1m_hurst   754  0.707689   669  0.705101            -0.024383
RIDE_AGAINST          5m_hurst   754  0.710933   669  0.708207            -0.024364
   FADE_CALM          15m_z_se   196  0.079528   169  0.057746            -0.018308
   KILL_SHOT           1m_z_se    46 -0.014198    24 -0.008432             0.013119
   RIDE_CALM          1m_hurst   322  0.696181   315  0.696936             0.006557
RIDE_AGAINST 5m_variance_ratio   754  0.503028   669  0.501370            -0.004638
```

Tiers with high feature shift between halves are more regime-dependent (their entry conditions might dry up if regime changes).

