# V2 features lead-lag (Step #3) - 2026-05-03 17:46 UTC

**Concepts:** ['price_sigma_w', 'bar_range', 'vol_mean_w', 'vol_velocity_w', 'price_velocity_w', 'body', 'price_velocity_1b', 'swing_noise_w', 'z_se_w', 'reversion_prob_w', 'hurst_w']

**TFs:** ['5s', '1m', '5m', '15m', '1h']

**Shifts (base bars):** [-12, -8, -4, -2, -1, 0, 1, 2, 4, 8, 12]

**Window:** 12 base bars

## Role distribution

- **lagging**: 296 (89.7%)
- **leading**: 29 (8.8%)
- **contemporaneous**: 5 (1.5%)

## Top 30 (concept, TF, regime) by |peak_corr|

          concept  tf   regime_2d  peak_shift  peak_corr  contemp_corr    role  lift_vs_contemp
 price_velocity_w  5m   UP_SMOOTH         -12   0.901196      0.114590 lagging         0.786606
 price_velocity_w  5m FLAT_CHOPPY         -12   0.866833     -0.013103 lagging         0.853730
 price_velocity_w  5m DOWN_SMOOTH         -12   0.860191     -0.079741 lagging         0.780450
 price_velocity_w  5m FLAT_SMOOTH         -12   0.851323     -0.065942 lagging         0.785382
 price_velocity_w  5m DOWN_CHOPPY         -12   0.839983     -0.032909 lagging         0.807074
 price_velocity_w  5m   UP_CHOPPY         -12   0.751567     -0.231076 lagging         0.520491
    price_sigma_w  5m   UP_SMOOTH         -12   0.666258      0.207660 lagging         0.458599
    swing_noise_w  1m   UP_SMOOTH         -12   0.599871      0.246158 lagging         0.353713
price_velocity_1b 15m   UP_SMOOTH         -12   0.566216      0.073794 lagging         0.492422
 price_velocity_w  1m   UP_SMOOTH          -8   0.562769      0.105063 lagging         0.457707
             body 15m   UP_SMOOTH         -12   0.561743      0.077266 lagging         0.484477
           z_se_w 15m DOWN_CHOPPY         -12   0.549569     -0.001065 lagging         0.548504
    swing_noise_w  5m   UP_SMOOTH         -12   0.546795      0.226228 lagging         0.320567
           z_se_w 15m FLAT_SMOOTH         -12   0.535820     -0.019563 lagging         0.516257
        bar_range 15m   UP_SMOOTH         -12   0.532610      0.224795 lagging         0.307816
           z_se_w 15m DOWN_SMOOTH         -12   0.528963     -0.053270 lagging         0.475692
 price_velocity_w 15m   UP_SMOOTH         -12   0.521500      0.082116 lagging         0.439384
           z_se_w 15m FLAT_CHOPPY         -12   0.513425     -0.023069 lagging         0.490357
    price_sigma_w  1m   UP_SMOOTH          -8   0.512633      0.230102 lagging         0.282531
price_velocity_1b 15m DOWN_SMOOTH          -8   0.508875     -0.037890 lagging         0.470985
             body 15m DOWN_SMOOTH          -8   0.508487     -0.047330 lagging         0.461157
 price_velocity_w  1m FLAT_CHOPPY          -4   0.504801      0.013472 lagging         0.491329
 price_velocity_w  1m DOWN_SMOOTH          -8   0.501531     -0.017130 lagging         0.484402
 price_velocity_w  1m   UP_CHOPPY          -4   0.499846     -0.191769 lagging         0.308076
price_velocity_1b 15m FLAT_CHOPPY          -8   0.498655     -0.000136 lagging         0.498519
    price_sigma_w  5m DOWN_SMOOTH         -12  -0.497633     -0.056813 lagging         0.440820
             body 15m FLAT_CHOPPY          -8   0.497310     -0.000889 lagging         0.496421
 price_velocity_w  1m DOWN_CHOPPY          -8   0.495951     -0.036463 lagging         0.459488
       vol_mean_w  1m DOWN_SMOOTH         -12  -0.488076     -0.141504 lagging         0.346572
        bar_range  5m   UP_SMOOTH          -8   0.486551      0.279906 lagging         0.206645

## Genuinely-leading cells (s>0)

          concept  tf   regime_2d  peak_shift  peak_corr  contemp_corr    role  lift_vs_contemp
        bar_range  1h   UP_CHOPPY          12   0.191605      0.109475 leading         0.082129
       vol_mean_w  1h DOWN_SMOOTH          12   0.103873      0.062304 leading         0.041569
   vol_velocity_w  5m FLAT_SMOOTH          12  -0.094374      0.048391 leading         0.045983
 price_velocity_w  1h DOWN_SMOOTH          12  -0.071211     -0.025555 leading         0.045657
          hurst_w  1h   UP_SMOOTH          12   0.071063      0.023985 leading         0.047077
price_velocity_1b  5s DOWN_CHOPPY           8   0.068061      0.003541 leading         0.064520
          hurst_w 15m   UP_SMOOTH          12  -0.065808     -0.013658 leading         0.052150
       vol_mean_w  1h FLAT_SMOOTH          12   0.065623      0.046729 leading         0.018894
             body  5s DOWN_CHOPPY           8   0.064354      0.004067 leading         0.060288
          hurst_w  5m   UP_CHOPPY           4  -0.064241     -0.050716 leading         0.013525
          hurst_w  1m DOWN_SMOOTH          12   0.064017     -0.017157 leading         0.046860
          hurst_w 15m DOWN_CHOPPY          12   0.059555      0.042326 leading         0.017230
