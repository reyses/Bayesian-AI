**DEPRECATED — LOOKAHEAD ARTIFACT, DO NOT QUOTE**

# Cross-TF anchor-pair drill (Layer Cross-TF 2) - 2026-05-03 23:11 UTC

Anchor pairs: [('price_velocity_w', 'price_sigma_w'), ('bar_range', 'body'), ('price_velocity_1b', 'bar_range'), ('price_velocity_w', 'swing_noise_w'), ('price_velocity_w', 'SE_low_w'), ('price_velocity_1b', 'reversion_prob_w'), ('body', 'reversion_prob_w'), ('price_velocity_w', 'vol_mean_w')]

## Diagonal vs off-diagonal — top lifts

                X                Y   regime_2d max_diag_tf  max_diag_r max_offdiag_tfX max_offdiag_tfY  max_offdiag_r  offdiag_lift  sign_match
 price_velocity_w    swing_noise_w         ALL          4h   -0.088487              1D              1h      -0.390009      0.301522        True
 price_velocity_w    swing_noise_w DOWN_CHOPPY          1h   -0.399179              1h             15m      -0.594025      0.194847        True
 price_velocity_w    price_sigma_w         ALL          1D   -0.155569              1D              1h      -0.311214      0.155644        True
 price_velocity_w    swing_noise_w FLAT_CHOPPY          1h   -0.219767              1h             15m      -0.365427      0.145661        True
 price_velocity_w    swing_noise_w FLAT_SMOOTH          4h    0.442169              1D              4h       0.567278      0.125109        True
 price_velocity_w         SE_low_w         ALL          1D   -0.189339              1D              1h      -0.293371      0.104032        True
 price_velocity_w       vol_mean_w         ALL          1D   -0.310380              1D              4h      -0.401480      0.091100        True
 price_velocity_w    swing_noise_w DOWN_SMOOTH          1h   -0.751025              4h              1h      -0.831309      0.080284        True
 price_velocity_w         SE_low_w FLAT_CHOPPY         15m   -0.245420              1h             15m      -0.319325      0.073905        True
 price_velocity_w    price_sigma_w FLAT_CHOPPY         15m   -0.259285              1h             15m      -0.331158      0.071873        True
 price_velocity_w    swing_noise_w   UP_CHOPPY          4h   -0.796358              1D              4h      -0.845721      0.049363        True
 price_velocity_w       vol_mean_w FLAT_CHOPPY          1D   -0.434340              1D              4h      -0.478572      0.044232        True
 price_velocity_w       vol_mean_w DOWN_SMOOTH          1h   -0.682212              1D              4h      -0.703068      0.020857        True
 price_velocity_w    swing_noise_w   UP_SMOOTH          1h    0.766465             15m              5m       0.778948      0.012483        True
 price_velocity_w       vol_mean_w   UP_CHOPPY          1D   -0.804161              1D              4h      -0.805123      0.000962        True
 price_velocity_w         SE_low_w   UP_CHOPPY          1D   -0.765089              1D              4h      -0.765724      0.000636        True
 price_velocity_w         SE_low_w DOWN_SMOOTH         15m   -0.734075              4h              1h      -0.727325     -0.006750        True
 price_velocity_w    price_sigma_w FLAT_SMOOTH          1D    0.574536              4h              1D       0.555695     -0.018840        True
 price_velocity_w    price_sigma_w   UP_CHOPPY          1D   -0.877124              1D              4h      -0.853995     -0.023129        True
 price_velocity_w       vol_mean_w FLAT_SMOOTH          1D    0.221573              4h              1D       0.184542     -0.037030        True
 price_velocity_w       vol_mean_w   UP_SMOOTH          1h    0.548084              1h              1D       0.483379     -0.064705        True
price_velocity_1b        bar_range         ALL          1D    0.200802              1D              4h      -0.127724     -0.073078       False
price_velocity_1b reversion_prob_w FLAT_SMOOTH          1D   -0.217321              1D              4h       0.141924     -0.075398       False
 price_velocity_w    price_sigma_w DOWN_SMOOTH          1h   -0.837553              4h              1h      -0.756617     -0.080937        True
             body reversion_prob_w         ALL          1D   -0.142843             15s              1m       0.038232     -0.104611       False
price_velocity_1b        bar_range   UP_CHOPPY          4h    0.559382              1D              1h      -0.447770     -0.111612       False
price_velocity_1b reversion_prob_w         ALL          1D   -0.161472              1D              4h       0.040541     -0.120932       False
        bar_range             body         ALL          1D    0.220560              4h              1D      -0.098561     -0.121999       False
        bar_range             body   UP_CHOPPY          4h    0.558397              1h              4h       0.425322     -0.133075        True
 price_velocity_w         SE_low_w FLAT_SMOOTH          4h    0.473111              1D              4h       0.339808     -0.133303        True
price_velocity_1b        bar_range FLAT_CHOPPY          4h   -0.345074              4h              1h      -0.198323     -0.146752        True
        bar_range             body FLAT_CHOPPY          4h   -0.348439              1h              4h      -0.197991     -0.150448        True
price_velocity_1b reversion_prob_w FLAT_CHOPPY          4h    0.228133              1D              4h       0.052324     -0.175809        True
 price_velocity_w       vol_mean_w DOWN_CHOPPY          1h   -0.568439              4h              1h      -0.362420     -0.206020        True
 price_velocity_w         SE_low_w   UP_SMOOTH          1h    0.801009              1h              4h       0.575141     -0.225868        True
             body reversion_prob_w FLAT_SMOOTH          1D   -0.300620             15s              1m       0.057269     -0.243351       False
             body reversion_prob_w FLAT_CHOPPY          4h    0.313036              4h              1D       0.060920     -0.252116        True
 price_velocity_w         SE_low_w DOWN_CHOPPY          1h   -0.605115             15m              5m      -0.340529     -0.264586        True
price_velocity_1b        bar_range DOWN_CHOPPY          4h   -0.578956              4h              1h      -0.296178     -0.282779        True
        bar_range             body DOWN_CHOPPY          4h   -0.578659              1h              4h      -0.294946     -0.283713        True
             body reversion_prob_w   UP_CHOPPY          4h   -0.398237              1D              1h       0.110940     -0.287297       False
             body reversion_prob_w   UP_SMOOTH          1h   -0.401644              1h              1D       0.112313     -0.289331       False
 price_velocity_w    price_sigma_w   UP_SMOOTH          1h    0.887296              1h              1D       0.597146     -0.290150        True
        bar_range             body FLAT_SMOOTH          1D    0.480658              1h              4h      -0.184534     -0.296124       False
price_velocity_1b        bar_range FLAT_SMOOTH          1D    0.467479              1D              1h       0.166055     -0.301424        True
price_velocity_1b        bar_range DOWN_SMOOTH          4h   -0.737708              4h              1h      -0.435407     -0.302301        True
price_velocity_1b reversion_prob_w   UP_CHOPPY          4h   -0.397829              1D              1h       0.093153     -0.304676       False
        bar_range             body DOWN_SMOOTH          4h   -0.752701              1h              4h      -0.445205     -0.307496        True
price_velocity_1b reversion_prob_w   UP_SMOOTH          1h   -0.421956              1h              1D       0.113781     -0.308175       False
 price_velocity_w    price_sigma_w DOWN_CHOPPY          1h   -0.759174              4h              1D      -0.415140     -0.344034        True

Off-diagonal exceeds diagonal: 16/56 (28.6%); strong (>0.05): 10 (17.9%).
