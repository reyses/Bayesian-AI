# V2 features × time-of-day × regime — 2026-05-03 17:41 UTC

**Concepts:** ['price_sigma_w', 'bar_range', 'vol_mean_w', 'vol_velocity_w', 'price_velocity_w', 'body', 'price_velocity_1b', 'swing_noise_w']

**TFs:** ['5s', '1m', '5m', '15m', '1h']

**Tracking quantile:** Q4 of 5

**Buckets (PT):** ['pre_market', 'asia_close', 'eu_open', 'us_pre', 'us_open', 'us_morning', 'us_lunch', 'us_pm', 'after_close']

## Top 30 (concept, TF, regime) by best-bucket forward return

          concept  tf   regime_2d  n_target  n_buckets_with_data best_bucket  best_bucket_fwd worst_bucket  worst_bucket_fwd  inversion
       vol_mean_w  1h   UP_SMOOTH      1441                    5  us_morning       316.756944   pre_market          3.031178      False
price_velocity_1b  1h   UP_SMOOTH      1446                    9      us_pre       238.427711        us_pm          0.550926      False
             body  1h   UP_SMOOTH      1446                    9      us_pre       238.427711        us_pm         -1.502193       True
        bar_range  1h   UP_SMOOTH      1439                    9      us_pre       217.242958   pre_market         -2.187500       True
 price_velocity_w 15m   UP_SMOOTH      1434                    9      us_pre       196.247222   pre_market          0.027344      False
    swing_noise_w 15m   UP_SMOOTH      1441                    8      us_pre       184.025000   pre_market          5.192953      False
    price_sigma_w 15m   UP_SMOOTH      1432                    9      us_pre       164.715517   pre_market          0.768229      False
    price_sigma_w  5s   UP_SMOOTH      1431                    9      us_pre       156.525641   pre_market          2.662621      False
price_velocity_1b  1h DOWN_SMOOTH       948                    9     us_open      -151.113095   pre_market         -1.370098      False
             body  1h DOWN_SMOOTH       936                    9     us_open      -151.113095   pre_market         -0.164062      False
    swing_noise_w  5s   UP_SMOOTH      1466                    9      us_pre       144.915929   asia_close         -0.472222      False
    swing_noise_w  1h   UP_SMOOTH      1477                    9      us_pre       138.166667   pre_market          1.643077      False
 price_velocity_w 15m DOWN_SMOOTH       937                    7     us_open      -136.715278      eu_open         -6.047297      False
    price_sigma_w  5m   UP_SMOOTH      1430                    9    us_lunch       136.138889   pre_market         -1.671717       True
 price_velocity_w  5s   UP_SMOOTH      1490                    9      us_pre       129.387597   pre_market          2.319672      False
       vol_mean_w  1h DOWN_SMOOTH       936                    4    us_lunch      -129.041667   pre_market          0.618827      False
    swing_noise_w  1m   UP_SMOOTH      1435                    9    us_lunch       128.816667   pre_market         -4.849462       True
       vol_mean_w 15m DOWN_CHOPPY       600                    5 after_close      -127.537037   us_morning        -18.042708      False
   vol_velocity_w  1h DOWN_SMOOTH       936                    5 after_close      -125.303571      us_open        -18.099462      False
   vol_velocity_w  1h DOWN_CHOPPY       600                    5 after_close      -123.708333        us_pm         -4.660256      False
 price_velocity_w  5m   UP_SMOOTH      1445                    9      us_pre       123.444643   asia_close         -0.763158      False
 price_velocity_w  1h DOWN_SMOOTH       936                    6      us_pre      -121.267857      eu_open         -1.946381      False
    price_sigma_w  1m   UP_SMOOTH      1430                    9    us_lunch       121.191176   pre_market         -8.457865       True
    swing_noise_w  1h   UP_CHOPPY       816                    9  asia_close       120.006944   pre_market         -8.752083       True
             body 15m DOWN_SMOOTH       939                    9      us_pre      -116.600000   pre_market         -3.810897      False
 price_velocity_w  5m DOWN_SMOOTH       937                    9      us_pre      -116.058824   pre_market         -3.355128      False
price_velocity_1b  5m   UP_SMOOTH      1443                    9      us_pre       115.993534   pre_market         -0.256944      False
price_velocity_1b 15m DOWN_SMOOTH       944                    9      us_pre      -113.974138   pre_market         -3.685938      False
price_velocity_1b 15m   UP_SMOOTH      1453                    9      us_pre       113.771825   pre_market         -5.964052       True
       vol_mean_w  5s DOWN_SMOOTH       933                    5 after_close      -113.372549        us_pm        -26.280220      False

## TOD inversions (same Q4-cell, opposite signs in different hours)

          concept  tf   regime_2d  n_target  n_buckets_with_data best_bucket  best_bucket_fwd worst_bucket  worst_bucket_fwd  inversion
             body  1h   UP_SMOOTH      1446                    9      us_pre       238.427711        us_pm         -1.502193       True
        bar_range  1h   UP_SMOOTH      1439                    9      us_pre       217.242958   pre_market         -2.187500       True
    price_sigma_w  5m   UP_SMOOTH      1430                    9    us_lunch       136.138889   pre_market         -1.671717       True
    swing_noise_w  1m   UP_SMOOTH      1435                    9    us_lunch       128.816667   pre_market         -4.849462       True
    price_sigma_w  1m   UP_SMOOTH      1430                    9    us_lunch       121.191176   pre_market         -8.457865       True
    swing_noise_w  1h   UP_CHOPPY       816                    9  asia_close       120.006944   pre_market         -8.752083       True
price_velocity_1b 15m   UP_SMOOTH      1453                    9      us_pre       113.771825   pre_market         -5.964052       True
        bar_range  5m   UP_SMOOTH      1434                    9      us_pre       111.071809   pre_market         -9.246795       True
        bar_range  1m   UP_SMOOTH      1435                    9      us_pre       110.656542   asia_close         -1.509740       True
    swing_noise_w  5m   UP_SMOOTH      1430                    9    us_lunch       107.023196   pre_market         -1.845109       True
   vol_velocity_w  5m DOWN_SMOOTH       935                    9      us_pre      -103.750000   pre_market          6.087209       True
 price_velocity_w  1m   UP_SMOOTH      1443                    9      us_pre       103.402174   pre_market         -3.104610       True
        bar_range 15m   UP_SMOOTH      1433                    9    us_lunch        99.072072   pre_market         -2.127976       True
             body 15m   UP_SMOOTH      1436                    9      us_pre        98.210744   pre_market         -5.976667       True
    price_sigma_w 15m   UP_CHOPPY       786                    8  asia_close        59.705556   pre_market         -8.666667       True
    price_sigma_w  1h   UP_CHOPPY       792                    9  asia_close        59.183333     us_lunch         -3.783333       True
   vol_velocity_w 15m   UP_CHOPPY       786                    6  asia_close        58.490741      eu_open        -11.006944       True
    price_sigma_w 15m DOWN_CHOPPY       599                    7 after_close       -51.293388   us_morning          2.476950       True
price_velocity_1b  5s   UP_CHOPPY       817                    9     us_open        50.612981   pre_market         -2.814159       True
    swing_noise_w  1h DOWN_CHOPPY       600                    9       us_pm       -48.375000  after_close          3.664062       True
    swing_noise_w 15m DOWN_CHOPPY       652                    7      us_pre       -44.891667   us_morning         15.076190       True
   vol_velocity_w  5m DOWN_CHOPPY       598                    9       us_pm       -38.322368   asia_close          4.075000       True
             body  1h   UP_CHOPPY       792                    7     us_open        35.788194   asia_close         -1.529762       True
price_velocity_1b  1h   UP_CHOPPY       792                    7     us_open        35.788194   asia_close         -3.083333       True
   vol_velocity_w 15m FLAT_CHOPPY      3664                    9  asia_close       -35.679825   us_morning          4.083333       True
    price_sigma_w  1h FLAT_CHOPPY      3672                    9      us_pre        35.350490        us_pm         -2.494141       True
             body  5m   UP_CHOPPY       795                    9  asia_close        35.141304   pre_market         -2.154762       True
price_velocity_1b  5m DOWN_CHOPPY       606                    9       us_pm       -34.906863   pre_market          1.004902       True
 price_velocity_w  1h FLAT_CHOPPY      3652                    9  us_morning       -34.852623   pre_market          1.568859       True
             body  5m DOWN_CHOPPY       613                    9       us_pm       -34.745000   pre_market          1.060185       True
        bar_range  1m FLAT_SMOOTH      2072                    9    us_lunch        31.575641      eu_open         -1.201923       True
    price_sigma_w  5m FLAT_SMOOTH      1977                    9    us_lunch        30.777500   asia_close         -3.002049       True
    price_sigma_w 15m FLAT_SMOOTH      1973                    8    us_lunch        30.491228  after_close         -2.548600       True
       vol_mean_w  5s FLAT_CHOPPY      3664                    7 after_close       -28.626562        us_pm          3.592085       True
             body  1m DOWN_CHOPPY       598                    9       us_pm       -27.805556   pre_market          1.428571       True
    price_sigma_w  1m FLAT_CHOPPY      3663                    9  pre_market        21.823370      eu_open         -1.347345       True
 price_velocity_w 15m FLAT_CHOPPY      3663                    9  asia_close       -20.543165  after_close          1.635417       True
       vol_mean_w  5m FLAT_SMOOTH      1977                    5    us_lunch        20.121233        us_pm         -4.118519       True
       vol_mean_w  1m FLAT_SMOOTH      1978                    6    us_lunch        19.698204        us_pm         -4.757862       True
       vol_mean_w  5s FLAT_SMOOTH      1981                    6    us_lunch        18.429825       us_pre         -2.787736       True
   vol_velocity_w  5m FLAT_CHOPPY      3668                    9 after_close       -17.244490        us_pm          1.953767       True
        bar_range  5s FLAT_CHOPPY      3706                    9  asia_close       -16.439236        us_pm          3.029255       True
price_velocity_1b 15m FLAT_CHOPPY      3688                    9      us_pre        14.237939  after_close         -3.734862       True
       vol_mean_w 15m FLAT_CHOPPY      3663                    5     us_open       -13.862710   us_morning          2.398399       True
 price_velocity_w  5s FLAT_CHOPPY      3754                    9  us_morning         9.224120      eu_open         -1.336356       True
