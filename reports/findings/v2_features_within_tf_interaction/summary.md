# V2 features within-TF interaction (Layer D1) - 2026-05-03 19:18 UTC

**Concepts:** 23 v2 features per TF

**TFs:** ['5s', '15s', '1m', '5m', '15m', '1h', '4h', '1D']

**Split:** IS

## Hypothesis test summary

               hypothesis predicted  pearson_min  pearson_max  pearson_mean  n_pos_tf  n_neg_tf  n_zero_tf  sign_flip_across_tfs  agree_count  n_tfs                                         interpretation
    velocity_needs_volume         +     0.401643     0.518206      0.467528         8         0          0                 False            8      8                       directional moves require volume
 velocity_kills_variation         ?     0.764109     0.890125      0.848411         8         0          0                 False            0      8        high velocity vs sigma — user predicts negative
            chop_vs_trend         -     0.039272     0.142204      0.076699         7         0          1                 False            0      8 Hurst trending vs reversion mean-revert (definitional)
      range_tracks_volume         +     0.321211     0.631594      0.533891         8         0          0                 False            8      8                                 busy bars are big bars
         body_is_velocity         +     0.909811     0.997267      0.953313         8         0          0                 False            8      8           body and bar-velocity are the same primitive
       vwap_vs_price_mean         +     0.998644     1.000000      0.999653         8         0          0                 False            8      8              VWAP and unweighted mean redundancy audit
        vol_mean_vs_sigma         +     0.761414     0.935648      0.855643         8         0          0                 False            8      8             are volume mean and volume sigma redundant
     swing_noise_vs_range         +     0.497571     0.743161      0.619095         8         0          0                 False            8      8                are swing_noise and bar_range redundant
              SE_symmetry         +     0.649502     0.966671      0.890351         8         0          0                 False            8      8              upper and lower standard errors symmetric
      z_extreme_wide_bars         +     0.140178     0.297840      0.197878         8         0          0                 False            8      8    extreme regression dislocation comes with wide bars
     z_extreme_high_sigma         +    -0.017802     0.034280      0.017033         0         0          8                 False            0      8                       extreme z = high local variation
        accel_vs_velocity         ?     0.114843     0.264607      0.190751         8         0          0                 False            0      8                 acceleration vs velocity within window
 vol_kinetic_drives_price         ?    -0.283139    -0.026385     -0.091843         0         5          3                 False            0      8                   do volume swings drive price swings?
abs_vol_velocity_vs_range         +    -0.037968     0.492054      0.349357         7         0          1                 False            7      8                      rapid volume change = bigger bars
 reversion_vs_swing_noise         ?    -0.098662     0.011322     -0.011306         0         1          7                 False            0      8               is reversion higher when path is noisier
      z_high_low_symmetry         +     0.300817     0.859303      0.715923         8         0          0                 False            8      8                  upper-band and lower-band z symmetric
     hurst_vs_swing_noise         -    -0.121316     0.209468      0.090796         6         1          1                  True            1      8                    trending market has less path noise
vol_accel_vs_vol_velocity         ?     0.295781     0.897601      0.569946         8         0          0                 False            0      8              volume acceleration leads volume velocity

## Top 30 redundant pairs across TFs

               c1             c2  mean_abs_corr  mean_corr  min_corr  max_corr  n_tfs  sign_flip
     price_mean_w         vwap_w       0.999661   0.999661  0.998707  1.000000      8      False
price_velocity_1b           body       0.953154   0.953154  0.909811  0.997267      8      False
        SE_high_w       SE_low_w       0.897937   0.897937  0.709152  0.967365      8      False
    price_sigma_w      SE_high_w       0.862572   0.862572  0.806557  0.885849      8      False
       vol_mean_w    vol_sigma_w       0.858236   0.858236  0.761426  0.935769      8      False
    price_sigma_w       SE_low_w       0.840377   0.840377  0.632624  0.893090      8      False
  vol_velocity_1b   vol_accel_1b       0.796117   0.796117  0.677806  0.917444      8      False
           z_se_w       z_high_w       0.783714   0.783714  0.527525  0.866144      8      False
           z_se_w        z_low_w       0.783095   0.783095  0.462234  0.869312      8      False
  vol_velocity_1b    vol_accel_w       0.754616   0.754616  0.547584  0.916490      8      False
    price_sigma_w  swing_noise_w       0.753490   0.753490  0.566463  0.839094      8      False
         SE_low_w  swing_noise_w       0.730354   0.730354  0.516804  0.810160      8      False
        SE_high_w  swing_noise_w       0.722504   0.722504  0.453742  0.818553      8      False
price_velocity_1b price_accel_1b       0.715152   0.715152  0.668104  0.810726      8      False
         z_high_w        z_low_w       0.712810   0.712810  0.278245  0.859303      8      False
   price_accel_1b           body       0.686056   0.686056  0.627185  0.821878      8      False
price_velocity_1b  price_accel_w       0.683358   0.683358  0.632025  0.730503      8      False
             body  price_accel_w       0.653192   0.653192  0.590626  0.699058      8      False
        bar_range  swing_noise_w       0.619095   0.619095  0.497571  0.743161      8      False
        bar_range      SE_high_w       0.615745   0.615745  0.408415  0.734000      8      False
        bar_range  price_sigma_w       0.610872   0.610872  0.401144  0.737010      8      False
        bar_range       SE_low_w       0.593004   0.593004  0.403729  0.748561      8      False
       vol_mean_w       SE_low_w       0.584206   0.584206  0.473269  0.659184      8      False
     vol_accel_1b    vol_accel_w       0.583263   0.583263  0.318133  0.848497      8      False
    price_sigma_w     vol_mean_w       0.574694   0.574694  0.477197  0.637702      8      False
       vol_mean_w  swing_noise_w       0.570450   0.570450  0.456272  0.699050      8      False
   vol_velocity_w    vol_accel_w       0.567333   0.567333  0.294490  0.897601      8      False
       vol_mean_w      SE_high_w       0.553317   0.553317  0.469656  0.601062      8      False
        bar_range     vol_mean_w       0.535436   0.535436  0.322563  0.631610      8      False
     price_mean_w  swing_noise_w       0.530043  -0.530043 -0.841436 -0.388793      8      False

## Pairs that flip sign by TF

               c1               c2  mean_abs_corr  mean_corr  min_corr  max_corr  n_tfs  sign_flip
   vol_velocity_w      vol_sigma_w       0.336119   0.331985 -0.016537  0.781441      8       True
     vol_accel_1b   vol_velocity_w       0.237489   0.210694 -0.078619  0.761029      8       True
      vol_accel_w      vol_sigma_w       0.220870   0.073724 -0.278210  0.664749      8       True
      vol_accel_w       vol_mean_w       0.203133   0.005326 -0.334386  0.566573      8       True
  vol_velocity_1b      vol_sigma_w       0.194105   0.139224 -0.158431  0.753342      8       True
  vol_velocity_1b       vol_mean_w       0.185052   0.042059 -0.238221  0.638681      8       True
   vol_velocity_w       vol_mean_w       0.169044   0.168083 -0.003844  0.646789      8       True
     vol_accel_1b      vol_sigma_w       0.148645   0.109237 -0.090263  0.612659      8       True
          hurst_w    swing_noise_w       0.121125   0.090796 -0.121316  0.209468      8       True
 price_velocity_w        SE_high_w       0.102985   0.039240 -0.254982  0.177654      8       True
     vol_accel_1b       vol_mean_w       0.102202   0.074277 -0.043602  0.521616      8       True
        SE_high_w          hurst_w       0.081238  -0.062917 -0.259727  0.039457      8       True
     price_mean_w          hurst_w       0.080224  -0.009798 -0.136356  0.277383      8       True
   price_accel_1b         z_high_w       0.079020  -0.060553 -0.118034  0.073868      8       True
           vwap_w          hurst_w       0.078997  -0.009854 -0.135967  0.272120      8       True
         SE_low_w          hurst_w       0.075812  -0.067670 -0.244113  0.029225      8       True
    price_accel_w        SE_high_w       0.074746   0.072689 -0.008229  0.291644      8       True
 price_velocity_w    price_sigma_w       0.073974  -0.008568 -0.161863  0.109781      8       True
price_velocity_1b        SE_high_w       0.073443   0.071083 -0.009440  0.255146      8       True
   price_accel_1b          z_low_w       0.067656  -0.066874 -0.113795  0.003128      8       True
             body reversion_prob_w       0.066039   0.026710 -0.157316  0.101383      8       True
    price_sigma_w          hurst_w       0.063221   0.016804 -0.129047  0.137748      8       True
             body        SE_high_w       0.063211   0.060392 -0.009311  0.228095      8       True
        bar_range    price_accel_w       0.059925   0.043684 -0.041558  0.312216      8       True
        bar_range             body       0.059922   0.023791 -0.072086  0.233456      8       True
price_velocity_1b reversion_prob_w       0.059536   0.011434 -0.192409  0.072995      8       True
      vol_accel_w    price_sigma_w       0.059264  -0.039589 -0.128272  0.066492      8       True
           z_se_w reversion_prob_w       0.057644   0.037452 -0.080764  0.097443      8       True
      vol_accel_w        SE_high_w       0.053060  -0.037163 -0.128603  0.049296      8       True
price_velocity_1b  vol_velocity_1b       0.052940  -0.036890 -0.067658  0.064200      8       True
 price_velocity_w         SE_low_w       0.052711  -0.027451 -0.167996  0.054930      8       True
      vol_accel_w         SE_low_w       0.052705  -0.030554 -0.132170  0.064013      8       True
price_velocity_1b        bar_range       0.052251   0.013760 -0.068530  0.212559      8       True
        bar_range          hurst_w       0.051750   0.032124 -0.065877  0.112489      8       True
  vol_velocity_1b             body       0.051256  -0.043871 -0.072657  0.029543      8       True
  vol_velocity_1b    price_sigma_w       0.050885  -0.032441 -0.099529  0.073775      8       True
   price_accel_1b     vol_accel_1b       0.050559  -0.026944 -0.059232  0.094460      8       True
price_velocity_1b     vol_accel_1b       0.049325  -0.024381 -0.068275  0.099775      8       True
    price_accel_w    price_sigma_w       0.047346   0.036268 -0.026406  0.201112      8       True
     vol_accel_1b             body       0.046697  -0.028405 -0.064359  0.073166      8       True
    price_accel_w      vol_accel_w       0.045757  -0.044668 -0.076469  0.004354      8       True
  vol_velocity_1b         SE_low_w       0.045533  -0.022326 -0.106553  0.075485      8       True
       vol_mean_w          hurst_w       0.045392   0.023767 -0.042876  0.087010      8       True
           z_se_w        SE_high_w       0.045258   0.045148 -0.000442  0.177522      8       True
   vol_velocity_w         z_high_w       0.044712   0.040834 -0.015512  0.197993      8       True
  vol_velocity_1b        SE_high_w       0.043789  -0.029360 -0.108056  0.057715      8       True
   price_accel_1b        bar_range       0.043427   0.033185 -0.026968  0.160968      8       True
 price_velocity_w    swing_noise_w       0.042926  -0.005716 -0.088487  0.058889      8       True
  vol_velocity_1b    price_accel_w       0.042117  -0.036766 -0.061096  0.021404      8       True
     vol_accel_1b    price_accel_w       0.041199  -0.021382 -0.067556  0.079267      8       True
   price_accel_1b reversion_prob_w       0.040202  -0.000847 -0.164194  0.048403      8       True
       vol_mean_w reversion_prob_w       0.037540   0.032095 -0.015308  0.106037      8       True
      vol_sigma_w reversion_prob_w       0.037198  -0.001338 -0.042607  0.069963      8       True
      vol_sigma_w          hurst_w       0.036337   0.030305 -0.024128  0.094279      8       True
             body      vol_accel_w       0.035919  -0.028152 -0.054115  0.031067      8       True
  vol_velocity_1b           z_se_w       0.035365  -0.033835 -0.085923  0.006116      8       True
    price_accel_w         SE_low_w       0.035211   0.017441 -0.025531  0.148139      8       True
price_velocity_1b      vol_accel_w       0.034566  -0.025750 -0.048411  0.035265      8       True
   price_accel_1b        SE_high_w       0.034557   0.029006 -0.022201  0.166659      8       True
 price_velocity_w         z_high_w       0.034396  -0.006098 -0.051449  0.091998      8       True
   vol_velocity_w    swing_noise_w       0.034165  -0.007432 -0.101605  0.048176      8       True
      vol_sigma_w          z_low_w       0.033153   0.029757 -0.009278  0.080372      8       True
 price_velocity_w          hurst_w       0.032641  -0.031625 -0.098478  0.004066      8       True
         z_high_w         SE_low_w       0.032211   0.010661 -0.037773  0.110416      8       True
 price_velocity_w reversion_prob_w       0.031642   0.012663 -0.075916  0.084223      8       True
price_velocity_1b    price_sigma_w       0.030970   0.010517 -0.039099  0.079966      8       True
    price_accel_w   vol_velocity_w       0.030676  -0.019954 -0.042512  0.042888      8       True
    price_accel_w       vol_mean_w       0.030429   0.025611 -0.019274  0.115629      8       True
 price_velocity_w          z_low_w       0.029758  -0.001359 -0.050742  0.095909      8       True
      vol_sigma_w         z_high_w       0.029741   0.006180 -0.066649  0.077457      8       True
 price_velocity_w           z_se_w       0.029633   0.003278 -0.052848  0.102624      8       True
        bar_range           z_se_w       0.029564  -0.003623 -0.051343  0.103764      8       True
    price_accel_w reversion_prob_w       0.029107   0.015636 -0.053884  0.056125      8       True
   price_accel_1b  vol_velocity_1b       0.028893  -0.014104 -0.041653  0.059158      8       True
      vol_accel_w           z_se_w       0.027042  -0.018632 -0.047462  0.033639      8       True
      vol_accel_w    swing_noise_w       0.026062  -0.003890 -0.028323  0.051679      8       True
   price_accel_1b      vol_accel_w       0.025689  -0.003258 -0.033647  0.072316      8       True
   price_accel_1b price_velocity_w       0.025123  -0.005426 -0.066355  0.043754      8       True
             body    price_sigma_w       0.024836   0.006251 -0.028633  0.067291      8       True
           z_se_w          hurst_w       0.024789  -0.008031 -0.108710  0.036007      8       True
          z_low_w          hurst_w       0.024708  -0.005467 -0.101162  0.047857      8       True
  vol_velocity_1b    swing_noise_w       0.023571  -0.007647 -0.043248  0.048807      8       True
price_velocity_1b         SE_low_w       0.023323  -0.020492 -0.058605  0.011325      8       True
   price_accel_1b   vol_velocity_w       0.023312   0.016860 -0.025808  0.109923      8       True
         z_high_w        SE_high_w       0.023231   0.015170 -0.016784  0.092017      8       True
    price_sigma_w         z_high_w       0.023197  -0.016236 -0.052048  0.027844      8       True
    price_accel_w     price_mean_w       0.022725  -0.016539 -0.068524  0.015878      8       True
       vol_mean_w         z_high_w       0.022574  -0.004445 -0.075022  0.037105      8       True
     vol_accel_1b         SE_low_w       0.022435  -0.005409 -0.039674  0.051404      8       True
         z_high_w          hurst_w       0.022330  -0.009620 -0.094180  0.017096      8       True
    price_accel_w      vol_sigma_w       0.020765   0.012620 -0.029021  0.074224      8       True
     vol_accel_1b        SE_high_w       0.020549  -0.007086 -0.034184  0.036474      8       True
     vol_accel_1b    price_sigma_w       0.020314  -0.003318 -0.035872  0.050799      8       True
         SE_low_w reversion_prob_w       0.020219   0.015454 -0.013547  0.071318      8       True
    price_sigma_w reversion_prob_w       0.020183  -0.010680 -0.029869  0.022348      8       True
   vol_velocity_w          hurst_w       0.019563  -0.019324 -0.042033  0.000954      8       True
    price_accel_w           vwap_w       0.019404  -0.013203 -0.052526  0.015912      8       True
   vol_velocity_w           z_se_w       0.018367  -0.007010 -0.052673  0.045426      8       True
             body          hurst_w       0.018096  -0.016603 -0.069384  0.002858      8       True
price_velocity_1b    swing_noise_w       0.017771   0.001810 -0.023364  0.028691      8       True
 reversion_prob_w    swing_noise_w       0.017198  -0.011306 -0.098662  0.011322      8       True
price_velocity_1b       vol_mean_w       0.016917  -0.016825 -0.038253  0.000368      8       True
             body    swing_noise_w       0.016600   0.003178 -0.022480  0.044216      8       True
     vol_accel_1b price_velocity_w       0.016597   0.003582 -0.026933  0.067614      8       True
     vol_accel_1b           z_se_w       0.016368  -0.011993 -0.045749  0.017498      8       True
       vol_mean_w           z_se_w       0.016171   0.014787 -0.003634  0.052119      8       True
price_velocity_1b          hurst_w       0.015478  -0.014461 -0.058885  0.002735      8       True
  vol_velocity_1b price_velocity_w       0.015364  -0.012377 -0.026953  0.011949      8       True
    price_accel_w          hurst_w       0.014867  -0.001073 -0.051662  0.053951      8       True
 price_velocity_w      vol_accel_w       0.014827  -0.012021 -0.030887  0.011221      8       True
        SE_high_w reversion_prob_w       0.014699   0.013500 -0.003592  0.044569      8       True
      vol_accel_w          hurst_w       0.013499  -0.006397 -0.055880  0.015197      8       True
  vol_velocity_1b          hurst_w       0.013195  -0.006849 -0.070392  0.010784      8       True
     price_mean_w         z_high_w       0.013126  -0.013085 -0.038438  0.000163      8       True
   price_accel_1b      vol_sigma_w       0.012254  -0.007738 -0.049321  0.006560      8       True
      vol_sigma_w           z_se_w       0.012228   0.007704 -0.007489  0.055482      8       True
           vwap_w         z_high_w       0.011532  -0.011481 -0.036044  0.000206      8       True
         z_high_w    swing_noise_w       0.011457  -0.001163 -0.012082  0.040719      8       True
     vol_accel_1b    swing_noise_w       0.011392   0.005752 -0.009060  0.033614      8       True
    price_accel_w    swing_noise_w       0.009661   0.003190 -0.019123  0.022982      8       True
    price_sigma_w           z_se_w       0.009317   0.007761 -0.006227  0.044467      8       True
   vol_velocity_w           vwap_w       0.009099   0.008211 -0.003553  0.038670      8       True
             body     price_mean_w       0.009070  -0.004622 -0.019848  0.009225      8       True
   price_accel_1b    price_sigma_w       0.008988  -0.000214 -0.017506  0.013993      8       True
          z_low_w    swing_noise_w       0.008640   0.007370 -0.002596  0.032015      8       True
             body       vol_mean_w       0.008426  -0.007620 -0.032940  0.001966      8       True
   vol_velocity_w     price_mean_w       0.008381   0.007507 -0.003497  0.032727      8       True
     price_mean_w reversion_prob_w       0.008152  -0.000330 -0.021853  0.021819      8       True
   price_accel_1b       vol_mean_w       0.007970  -0.005097 -0.026865  0.006174      8       True
           vwap_w reversion_prob_w       0.007769  -0.000824 -0.021706  0.018194      8       True
      vol_accel_w           vwap_w       0.007278   0.005397 -0.007181  0.035306      8       True
      vol_accel_w     price_mean_w       0.006841   0.004972 -0.007158  0.033198      8       True
     vol_accel_1b          hurst_w       0.006780  -0.001554 -0.027302  0.007404      8       True
price_velocity_1b     price_mean_w       0.006686  -0.001236 -0.010642  0.009630      8       True
   price_accel_1b    swing_noise_w       0.006546  -0.001632 -0.019737  0.013922      8       True
             body           vwap_w       0.006403  -0.001842 -0.009176  0.009345      8       True
price_velocity_1b           vwap_w       0.005085   0.002281 -0.004744  0.009745      8       True
   price_accel_1b          hurst_w       0.004854  -0.000686 -0.013132  0.007448      8       True
   price_accel_1b           vwap_w       0.004848   0.002994 -0.003860  0.018972      8       True
   price_accel_1b     price_mean_w       0.004360   0.002776 -0.002978  0.016531      8       True
           z_se_w    swing_noise_w       0.003242  -0.002367 -0.006254  0.002991      8       True
     vol_accel_1b     price_mean_w       0.003184  -0.001637 -0.010813  0.005162      8       True
     vol_accel_1b           vwap_w       0.003130  -0.001671 -0.010844  0.004701      8       True
  vol_velocity_1b           vwap_w       0.002104   0.001138 -0.003756  0.005631      8       True
  vol_velocity_1b     price_mean_w       0.002041   0.001086 -0.003735  0.005431      8       True
