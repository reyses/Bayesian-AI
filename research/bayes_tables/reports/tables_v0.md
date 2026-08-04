# BAYESIAN TABLE ACTUARY v0

Counted, not trained. Every cell is a Beta posterior whose prior is the event's own global rate with 20 pseudo-counts (hierarchical shrinkage), so a thin cell is pulled toward the base rate rather than shouting from 3 samples. A cell is ACTIONABLE only when it survives BOTH a day-clustered bootstrap (4,000 draws resampling DAYS, because events inside one session are correlated and the iid Beta interval is too narrow) AND Benjamini-Hochberg FDR control at q=0.05 across every cell in the table, with n >= 15. Live sim day excluded.

## fakeout_poke — P(exceed_ref_first == True)

N = 153,029 events, global rate 0.782, 198 cells, 158 ACTIONABLE

    kind dir_s depth_b age_b clock_b    n  days    raw   post  day_lo  day_hi    lift  actionable
BREAKOUT    up     2-5 5-30m    1200 3842   519 0.9258 0.9251  0.9168  0.9347  0.1429        True
BREAKOUT    up     2-5 5-30m    1030 3563   523 0.8984 0.8978  0.8868  0.9095  0.1155        True
BREAKOUT    up     2-5  30m+    1030 3477   494 0.9177 0.9170  0.9073  0.9271  0.1348        True
BREAKOUT    up     2-5  30m+    1200 3432   507 0.9175 0.9168  0.9069  0.9280  0.1345        True
BREAKOUT    dn     2-5 5-30m    1200 3215   510 0.9061 0.9053  0.8961  0.9161  0.1231        True
BREAKOUT    dn     2-5  30m+    1200 2847   484 0.9031 0.9022  0.8917  0.9141  0.1200        True
BREAKOUT    dn     2-5 5-30m    1030 2844   515 0.8822 0.8815  0.8698  0.8942  0.0993        True
BREAKOUT    dn     2-5  30m+    1030 2662   481 0.8775 0.8768  0.8638  0.8909  0.0946        True
BREAKOUT    up     2-5 5-30m    1400 2586   501 0.9138 0.9128  0.9018  0.9253  0.1305        True
  RETURN    up     1-2 5-30m    1200 2253   504 0.7230 0.7236  0.7008  0.7443 -0.0587        True
BREAKOUT    dn     2-5 5-30m    1400 2216   490 0.9057 0.9046  0.8943  0.9169  0.1224        True
  RETURN    up     1-2 5-30m    1030 2204   507 0.6842 0.6851  0.6630  0.7052 -0.0971        True

**Strongest actionable cells (|lift| vs global):**

  kind dir_s depth_b age_b clock_b   n  days   post  day_lo  day_hi    lift       p
RETURN    dn     1-2  30m+    0930 475   243 0.5710  0.5162  0.6061 -0.2112 0.00025
RETURN    up   0.5-1 5-30m    0930 428   275 0.5863  0.5271  0.6244 -0.1960 0.00025
RETURN    dn   <=0.5 5-30m    0930 414   259 0.5937  0.5386  0.6321 -0.1886 0.00025
RETURN    up   <=0.5  30m+    0930 289   177 0.5943  0.5188  0.6418 -0.1879 0.00025
RETURN    up   0.5-1  30m+    1000 265   170 0.6093  0.5316  0.6603 -0.1729 0.00025
RETURN    dn   <=0.5  30m+    0930 248   169 0.6106  0.5357  0.6579 -0.1716 0.00025

## fakeout_poke — P(sym_race == CONT)

N = 153,029 events, global rate 0.498, 70 cells, 2 ACTIONABLE

    kind dir_s depth_b clock_b    n  days    raw   post  day_lo  day_hi    lift  actionable
BREAKOUT    up     2-5    1200 9133   529 0.5071 0.5070  0.4941  0.5209  0.0094       False
BREAKOUT    up     2-5    1030 9086   534 0.4997 0.4997  0.4846  0.5139  0.0020       False
BREAKOUT    dn     2-5    1200 7624   524 0.4934 0.4935  0.4788  0.5076 -0.0042       False
BREAKOUT    dn     2-5    1030 7171   532 0.4910 0.4910  0.4757  0.5057 -0.0067       False
BREAKOUT    up     2-5    1400 5999   514 0.4679 0.4680  0.4512  0.4840 -0.0297        True
  RETURN    up     1-2    1030 5695   531 0.5243 0.5242  0.5083  0.5397  0.0265        True
  RETURN    up     1-2    1200 5507   528 0.5119 0.5118  0.4957  0.5277  0.0142       False
  RETURN    dn     1-2    1030 5305   523 0.5084 0.5083  0.4921  0.5251  0.0107       False
BREAKOUT    dn     2-5    1400 5144   506 0.4868 0.4868  0.4705  0.5030 -0.0109       False
  RETURN    dn     1-2    1200 5122   523 0.5158 0.5157  0.4976  0.5336  0.0181       False
  RETURN    up     1-2    1400 3860   502 0.4736 0.4737  0.4560  0.4915 -0.0240       False
  RETURN    dn     1-2    1400 3640   489 0.4926 0.4926  0.4741  0.5110 -0.0051       False

**Strongest actionable cells (|lift| vs global):**

    kind dir_s depth_b clock_b    n  days   post  day_lo  day_hi    lift       p
BREAKOUT    up     2-5    1400 5999   514 0.4680  0.4512  0.4840 -0.0297 0.00025
  RETURN    up     1-2    1030 5695   531 0.5242  0.5083  0.5397  0.0265 0.00100

## leg_descent — P(race == NEW_LOW)

N = 58,480 events, global rate 0.690, 15 cells, 6 ACTIONABLE

chain_b clock_b    n  days    raw   post  day_lo  day_hi    lift  actionable
      1    1030 6825   536 0.6788 0.6789  0.6681  0.6898 -0.0109       False
      1    1200 6450   531 0.6958 0.6958  0.6843  0.7075  0.0060       False
     3+    1030 5097   528 0.7036 0.7035  0.6912  0.7159  0.0137       False
     3+    1200 4980   527 0.6948 0.6948  0.6812  0.7082  0.0050       False
      1    1400 4536   518 0.7055 0.7054  0.6919  0.7194  0.0156       False
      1    0930 4523   539 0.6732 0.6733  0.6602  0.6865 -0.0165        True
      2    1030 3826   534 0.6921 0.6921  0.6771  0.7069  0.0023       False
      2    1200 3717   527 0.7127 0.7125  0.6978  0.7271  0.0227        True
     3+    1400 3684   494 0.7090 0.7089  0.6937  0.7239  0.0191        True
      1    1000 3251   530 0.6635 0.6636  0.6469  0.6809 -0.0262        True
      2    1400 2727   514 0.7059 0.7058  0.6890  0.7226  0.0160       False
     3+    0930 2459   506 0.6625 0.6627  0.6438  0.6811 -0.0271        True

**Strongest actionable cells (|lift| vs global):**

chain_b clock_b    n  days   post  day_lo  day_hi    lift      p
      2    1000 1837   522 0.6617  0.6400  0.6831 -0.0281 0.0090
     3+    0930 2459   506 0.6627  0.6438  0.6811 -0.0271 0.0030
      1    1000 3251   530 0.6636  0.6469  0.6809 -0.0262 0.0040
      2    1200 3717   527 0.7125  0.6978  0.7271  0.0227 0.0030
     3+    1400 3684   494 0.7089  0.6937  0.7239  0.0191 0.0125
      1    0930 4523   539 0.6733  0.6602  0.6865 -0.0165 0.0180

## stall — P(race == NEW_EXTREME)

N = 41,180 events, global rate 0.103, 10 cells, 8 ACTIONABLE

dir_s clock_b    n  days    raw   post  day_lo  day_hi    lift  actionable
   dn    1200 6069   537 0.1048 0.1048  0.0958  0.1142  0.0015       False
   up    1200 5862   538 0.1464 0.1462  0.1346  0.1593  0.0429        True
   dn    1030 5679   539 0.0750 0.0751  0.0674  0.0832 -0.0282        True
   up    1030 5641   539 0.1037 0.1037  0.0941  0.1143  0.0004       False
   up    1400 4865   524 0.1379 0.1378  0.1262  0.1508  0.0345        True
   dn    1400 4669   522 0.1157 0.1156  0.1050  0.1269  0.0123        True
   dn    1000 2332   514 0.0527 0.0532  0.0432  0.0634 -0.0501        True
   up    1000 2250   508 0.0818 0.0820  0.0682  0.0964 -0.0213        True
   dn    0930 1920   493 0.0578 0.0583  0.0467  0.0699 -0.0450        True
   up    0930 1893   477 0.0634 0.0638  0.0523  0.0760 -0.0395        True

**Strongest actionable cells (|lift| vs global):**

dir_s clock_b    n  days   post  day_lo  day_hi    lift       p
   dn    1000 2332   514 0.0532  0.0432  0.0634 -0.0501 0.00025
   dn    0930 1920   493 0.0583  0.0467  0.0699 -0.0450 0.00025
   up    1200 5862   538 0.1462  0.1346  0.1593  0.0429 0.00025
   up    0930 1893   477 0.0638  0.0523  0.0760 -0.0395 0.00025
   up    1400 4865   524 0.1378  0.1262  0.1508  0.0345 0.00025
   dn    1030 5679   539 0.0751  0.0674  0.0832 -0.0282 0.00025

## ultra_chop — P(escape_dir == 1)

N = 18,601 events, global rate 0.509, 15 cells, 0 ACTIONABLE

clock_b ratio_b    n  days    raw   post  day_lo  day_hi    lift  actionable
   1030   loose 6054   523 0.5137 0.5137  0.5003  0.5273  0.0052       False
   1200   loose 5131   514 0.5065 0.5065  0.4928  0.5205 -0.0020       False
   1400   loose 3111   492 0.4995 0.4996  0.4816  0.5180 -0.0089       False
   1000   loose 1464   480 0.5178 0.5176  0.4900  0.5450  0.0091       False
   1030     mid 1055   430 0.4976 0.4978  0.4679  0.5281 -0.0107       False
   1200     mid  818   381 0.5330 0.5324  0.4976  0.5668  0.0239       False
   1400     mid  428   291 0.4836 0.4848  0.4369  0.5315 -0.0238       False
   1000     mid  180   147 0.5444 0.5409  0.4689  0.6145  0.0323       False
   1030   tight  119   105 0.4370 0.4473  0.3482  0.5294 -0.0613       False
   1200   tight  105    94 0.4762 0.4814  0.3796  0.5755 -0.0272       False
   0930   loose   65    57 0.5692 0.5549  0.4426  0.6984  0.0464       False
   1400   tight   45    43 0.4000 0.4334  0.2558  0.5455 -0.0751       False

**No cell separates from the global rate** — this question is answered by the base rate alone.

## defended_poke_shelf — P(outcome == CRACK)

N = 1,585 events, global rate 0.374, 10 cells, 2 ACTIONABLE

day_class clock_b   n  days    raw   post  day_lo  day_hi    lift  actionable
    other    1200 413   244 0.4310 0.4284  0.3812  0.4814  0.0542       False
    other    1400 360   235 0.4861 0.4802  0.4336  0.5359  0.1061        True
    other    0930 272   272 0.1765 0.1900  0.1324  0.2206 -0.1841        True
    other    1030 188   131 0.3457 0.3485  0.2821  0.4098 -0.0257       False
    other    1000 127   127 0.3071 0.3162  0.2283  0.3858 -0.0579       False
   flushV    1400  93    56 0.3763 0.3760  0.3000  0.4536  0.0018       False
   flushV    1200  86    51 0.4419 0.4291  0.3452  0.5395  0.0549       False
   flushV    1030  31    27 0.4194 0.4016  0.2424  0.6071  0.0275       False
   flushV    1000  11    11 0.0909 0.2736     NaN     NaN -0.1005       False
   flushV    0930   4     4 0.2500 0.3534     NaN     NaN -0.0207       False

**Strongest actionable cells (|lift| vs global):**

day_class clock_b   n  days   post  day_lo  day_hi    lift       p
    other    0930 272   272 0.1900  0.1324  0.2206 -0.1841 0.00025
    other    1400 360   235 0.4802  0.4336  0.5359  0.1061 0.00025

