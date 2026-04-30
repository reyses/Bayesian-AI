# LinReg(30) EDA — ZigzagRunner R=45.0

Generated: 2026-04-29 06:38

## Window
- Atlas: `DATA/ATLAS`
- Range: 2025-01-01 -> 2026-03-21
- LinReg period: 30
- R: 45.0 pts
- Direction: default (HighPivot=Short)
- Total trades: 5817
- Total PnL: $-19,103.80

## PnL by slope sign at entry

Slope is positive (uptrend), negative (downtrend), or near zero (chop).

```
               n         wr  pnl_total  pnl_mean
slope_sign                                      
+           2617  45.624761    -7129.3 -2.724226
-           3200  46.093750   -11974.5 -3.742031
```

## PnL by side (Long/Short) × slope sign

Critical regime question: do longs lose more when slope is negative?

```
                     n         wr  pnl_total   pnl_mean
side  slope_sign                                       
Long  +           2425  46.886598    -2829.0  -1.166598
      -            196  34.183673    -2215.9 -11.305612
Short +            192  29.687500    -4300.3 -22.397396
      -           3004  46.870839    -9758.6  -3.248535
```

## PnL by slope magnitude bucket

Where in slope space does the strategy concentrate wins/losses?

```
                     n         wr  pnl_total  pnl_mean
slope_bucket                                          
strong_neg(-1.5)  1119  54.334227    -3925.6 -3.508132
weak_neg(-0.5)    1351  43.597335    -3535.4 -2.616876
flat              1430  35.594406   -12668.5 -8.859091
weak_pos(0.5)     1082  46.487985     1372.7  1.268669
strong_pos(1.5)    835  55.089820     -347.0 -0.415569
```

## PnL by slope-flipped-during-trade

If slope flips sign during the trade, did the trade tend to lose?
Strong signal here = exit-on-slope-reverse rule would help.

```
                  n         wr  pnl_total  pnl_mean
slope_flipped                                      
False          2881  57.410621    82979.6  28.80236
True           2936  34.570845  -102083.4 -34.76955
```

## Hypothetical filter: skip trades against strong slope

'Skip long when slope < -T, skip short when slope > +T'.
Reports kept-trade PnL delta vs no-filter baseline.

| Threshold T | Kept | Skipped | Total PnL | WR | Delta vs base |
|---:|---:|---:|---:|---:|---:|
| 0.5 | 5789 | 28 | $-18,148.10 | 46.0% | $+955.70 |
| 1.0 | 5813 | 4 | $-19,237.70 | 45.9% | $-133.90 |
| 1.5 | 5815 | 2 | $-18,958.50 | 45.9% | $+145.30 |
| 2.0 | 5816 | 1 | $-18,970.90 | 45.9% | $+132.90 |

## Caveats

- LinReg slope is computed from CLOSE prices, not synchronized with intra-bar fills.
- Trade simulation uses 1m close prices; intra-bar SL fires not modeled.
- `slope_at_entry` reflects bar AT WHICH the pivot fired (one bar BEFORE entry).
- Hypothetical filter table assumes filter cleanly skips trades; ignores re-pivoting effects.
