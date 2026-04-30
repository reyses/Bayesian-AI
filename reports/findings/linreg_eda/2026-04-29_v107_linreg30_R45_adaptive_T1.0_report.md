# LinReg(30) EDA — ZigzagRunner R=45.0

Generated: 2026-04-29 06:38

## Window
- Atlas: `DATA/ATLAS`
- Range: 2025-01-01 -> 2026-03-21
- LinReg period: 30
- R: 45.0 pts
- Direction: default (HighPivot=Short)
- Total trades: 5817
- Total PnL: $-8,180.80

## PnL by slope sign at entry

Slope is positive (uptrend), negative (downtrend), or near zero (chop).

```
               n         wr  pnl_total  pnl_mean
slope_sign                                      
+           2617  44.669469     -296.3 -0.113221
-           3200  44.656250    -7884.5 -2.463906
```

## PnL by side (Long/Short) × slope sign

Critical regime question: do longs lose more when slope is negative?

```
                     n         wr  pnl_total  pnl_mean
side  slope_sign                                      
Long  +           2230  46.188341     1153.0  0.517040
      -            430  36.744186     -849.5 -1.975581
Short +            387  35.917313    -1449.3 -3.744961
      -           2770  45.884477    -7035.0 -2.539711
```

## PnL by slope magnitude bucket

Where in slope space does the strategy concentrate wins/losses?

```
                     n         wr  pnl_total  pnl_mean
slope_bucket                                          
strong_neg(-1.5)  1119  54.334227    -3925.6 -3.508132
weak_neg(-0.5)    1351  41.820873    -1385.4 -1.025463
flat              1430  34.685315    -4785.5 -3.346503
weak_pos(0.5)     1082  43.345656     2262.7  2.091220
strong_pos(1.5)    835  55.089820     -347.0 -0.415569
```

## PnL by slope-flipped-during-trade

If slope flips sign during the trade, did the trade tend to lose?
Strong signal here = exit-on-slope-reverse rule would help.

```
                  n         wr  pnl_total   pnl_mean
slope_flipped                                       
False          2881  53.974314    69485.6  24.118570
True           2936  35.524523   -77666.4 -26.453134
```

## Hypothetical filter: skip trades against strong slope

'Skip long when slope < -T, skip short when slope > +T'.
Reports kept-trade PnL delta vs no-filter baseline.

| Threshold T | Kept | Skipped | Total PnL | WR | Delta vs base |
|---:|---:|---:|---:|---:|---:|
| 0.5 | 5557 | 260 | $-8,072.30 | 45.1% | $+108.50 |
| 1.0 | 5796 | 21 | $-8,933.90 | 44.7% | $-753.10 |
| 1.5 | 5815 | 2 | $-8,035.50 | 44.7% | $+145.30 |
| 2.0 | 5816 | 1 | $-8,047.90 | 44.7% | $+132.90 |

## Caveats

- LinReg slope is computed from CLOSE prices, not synchronized with intra-bar fills.
- Trade simulation uses 1m close prices; intra-bar SL fires not modeled.
- `slope_at_entry` reflects bar AT WHICH the pivot fired (one bar BEFORE entry).
- Hypothetical filter table assumes filter cleanly skips trades; ignores re-pivoting effects.
