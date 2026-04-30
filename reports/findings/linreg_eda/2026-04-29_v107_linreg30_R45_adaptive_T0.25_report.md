# LinReg(30) EDA — ZigzagRunner R=45.0

Generated: 2026-04-29 06:38

## Window
- Atlas: `DATA/ATLAS`
- Range: 2025-01-01 -> 2026-03-21
- LinReg period: 30
- R: 45.0 pts
- Direction: default (HighPivot=Short)
- Total trades: 5817
- Total PnL: $-17,121.80

## PnL by slope sign at entry

Slope is positive (uptrend), negative (downtrend), or near zero (chop).

```
               n         wr  pnl_total  pnl_mean
slope_sign                                      
+           2617  46.847535    -7458.3 -2.849943
-           3200  47.375000    -9663.5 -3.019844
```

## PnL by side (Long/Short) × slope sign

Critical regime question: do longs lose more when slope is negative?

```
                     n         wr  pnl_total   pnl_mean
side  slope_sign                                       
Long  +           2512  47.571656    -3245.8  -1.292118
      -             99  41.414141     -779.1  -7.869697
Short +            105  29.523810    -4212.5 -40.119048
      -           3101  47.565302    -8884.4  -2.865011
```

## PnL by slope magnitude bucket

Where in slope space does the strategy concentrate wins/losses?

```
                     n         wr  pnl_total  pnl_mean
slope_bucket                                          
strong_neg(-1.5)  1119  54.423592    -3665.6 -3.275782
weak_neg(-0.5)    1351  43.893412    -3205.4 -2.372613
flat              1430  39.930070   -11990.5 -8.384965
weak_pos(0.5)     1082  47.042514     2086.7  1.928558
strong_pos(1.5)    835  55.089820     -347.0 -0.415569
```

## PnL by slope-flipped-during-trade

If slope flips sign during the trade, did the trade tend to lose?
Strong signal here = exit-on-slope-reverse rule would help.

```
                  n         wr  pnl_total   pnl_mean
slope_flipped                                       
False          2881  59.666782    90576.6  31.439292
True           2936  34.843324  -107698.4 -36.682016
```

## Hypothetical filter: skip trades against strong slope

'Skip long when slope < -T, skip short when slope > +T'.
Reports kept-trade PnL delta vs no-filter baseline.

| Threshold T | Kept | Skipped | Total PnL | WR | Delta vs base |
|---:|---:|---:|---:|---:|---:|
| 0.5 | 5805 | 12 | $-16,864.50 | 47.2% | $+257.30 |
| 1.0 | 5815 | 2 | $-17,185.00 | 47.1% | $-63.20 |
| 1.5 | 5816 | 1 | $-17,109.40 | 47.1% | $+12.40 |
| 2.0 | 5817 | 0 | $-17,121.80 | 47.1% | $+0.00 |

## Caveats

- LinReg slope is computed from CLOSE prices, not synchronized with intra-bar fills.
- Trade simulation uses 1m close prices; intra-bar SL fires not modeled.
- `slope_at_entry` reflects bar AT WHICH the pivot fired (one bar BEFORE entry).
- Hypothetical filter table assumes filter cleanly skips trades; ignores re-pivoting effects.
