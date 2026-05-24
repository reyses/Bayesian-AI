**DEPRECATED — LOOKAHEAD ARTIFACT, DO NOT QUOTE**

# LinReg(30) EDA — ZigzagRunner R=45.0

Generated: 2026-04-29 06:39

## Window
- Atlas: `DATA/ATLAS`
- Range: 2025-01-01 -> 2026-03-21
- LinReg period: 30
- R: 45.0 pts
- Direction: default (HighPivot=Short)
- Total trades: 5817
- Total PnL: $-19,553.80

## PnL by slope sign at entry

Slope is positive (uptrend), negative (downtrend), or near zero (chop).

```
               n         wr  pnl_total  pnl_mean
slope_sign                                      
+           2617  41.497898    -7135.3 -2.726519
-           3200  41.031250   -12418.5 -3.880781
```

## PnL by side (Long/Short) × slope sign

Critical regime question: do longs lose more when slope is negative?

```
                     n         wr  pnl_total  pnl_mean
side  slope_sign                                      
Long  +           1940  43.608247    -1425.5 -0.734794
      -            777  34.877735    -4122.8 -5.306049
Short +            677  35.450517    -5709.8 -8.433973
      -           2423  43.004540    -8295.7 -3.423731
```

## PnL by slope magnitude bucket

Where in slope space does the strategy concentrate wins/losses?

```
                     n         wr  pnl_total  pnl_mean
slope_bucket                                          
strong_neg(-1.5)  1119  50.580876    -8620.6 -7.703843
weak_neg(-0.5)    1351  36.713546     -983.4 -0.727905
flat              1430  34.405594    -4980.5 -3.482867
weak_pos(0.5)     1082  37.615527    -2380.3 -2.199908
strong_pos(1.5)    835  52.455090    -2589.0 -3.100599
```

## PnL by slope-flipped-during-trade

If slope flips sign during the trade, did the trade tend to lose?
Strong signal here = exit-on-slope-reverse rule would help.

```
                  n         wr  pnl_total   pnl_mean
slope_flipped                                       
False          2881  48.004165    41149.6  14.283096
True           2936  34.604905   -60703.4 -20.675545
```

## Hypothetical filter: skip trades against strong slope

'Skip long when slope < -T, skip short when slope > +T'.
Reports kept-trade PnL delta vs no-filter baseline.

| Threshold T | Kept | Skipped | Total PnL | WR | Delta vs base |
|---:|---:|---:|---:|---:|---:|
| 0.5 | 4926 | 891 | $-12,026.40 | 42.5% | $+7,527.40 |
| 1.0 | 5352 | 465 | $-12,998.30 | 42.0% | $+6,555.50 |
| 1.5 | 5653 | 164 | $-15,470.20 | 41.6% | $+4,083.60 |
| 2.0 | 5797 | 20 | $-18,974.30 | 41.3% | $+579.50 |

## Caveats

- LinReg slope is computed from CLOSE prices, not synchronized with intra-bar fills.
- Trade simulation uses 1m close prices; intra-bar SL fires not modeled.
- `slope_at_entry` reflects bar AT WHICH the pivot fired (one bar BEFORE entry).
- Hypothetical filter table assumes filter cleanly skips trades; ignores re-pivoting effects.
