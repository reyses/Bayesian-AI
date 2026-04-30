# LinReg(30) EDA — ZigzagRunner R=45.0

Generated: 2026-04-29 06:39

## Window
- Atlas: `DATA/ATLAS`
- Range: 2025-01-01 -> 2026-03-21
- LinReg period: 30
- R: 45.0 pts
- Direction: default (HighPivot=Short)
- Total trades: 5817
- Total PnL: $-12,195.80

## PnL by slope sign at entry

Slope is positive (uptrend), negative (downtrend), or near zero (chop).

```
               n         wr  pnl_total  pnl_mean
slope_sign                                      
+           2617  42.873519    -3797.3 -1.451013
-           3200  42.875000    -8398.5 -2.624531
```

## PnL by side (Long/Short) × slope sign

Critical regime question: do longs lose more when slope is negative?

```
                     n         wr  pnl_total  pnl_mean
side  slope_sign                                      
Long  +           2066  44.820910     -121.9 -0.059003
      -            626  36.102236    -1674.9 -2.675559
Short +            551  35.571688    -3675.4 -6.670417
      -           2574  44.522145    -6723.6 -2.612121
```

## PnL by slope magnitude bucket

Where in slope space does the strategy concentrate wins/losses?

```
                     n         wr  pnl_total  pnl_mean
slope_bucket                                          
strong_neg(-1.5)  1119  53.976765    -4585.6 -4.097945
weak_neg(-0.5)    1351  38.119911    -1309.4 -0.969208
flat              1430  34.545455    -4669.5 -3.265385
weak_pos(0.5)     1082  39.463956     -143.3 -0.132440
strong_pos(1.5)    835  54.371257    -1488.0 -1.782036
```

## PnL by slope-flipped-during-trade

If slope flips sign during the trade, did the trade tend to lose?
Strong signal here = exit-on-slope-reverse rule would help.

```
                  n         wr  pnl_total   pnl_mean
slope_flipped                                       
False          2881  51.197501    57483.6  19.952655
True           2936  34.707084   -69679.4 -23.732766
```

## Hypothetical filter: skip trades against strong slope

'Skip long when slope < -T, skip short when slope > +T'.
Reports kept-trade PnL delta vs no-filter baseline.

| Threshold T | Kept | Skipped | Total PnL | WR | Delta vs base |
|---:|---:|---:|---:|---:|---:|
| 0.5 | 5203 | 614 | $-8,995.20 | 43.8% | $+3,200.60 |
| 1.0 | 5617 | 200 | $-9,850.80 | 43.2% | $+2,345.00 |
| 1.5 | 5803 | 14 | $-11,115.20 | 43.0% | $+1,080.60 |
| 2.0 | 5815 | 2 | $-12,048.50 | 42.9% | $+147.30 |

## Caveats

- LinReg slope is computed from CLOSE prices, not synchronized with intra-bar fills.
- Trade simulation uses 1m close prices; intra-bar SL fires not modeled.
- `slope_at_entry` reflects bar AT WHICH the pivot fired (one bar BEFORE entry).
- Hypothetical filter table assumes filter cleanly skips trades; ignores re-pivoting effects.
