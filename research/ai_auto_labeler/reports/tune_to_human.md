# Tuning the AI filler to 398 human picks across 9 days

## Human swing size (|Δprice| between consecutive picks): median 15.8pt, 25th 7.8, 75th 32.2
- share of human swings < 7pt (below current TREND_PTS): 22% -> this is what the 7pt threshold misses.

## Parameter sweep — recall (human found) / precision (pivots real) / F1
```
  N    T |  #piv | recall | prec | F1
 10    3 |   744 |   86% |  43% | 0.573 *
 10    4 |   643 |   80% |  47% | 0.589 *
 10    5 |   532 |   72% |  50% | 0.591 *
 10    6 |   470 |   69% |  54% | 0.607 *
 10    7 |   434 |   65% |  55% | 0.593
 10    8 |   393 |   60% |  56% | 0.582
 15    3 |   571 |   80% |  52% | 0.630 *
 15    4 |   492 |   74% |  56% | 0.640 *
 15    5 |   422 |   69% |  60% | 0.645 *
 15    6 |   372 |   63% |  62% | 0.621
 15    7 |   350 |   58% |  61% | 0.598
 15    8 |   311 |   55% |  65% | 0.596
 20    3 |   427 |   72% |  62% | 0.664 *
 20    4 |   375 |   66% |  65% | 0.657
 20    5 |   332 |   60% |  67% | 0.634
 20    6 |   305 |   57% |  68% | 0.620
 20    7 |   280 |   54% |  71% | 0.614
 20    8 |   254 |   51% |  73% | 0.596
 30    3 |   303 |   57% |  68% | 0.619
 30    4 |   268 |   53% |  72% | 0.608
 30    5 |   243 |   49% |  73% | 0.589
 30    6 |   229 |   47% |  74% | 0.574
 30    7 |   207 |   43% |  75% | 0.546
 30    8 |   191 |   39% |  74% | 0.514
```

## Best match: CUBIC_N=20, TREND_PTS=3  (F1=0.664)
- current default is N=20, TREND_PTS=7. Recommend moving toward the best-F1 cell.
- recall<100% is expected: the human marks some sub-cubic turns no pivot scale will catch;
  raising recall trades precision (more spurious pivots). Pick the knee, not max recall.
