# Auto peaks (ZigZag) - TF=1h

Generated: 2026-04-29 22:06

## Setup

- Atlas: `DATA/ATLAS`
- TF: 1h
- Total bars: 6,524
- Algorithm: detect_swings from auto_swing_marker.py (ZigZag with R-threshold)
- Calibrated against: `DATA/regime_seeds\human_peaks_2026-02-01_to_2026-02-07_1h.json`
- Tolerance: +/- 3 bars
- Best F1: 0.600 (precision=0.545, recall=0.667)

## Calibration sweep

```
 min_reversal  min_bars  n_pivots  n_manual  n_matched  precision   recall       f1
          120         5        11         9          6   0.545455 0.666667 0.600000
           50         5        11         9          6   0.545455 0.666667 0.600000
           80         5        11         9          6   0.545455 0.666667 0.600000
           20         5        13         9          6   0.461538 0.666667 0.545455
           30         5        13         9          6   0.461538 0.666667 0.545455
           20         8        10         9          5   0.500000 0.555556 0.526316
           30         8        10         9          5   0.500000 0.555556 0.526316
          120         3        18         9          7   0.388889 0.777778 0.518519
           80         3        18         9          7   0.388889 0.777778 0.518519
           20         3        18         9          7   0.388889 0.777778 0.518519
           50         3        18         9          7   0.388889 0.777778 0.518519
           30         3        18         9          7   0.388889 0.777778 0.518519
           50         2        24         9          8   0.333333 0.888889 0.484848
           30         2        24         9          8   0.333333 0.888889 0.484848
           20         2        24         9          8   0.333333 0.888889 0.484848
          120         2        20         9          7   0.350000 0.777778 0.482759
           50         8         8         9          4   0.500000 0.444444 0.470588
          200         5         8         9          4   0.500000 0.444444 0.470588
           80         8         8         9          4   0.500000 0.444444 0.470588
           80         2        22         9          7   0.318182 0.777778 0.451613
          120         8         6         9          3   0.500000 0.333333 0.400000
          200         3        11         9          4   0.363636 0.444444 0.400000
          200         2        11         9          3   0.272727 0.333333 0.300000
          200         8         5         9          2   0.400000 0.222222 0.285714
```

## Final

- min_reversal: 120 ticks (30.00 pts)
- min_bars: 5
- max_bars: 0
- Auto peaks (full range): 610
- Output: `DATA/regime_seeds\auto_peaks_2025-01-01_to_2026-03-20_1h.json`

## Next

```
# Re-run macro segmenter to use these auto peaks:
python tools/macro_slope_segmenter.py --tfs 1h
```
