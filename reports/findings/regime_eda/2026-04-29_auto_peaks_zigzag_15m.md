# Auto peaks (ZigZag) - TF=15m

Generated: 2026-04-29 22:06

## Setup

- Atlas: `DATA/ATLAS`
- TF: 15m
- Total bars: 26,075
- Algorithm: detect_swings from auto_swing_marker.py (ZigZag with R-threshold)
- Calibrated against: `DATA/regime_seeds\human_peaks_2026-02-01_to_2026-02-07_15m.json`
- Tolerance: +/- 3 bars
- Best F1: 0.538 (precision=0.439, recall=0.694)

## Calibration sweep

```
 min_reversal  min_bars  n_pivots  n_manual  n_matched  precision   recall       f1
           80         3        57        36         25   0.438596 0.694444 0.537634
           50         3        63        36         26   0.412698 0.722222 0.525253
          120         3        49        36         22   0.448980 0.611111 0.517647
          120         2        57        36         24   0.421053 0.666667 0.516129
           80         2        69        36         27   0.391304 0.750000 0.514286
           30         3        67        36         25   0.373134 0.694444 0.485437
           20         5        43        36         19   0.441860 0.527778 0.481013
           20         3        69        36         25   0.362319 0.694444 0.476190
           30         5        39        36         17   0.435897 0.472222 0.453333
          120         5        39        36         17   0.435897 0.472222 0.453333
           50         5        39        36         17   0.435897 0.472222 0.453333
           80         5        39        36         17   0.435897 0.472222 0.453333
           30         2        92        36         29   0.315217 0.805556 0.453125
           50         2        84        36         27   0.321429 0.750000 0.450000
          200         3        31        36         15   0.483871 0.416667 0.447761
           20         2       100        36         29   0.290000 0.805556 0.426471
           50         8        26        36         13   0.500000 0.361111 0.419355
           30         8        26        36         13   0.500000 0.361111 0.419355
           20         8        26        36         13   0.500000 0.361111 0.419355
          200         2        37        36         15   0.405405 0.416667 0.410959
           80         8        24        36         12   0.500000 0.333333 0.400000
          120         8        24        36         12   0.500000 0.333333 0.400000
          200         8        19        36         10   0.526316 0.277778 0.363636
          200         5        25        36         11   0.440000 0.305556 0.360656
```

## Final

- min_reversal: 80 ticks (20.00 pts)
- min_bars: 3
- max_bars: 0
- Auto peaks (full range): 3087
- Output: `DATA/regime_seeds\auto_peaks_2025-01-01_to_2026-03-20_15m.json`

## Next

```
# Re-run macro segmenter to use these auto peaks:
python tools/macro_slope_segmenter.py --tfs 15m
```
