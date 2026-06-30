# Oscillation period — first-return time, anchored (almost) every bar
STRIDE=1 bar(s), forward cap 360 min (beyond = no-return/censored = trend). Pure measurement — no positions.

## 2024: 338632 anchors
- returned (finite period): 314407/338632 = 92.8%
- NO return within 360m (censored = trend): 24225/338632 = 7.2%
- mode period ~2m | median 5m | mean 20m
```
first-return time (oscillation period), mode 0-10m, median 5m, n=314407
  0-10 m|############################################## 215269
 10-20 m|######## 37051
 20-30 m|### 15552
 30-40 m|## 9210
 40-50 m|# 6192
 50-60 m|# 4648
 60-70 m|# 3693
 70-80 m|# 2828
 80-90 m|# 2352
 90-100m| 1781
100-110m| 1572
110-120m| 1376
120-130m| 1226
130-140m| 1131
140-150m| 995
150-160m| 854
160-170m| 738
170-180m| 685
180-190m| 648
190-200m| 602
200-210m| 571
210-220m| 483
220-230m| 479
   >230m|# 4471
```

## 2025: 305019 anchors
- returned (finite period): 282670/305019 = 92.7%
- NO return within 360m (censored = trend): 22349/305019 = 7.3%
- mode period ~2m | median 5m | mean 20m
```
first-return time (oscillation period), mode 0-10m, median 5m, n=282670
  0-10 m|############################################## 193358
 10-20 m|######## 34021
 20-30 m|### 14263
 30-40 m|## 8048
 40-50 m|# 5539
 50-60 m|# 4231
 60-70 m|# 3139
 70-80 m|# 2471
 80-90 m|# 2125
 90-100m| 1687
100-110m| 1491
110-120m| 1228
120-130m| 1076
130-140m| 1024
140-150m| 791
150-160m| 725
160-170m| 683
170-180m| 569
180-190m| 603
190-200m| 511
200-210m| 474
210-220m| 448
220-230m| 375
   >230m|# 3790
```

## 2024 vs 2025
- median period: 5m vs 5m
- no-return (trend) share: 7.2% vs 7.3%

## Read
This is the period field: how long price takes to return to an arbitrary level (the oscillation
timescale), with the no-return share = the trend (censored) fraction. It is a measurement of the
market's oscillation period, sampled everywhere — the empirical OU first-return-time distribution.
