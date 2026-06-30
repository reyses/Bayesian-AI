# Trades unlocked by cutting a wrong trade IN TIME (2024+2025)
21383 wrong trades. Opportunity baseline: ~6.1 tradeable swings/hour available (all good+bad).

## Cut at 15 min underwater
- bounced in time (no cut needed): 10545/21383 = 49%
- STILL STUCK (cutting helps): 10838/21383 = 51%
- for the stuck ones, TRADES UNLOCKED by cutting: mode ~1, median 4, mean 17.8, 90th pct 58
```
TRADES UNLOCKED (mode 0-2, median 4, n=10838)
  0-2  |###################################### 3902
  2-4  |############## 1481
  4-6  |######## 857
  6-8  |###### 627
  8-10 |#### 429
 10-12 |### 304
 12-14 |## 256
 14-16 |## 194
 16-18 |## 180
 18-20 |# 140
 20-22 |# 137
 22-24 |# 119
 24-26 |# 89
 26-28 |# 113
 28-30 |# 61
 30-32 |################### 1949
```

## Cut at 30 min underwater
- bounced in time (no cut needed): 13477/21383 = 63%
- STILL STUCK (cutting helps): 7906/21383 = 37%
- for the stuck ones, TRADES UNLOCKED by cutting: mode ~1, median 6, mean 22.8, 90th pct 72
```
TRADES UNLOCKED (mode 0-2, median 6, n=7906)
  0-2  |###################################### 2198
  2-4  |################# 1012
  4-6  |########### 638
  6-8  |######## 456
  8-10 |##### 316
 10-12 |##### 262
 12-14 |#### 220
 14-16 |### 180
 16-18 |## 137
 18-20 |## 117
 20-22 |## 137
 22-24 |## 96
 24-26 |## 101
 26-28 |# 84
 28-30 |# 69
 30-32 |################################# 1883
```

## Cut at 60 min underwater
- bounced in time (no cut needed): 15838/21383 = 74%
- STILL STUCK (cutting helps): 5545/21383 = 26%
- for the stuck ones, TRADES UNLOCKED by cutting: mode ~1, median 11, mean 29.2, 90th pct 84
```
TRADES UNLOCKED (mode 30-32, median 11, n=5545)
  0-2  |######################## 1117
  2-4  |############ 556
  4-6  |######### 435
  6-8  |###### 277
  8-10 |##### 223
 10-12 |##### 219
 12-14 |#### 167
 14-16 |### 141
 16-18 |### 117
 18-20 |### 120
 20-22 |## 108
 22-24 |## 89
 24-26 |## 86
 26-28 |## 78
 28-30 |# 64
 30-32 |###################################### 1748
```

## Read
Most wrong trades bounce fast (no cut needed). The value of cutting is concentrated in the
STILL-STUCK minority — and there it is large: cutting frees a meaningful number of trades
instead of staring at dead money. This is the cost-of-not-cutting in the unit that matters,
with NO dependence on how long the eventual 'recovery' takes (the trap we avoided).
