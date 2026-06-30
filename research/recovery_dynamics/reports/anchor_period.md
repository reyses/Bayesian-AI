# Oscillation period — first-return time, anchored (almost) every bar
STRIDE=1 bar(s), forward cap 360 min (beyond = no-return/censored = trend). Pure measurement — no positions.

## 2024: 338632 anchors
- returned (finite period): 314407/338632 = 92.8%
- NO return within 360m (censored = trend): 24225/338632 = 7.2%
- mode period ~2m | median 5m | mean 20m
```
bucket                  | share | cum  |
2-3 min   (mode)        | 25.5% |  26% |############
3-5 min                 | 20.7% |  46% |##########
5-8 min                 | 12.7% |  59% |######
8-15 min                | 11.7% |  71% |#####
15-30 min               |  8.6% |  79% |####
30-60 min               |  5.9% |  85% |###
1-2 h                   |  4.0% |  89% |##
2-6 h                   |  3.8% |  93% |##
NO RETURN (TREND)       |  7.2% | 100% |###

AMPLITUDE (peak excursion, pt) WITHIN each period bucket (rows sum to 100%):
period bucket    | med amp | amp/sqrt(min) |     0-2     2-5    5-10   10-20   20-40  40-inf
--------------------------------------------------------------------------------------------
2-3 min   (mode) |    2pt  |      1.42    |    43%    37%    14%     5%     1%     0%
3-5 min          |    3pt  |      1.62    |    25%    42%    23%     9%     2%     0%
5-8 min          |    5pt  |      2.06    |     9%    38%    32%    17%     5%     1%
8-15 min         |    8pt  |      2.43    |     2%    22%    34%    28%    11%     2%
15-30 min        |   13pt  |      2.79    |     0%     8%    27%    35%    23%     8%
30-60 min        |   21pt  |      3.09    |     0%     1%    14%    32%    33%    19%
1-2 h            |   33pt  |      3.45    |     0%     0%     5%    21%    35%    39%
2-6 h            |   53pt  |      3.42    |     0%     0%     0%     8%    26%    65%
```

## 2025: 305019 anchors
- returned (finite period): 282670/305019 = 92.7%
- NO return within 360m (censored = trend): 22349/305019 = 7.3%
- mode period ~2m | median 5m | mean 20m
```
bucket                  | share | cum  |
2-3 min   (mode)        | 25.4% |  25% |############
3-5 min                 | 20.6% |  46% |#########
5-8 min                 | 12.6% |  59% |######
8-15 min                | 11.8% |  70% |#####
15-30 min               |  8.7% |  79% |####
30-60 min               |  5.8% |  85% |###
1-2 h                   |  4.0% |  89% |##
2-6 h                   |  3.6% |  93% |##
NO RETURN (TREND)       |  7.3% | 100% |###

AMPLITUDE (peak excursion, pt) WITHIN each period bucket (rows sum to 100%):
period bucket    | med amp | amp/sqrt(min) |     0-2     2-5    5-10   10-20   20-40  40-inf
--------------------------------------------------------------------------------------------
2-3 min   (mode) |    3pt  |      1.90    |    30%    38%    20%     9%     2%     0%
3-5 min          |    5pt  |      2.38    |    13%    39%    28%    14%     4%     1%
5-8 min          |    7pt  |      2.84    |     3%    27%    34%    23%    10%     2%
8-15 min         |   12pt  |      3.39    |     0%    11%    31%    33%    18%     7%
15-30 min        |   18pt  |      3.85    |     0%     2%    17%    35%    29%    16%
30-60 min        |   28pt  |      4.25    |     0%     0%     5%    26%    37%    32%
1-2 h            |   45pt  |      4.72    |     0%     0%     0%     9%    34%    57%
2-6 h            |   71pt  |      4.58    |     0%     0%     0%     1%    17%    82%
```

## 2024 vs 2025
- median period: 5m vs 5m
- no-return (trend) share: 7.2% vs 7.3%

## Read
This is the period field: how long price takes to return to an arbitrary level (the oscillation
timescale), with the no-return share = the trend (censored) fraction. It is a measurement of the
market's oscillation period, sampled everywhere — the empirical OU first-return-time distribution.
