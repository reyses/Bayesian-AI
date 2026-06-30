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
```

## 2024 vs 2025
- median period: 5m vs 5m
- no-return (trend) share: 7.2% vs 7.3%

## Read
This is the period field: how long price takes to return to an arbitrary level (the oscillation
timescale), with the no-return share = the trend (censored) fraction. It is a measurement of the
market's oscillation period, sampled everywhere — the empirical OU first-return-time distribution.
