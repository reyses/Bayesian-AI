# Detrended oscillation — the zigzag around a 30-min moving mean (2024+2025)
66398 oscillation cycles (mean-crossings). price = drift(MA) + oscillation(deviation).

- median zigzag period: **3 min** | median amplitude (max dev): **6pt** | mode period 1m
- vs fixed-anchor: trend/no-return share was 7.2%; DETRENDED there is no 'no-return' — every
  cycle crosses the moving mean. The oscillation is EVERYWHERE, including inside trends.

## Zigzag period distribution (detrended)
```
<3 min    | 42.6% |####################
3-5 min   | 13.5% |######
5-8 min   | 10.2% |#####
8-15 min  | 11.6% |#####
15-30 min | 13.0% |######
30-60 min |  7.8% |####
>60 min   |  1.2% |#
```

## Does the zigzag PERIOD survive the trend? (split cycles by drift = MA slope)
- low-drift (chop) median period:  3 min
- high-drift (trend) median period: 5 min
- low-drift median amplitude:  3pt
- high-drift median amplitude: 11pt
```
low-drift (chop):
<3 min    | 47.0% |######################
3-5 min   | 16.3% |#######
5-8 min   | 12.5% |######
8-15 min  | 13.8% |######
15-30 min |  9.0% |####
30-60 min |  1.3% |#
>60 min   |  0.1% |

high-drift (trend):
<3 min    | 38.3% |##################
3-5 min   | 10.8% |#####
5-8 min   |  7.8% |####
8-15 min  |  9.4% |####
15-30 min | 17.0% |########
30-60 min | 14.4% |#######
>60 min   |  2.2% |#
```

## Read
If the period is ~the same in trend and chop, the oscillation CLOCK keeps ticking while price
climbs — the trend is just the moving mean drifting, with the SAME zigzag riding on it. The
fixed-anchor '7% trend = no oscillation' was an artifact of a fixed reference. Correct model:
price = drift + persistent oscillation; the 'trend' is drift in the mean, not an absence of cycle.
