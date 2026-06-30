# Oscillators vs the trend tail — there are no good/bad trades, just trades (2024+2025)
random entries, horizon = rest of day, 'real' once |PnL|>= 5.0pt. n_real=42855

- **OSCILLATORS (return to zero, the wash): 91.4%**
- **RUNAWAYS (trend tail, never return): 8.6%** = the taper, split win 49% / loss 51%
  (random entry => ~symmetric: the trend doesn't care which way you bet)
- (plus 25 entries that never moved 5.0pt = sub-threshold noise)

- runaway reach (terminal |PnL|): median 94pt ($188), mean 144pt, 90th 335pt — the 'to infinity' magnitude

## Signed terminal PnL distribution (oscillators pinned at 0; runaways = the two tails)
```
 -100..  -90 |# 942
  -90..  -80 | 61
  -80..  -70 | 50
  -70..  -60 | 72
  -60..  -50 | 78
  -50..  -40 | 108
  -40..  -30 | 107
  -30..  -20 | 129
  -20..  -10 | 182
  -10..   +0 | 145
   +0..  +10 |############################################## 39300
  +10..  +20 | 173
  +20..  +30 | 160
  +30..  +40 | 91
  +40..  +50 | 98
  +50..  +60 | 59
  +60..  +70 | 59
  +70..  +80 | 62
  +80..  +90 | 52
  +90.. +100 |# 927
```

## Read
The bulk returns to zero (no edge — a wash). The small symmetric taper is the trend tail, and a
'win' vs 'death' is the SAME runaway on opposite sides. So the only edge is: (1) get on the right
side of a runaway, and (2) RIDE it (don't cut at zero) while CUTTING the adverse runaway fast.
The oscillating majority is indifferent — which is why entry-direction prediction found no edge.
