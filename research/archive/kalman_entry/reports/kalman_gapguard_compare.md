# Gap-guarded GA-Kalman re-run — clean trades (vs contaminated)

Guard: force-flat before any time-gap>300s or price-jump>15.0pt; no entry on gap bars.
Total trades: 6780 | gap-forced closes: 1874
Worst single loss now: $-127 (was -$454 contaminated)

| split | trades | net $/tr | PF | net $/day | 95% day-block CI | sig | (contaminated $/day) |
|---|---|---|---|---|---|---|---|
| IS | 711 | +6.49 | 1.15 | +36.0 | [-20.0,+90.1] | incl 0 | +57.9 |
| OOS_H2_24 | 1320 | +1.53 | 1.03 | +15.4 | [-61.8,+92.4] | incl 0 | +32.0 |
| OOS_25_26 | 4749 | -2.00 | 0.96 | -28.4 | [-80.8,+22.7] | incl 0 | +1.4 |