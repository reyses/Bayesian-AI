# Pivot-entry latency — t = pvt + n (causal zigzag, ATLAS 1m)
604 days. Fire at confirmation; measure to next pivot (cap 60 bars).

| retrace R (pt) | pivots | median n | dir-hit% (to peak) | captured-to-PEAK (pt) | net round-trip (pt) | phase@entry |
|---|---|---|---|---|---|---|
| 4 | 156324 | 1 | 63.6% | +8.4 | -0.2 | 60% |
| 6 | 113442 | 2 | 67.2% | +10.7 | -0.2 | 58% |
| 8 | 86807 | 2 | 69.8% | +12.9 | -0.3 | 57% |
| 12 | 56297 | 3 | 74.1% | +17.3 | -0.4 | 55% |
| 16 | 40232 | 3 | 76.6% | +21.4 | -0.6 | 54% |
| 24 | 23125 | 5 | 80.9% | +30.1 | -0.5 | 52% |

Reading: hit-rate = fraction of fires whose direction was right to the next pivot; captured = points won in the fired direction; phase@entry = how deep into the leg the confirmation lands. The trade-off is earliness (small n / small R) vs direction reliability (hit-rate) and phase. Note: hit-rate here is pivot-to-pivot direction, BEFORE costs (~3.6 tick RT) — a reliable direction still needs captured > cost to trade.
