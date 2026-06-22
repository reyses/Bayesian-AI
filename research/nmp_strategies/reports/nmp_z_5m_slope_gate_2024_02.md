# 5m-slope regime stratification of the z-round-trip (Feb 2024 IS)

|slope| terciles (pts/5m): flat<= 0.735, steep> 2.042

| bucket | n | $/trade mean | median | PF | $/day [95% day-block CI] |
|---|---|---|---|---|---|
| ALL z-round-trip | 1502 | -0.54 | +1.79 | 0.92 | -39 [-140,+58] |
| FLAT 5m (range) | 501 | +0.02 | +1.79 | 1.00 | +1 [-38,+39] |
| sloping + WITH-trend | 555 | -1.58 | +1.79 | 0.79 | -42 [-91,+8] |
| sloping + COUNTER-trend | 446 | +0.11 | +2.38 | 1.01 | +2 [-70,+74] |
| >> KEEP: flat OR with-trend | 1056 | -0.82 | +1.79 | 0.88 | -41 [-104,+20] |
| >> DROP: sloping counter-trend | 446 | +0.11 | +2.38 | 1.01 | +2 [-73,+76] |

Speed-of-change (|dSlope|) split:
| bucket | n | $/trade mean | median | PF | $/day [95% CI] |
|---|---|---|---|---|---|
| calm (|dSlope|<=med) | 751 | +0.24 | +1.79 | 1.06 | +9 [-29,+49] |
| turning (|dSlope|>med) | 750 | -1.34 | +2.38 | 0.85 | -48 [-128,+28] |

CAVEATS: IS (Feb only); tercile/bucket selection invites overfit (graveyard: ~25% cell survival IS->OOS) -> treat structural splits (with/counter sign) over magnitude cells, and OOS-validate before trusting; pre-cost.