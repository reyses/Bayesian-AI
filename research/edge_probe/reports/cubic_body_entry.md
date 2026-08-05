# Leg-body entry — cubic-slope direction vs random (ATLAS 1m)

150 days. Enter sign(slope) when |slope|>=THR, ride till slope flips. Friction 0.89pt/RT.

| THR (pt/min) | mode | N | mean pt/trade | $/day (×$2) | 95% CI | PF-WR |
|---|---|---|---|---|---|---|
| 5 | body | 32,009 | -1.14 | -485.8 | [-563.8, -411.7] | -0.225 |
| 5 | anti | 37,233 | -0.89 | -441.2 | [-512.8, -371.2] | -0.214 |
| 5 | rand | 34,347 | -0.96 | -441.9 | [-521.6, -366.2] | -0.211 |
| 10 | body | 18,133 | -1.19 | -286.9 | [-355.9, -224.3] | -0.193 |
| 10 | anti | 20,755 | -0.82 | -228.2 | [-286.1, -173.8] | -0.170 |
| 10 | rand | 19,052 | -1.08 | -274.5 | [-342.6, -208.3] | -0.193 |
| 20 | body | 6,230 | -1.02 | -91.1 | [-146.1, -38.1] | -0.125 |
| 20 | anti | 7,260 | -1.17 | -121.0 | [-176.6, -73.4] | -0.182 |
| 20 | rand | 6,437 | -0.82 | -75.7 | [-122.9, -28.5] | -0.113 |
| 40 | body | 1,156 | -2.57 | -55.5 | [-100.8, -13.2] | -0.211 |
| 40 | anti | 1,206 | -0.41 | -9.3 | [-36.8, +18.0] | -0.050 |
| 40 | rand | 1,189 | -1.14 | -25.4 | [-61.1, +11.2] | -0.111 |

Read: if BODY mean pt/trade > 0 and >> ANTI/RAND, being on the leg's SIDE (cubic-slope direction) is a real edge with random timing = "just be in a leg" holds, and the cubic is the leg-body+direction detector. If BODY ≈ RAND ≈ 0 after friction, leg-body entry has no edge (the slope is not causally predictive of the continuation).
