# Z-stratified direction EDA — target $15, 8 min

Sampling: 1m-boundary sampling only

## TF summary

| TF | Max IS bias | Max OOS bias | Walk-forward-stable bins |
|---|---:|---:|---:|
| 15s | +2.2pp | +13.5pp | 0/7 |
| 1m | +3.0pp | +3.6pp | 0/7 |
| 5m | +4.0pp | +6.2pp | 0/7 |
| 15m | +1.8pp | +8.5pp | 0/7 |
| 1h | +4.8pp | +2.5pp | 0/7 |

IS events: 2,873,527 · OOS events: 802,692

## 15s_z_se stratified (1m boundaries only)

Directional bias = P(LONG first) - P(SHORT first) per z-bin. **Mean-reversion hypothesis**: negative bias at z_high, positive bias at z_low.

| z bin | z mean | IS N | IS P(L) | IS P(S) | IS bias | OOS N | OOS P(L) | OOS P(S) | OOS bias | Walk-fwd |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| z < -2.5 | -2.78 | 2,196 | 50.7% | 49.0% | +1.7pp | 544 | 45.8% | 54.0% | -8.3pp | — |
| -2.5 ≤ z < -1.5 | -1.86 | 23,418 | 49.4% | 50.4% | -0.9pp | 5,577 | 49.1% | 50.8% | -1.7pp | — |
| -1.5 ≤ z < -0.5 | -0.97 | 62,113 | 49.4% | 50.4% | -1.0pp | 14,562 | 49.7% | 50.2% | -0.5pp | — |
| -0.5 ≤ z ≤ 0.5 | -0.00 | 71,281 | 49.3% | 50.5% | -1.2pp | 16,566 | 49.6% | 50.3% | -0.6pp | — |
| 0.5 < z ≤ 1.5 | +0.97 | 62,442 | 49.5% | 50.3% | -0.9pp | 14,288 | 49.4% | 50.4% | -1.0pp | — |
| 1.5 < z ≤ 2.5 | +1.85 | 22,270 | 48.8% | 51.0% | -2.2pp | 5,652 | 49.3% | 50.5% | -1.2pp | — |
| z > 2.5 | +2.77 | 1,788 | 50.3% | 49.1% | +1.2pp | 512 | 43.2% | 56.6% | -13.5pp | — |

## 1m_z_se stratified (1m boundaries only)

Directional bias = P(LONG first) - P(SHORT first) per z-bin. **Mean-reversion hypothesis**: negative bias at z_high, positive bias at z_low.

| z bin | z mean | IS N | IS P(L) | IS P(S) | IS bias | OOS N | OOS P(L) | OOS P(S) | OOS bias | Walk-fwd |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| z < -2.5 | -2.81 | 3,073 | 51.3% | 48.3% | +3.0pp | 624 | 51.3% | 48.1% | +3.2pp | — |
| -2.5 ≤ z < -1.5 | -1.88 | 25,566 | 49.2% | 50.5% | -1.3pp | 6,041 | 50.3% | 49.6% | +0.7pp | — |
| -1.5 ≤ z < -0.5 | -0.97 | 60,862 | 49.1% | 50.7% | -1.6pp | 14,543 | 49.8% | 50.1% | -0.3pp | — |
| -0.5 ≤ z ≤ 0.5 | +0.00 | 68,467 | 49.3% | 50.6% | -1.3pp | 15,906 | 49.1% | 50.8% | -1.7pp | — |
| 0.5 < z ≤ 1.5 | +0.97 | 60,883 | 49.4% | 50.4% | -0.9pp | 13,995 | 49.4% | 50.4% | -1.0pp | — |
| 1.5 < z ≤ 2.5 | +1.86 | 24,326 | 50.1% | 49.7% | +0.4pp | 5,959 | 48.1% | 51.7% | -3.6pp | — |
| z > 2.5 | +2.81 | 2,331 | 48.7% | 51.0% | -2.3pp | 633 | 51.0% | 48.5% | +2.5pp | — |

## 5m_z_se stratified (1m boundaries only)

Directional bias = P(LONG first) - P(SHORT first) per z-bin. **Mean-reversion hypothesis**: negative bias at z_high, positive bias at z_low.

| z bin | z mean | IS N | IS P(L) | IS P(S) | IS bias | OOS N | OOS P(L) | OOS P(S) | OOS bias | Walk-fwd |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| z < -2.5 | -2.86 | 3,877 | 49.4% | 50.3% | -0.9pp | 705 | 51.9% | 48.1% | +3.8pp | — |
| -2.5 ≤ z < -1.5 | -1.87 | 25,506 | 48.9% | 50.9% | -2.0pp | 6,040 | 52.4% | 47.5% | +4.9pp | — |
| -1.5 ≤ z < -0.5 | -0.98 | 61,041 | 48.9% | 50.9% | -2.0pp | 14,856 | 50.5% | 49.3% | +1.2pp | — |
| -0.5 ≤ z ≤ 0.5 | -0.00 | 68,763 | 49.3% | 50.5% | -1.2pp | 15,974 | 49.1% | 50.8% | -1.7pp | — |
| 0.5 < z ≤ 1.5 | +0.97 | 59,994 | 49.9% | 50.0% | -0.1pp | 13,880 | 48.2% | 51.7% | -3.5pp | — |
| 1.5 < z ≤ 2.5 | +1.86 | 23,859 | 50.1% | 49.7% | +0.4pp | 5,458 | 46.8% | 53.0% | -6.2pp | — |
| z > 2.5 | +2.85 | 2,468 | 47.7% | 51.7% | -4.0pp | 788 | 50.0% | 49.6% | +0.4pp | — |

## 15m_z_se stratified (1m boundaries only)

Directional bias = P(LONG first) - P(SHORT first) per z-bin. **Mean-reversion hypothesis**: negative bias at z_high, positive bias at z_low.

| z bin | z mean | IS N | IS P(L) | IS P(S) | IS bias | OOS N | OOS P(L) | OOS P(S) | OOS bias | Walk-fwd |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| z < -2.5 | -2.91 | 5,211 | 49.2% | 50.3% | -1.1pp | 1,224 | 54.2% | 45.7% | +8.5pp | — |
| -2.5 ≤ z < -1.5 | -1.89 | 24,593 | 49.2% | 50.6% | -1.4pp | 6,375 | 50.1% | 49.8% | +0.3pp | — |
| -1.5 ≤ z < -0.5 | -0.96 | 60,143 | 50.0% | 49.8% | +0.2pp | 14,811 | 49.1% | 50.8% | -1.7pp | — |
| -0.5 ≤ z ≤ 0.5 | -0.00 | 69,884 | 49.0% | 50.8% | -1.8pp | 14,979 | 49.3% | 50.5% | -1.2pp | — |
| 0.5 < z ≤ 1.5 | +0.96 | 59,642 | 49.4% | 50.5% | -1.1pp | 13,720 | 49.7% | 50.2% | -0.5pp | — |
| 1.5 < z ≤ 2.5 | +1.87 | 22,403 | 49.0% | 50.7% | -1.7pp | 5,780 | 48.2% | 51.6% | -3.4pp | — |
| z > 2.5 | +2.85 | 3,632 | 49.3% | 50.1% | -0.8pp | 812 | 48.2% | 51.7% | -3.6pp | — |

## 1h_z_se stratified (1m boundaries only)

Directional bias = P(LONG first) - P(SHORT first) per z-bin. **Mean-reversion hypothesis**: negative bias at z_high, positive bias at z_low.

| z bin | z mean | IS N | IS P(L) | IS P(S) | IS bias | OOS N | OOS P(L) | OOS P(S) | OOS bias | Walk-fwd |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| z < -2.5 | -2.90 | 7,166 | 48.5% | 51.0% | -2.5pp | 1,281 | 50.4% | 49.3% | +1.1pp | — |
| -2.5 ≤ z < -1.5 | -1.90 | 24,939 | 49.9% | 49.9% | +0.0pp | 5,834 | 48.7% | 51.2% | -2.5pp | — |
| -1.5 ≤ z < -0.5 | -0.97 | 59,350 | 49.3% | 50.6% | -1.2pp | 13,462 | 50.1% | 49.7% | +0.4pp | — |
| -0.5 ≤ z ≤ 0.5 | +0.02 | 65,388 | 49.8% | 50.0% | -0.2pp | 16,873 | 49.4% | 50.5% | -1.1pp | — |
| 0.5 < z ≤ 1.5 | +0.97 | 61,581 | 48.6% | 51.1% | -2.5pp | 14,392 | 48.7% | 51.2% | -2.5pp | — |
| 1.5 < z ≤ 2.5 | +1.87 | 23,125 | 50.1% | 49.6% | +0.5pp | 4,628 | 50.3% | 49.5% | +0.8pp | — |
| z > 2.5 | +2.84 | 3,959 | 47.4% | 52.2% | -4.8pp | 1,231 | 49.5% | 50.2% | -0.7pp | — |
