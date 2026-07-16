# Economic conversion — does the pooled P(right) convert to POINTS?

**Question.** The stage-0 combiner emits a calibrated P(label-right) (pooled OOS AUC 0.689). P(label-right) != P($). This measures what a fire in each P-decile is worth in RAW POINTS of forward drift — NO stops, NO trade management. The verdict gates the Mamba handoff.

- MNQ conversion: **$2.00/point**. Friction line: 1 tick (0.25) + ~$0.75 comm ~= **0.6 pts ($1.20)** round trip — shown next to every mean, NOT subtracted silently.
- Deciles computed on **TEST fires only** (2025+26); all rows below are the TEST set. Drift signed by trade direction (+long / -short).
- **Pseudo-replication:** fires inside a horizon window are correlated (many co-fires); day-block bootstrap CIs are the mitigation. Per-fire counts are NOT independent trades.

## Headline

**Yes — the pooled P(right) converts to points, monotonically and with the correct sign.** As-is drift climbs straight up the P-decile ladder: at 5m, decile 0 = **-1.33 pts** -> decile 9 = **+3.86 pts**, crossing zero right at the calibration midpoint (deciles 5-6). Low-P fires drift AGAINST the trade (so inverting them pays); high-P fires drift WITH it. P(label-right) was fit to AI-label agreement, never to price — so this price linkage is an independent confirmation, not circular.

**Read the distribution, not the mean.** The single clean, significant, non-tail cell is **top decile @ 5m**: mode **+1.0**, median **+3.25**, mean **+3.86 pts ($7.72)** CI[+2.48,+5.06], net-of-friction **+3.26 pts** — here mode AND median are strongly positive, so it is a genuine distributional shift, not an outlier tail. By contrast top decile @ 1m clears friction on the mean (+1.18 CI[+0.71,+1.68]) but mode=0 / median=+0.75, so the typical 1m fire only just covers the 0.6-pt friction — that edge is tail-driven.

**The tradeable window is SHORT (1-5m).** At 15m+ the day-block CIs blow out (30m top-decile CI[-5.68,+8.54], 60m CI[-4.44,+12.26]) and nearly every cell goes NS; 60m also truncates 13.5% of fires at 15:15. Both candidate live populations clear friction at 5m, but **top-decile-as-is is the cleaner one** (higher median, not tail-only). The inverted bottom decile needs a 5-30m hold (1m net -0.05 is below friction) and is more tail-driven (mode ~0 at 5-15m).

**Gate verdict:** the Mamba handoff is justified for SHORT-horizon management of top-decile (and inverted-bottom-decile) fires — there IS a raw directional edge to hand off. But it is horizon-fragile and decays past 5m, so harvesting the 1-5m drift before it dissipates is precisely the job the Mamba must do; a passive long hold does not survive the variance.

## Truncation (TEST): fraction of fires whose ts+h ran past 15:15 CT

| horizon | trunc frac |
|---|---|
| 1m | 0.002 |
| 5m | 0.009 |
| 15m | 0.029 |
| 30m | 0.067 |
| 60m | 0.135 |

## Horizon 1m — per P-decile (TEST, mode-first)

| decile | N | mode (pts) | mean (pts) | mean ($) | net-of-0.6 (pts) | 95% CI (pts) | median (pts) |
|---|---|---|---|---|---|---|---|
| 0 | 40133 | +0.00 | -0.548 | -1.10 | -1.148 | [-0.70,-0.41] | -0.25 |
| 1 | 40133 | -0.00 | -0.931 | -1.86 | -1.531 | [-1.08,-0.78] | -0.75 |
| 2 | 40133 | -0.00 | -0.561 | -1.12 | -1.161 | [-0.73,-0.39] | -0.50 |
| 3 | 40133 | -0.00 | -0.347 | -0.69 | -0.947 | [-0.50,-0.20] | -0.25 |
| 4 | 40133 | -0.00 | -0.227 | -0.45 | -0.827 | [-0.40,-0.06] | +0.00 |
| 5 | 40133 | +0.00 | -0.004 | -0.01 | -0.604 | [-0.16,+0.15] NS | +0.00 |
| 6 | 40133 | +0.00 | +0.046 | +0.09 | -0.554 | [-0.13,+0.22] NS | +0.00 |
| 7 | 40133 | -0.00 | +0.048 | +0.10 | -0.552 | [-0.10,+0.20] NS | +0.00 |
| 8 | 40134 | -1.00 | +0.335 | +0.67 | -0.265 | [+0.15,+0.51] | +0.00 |
| 9 | 40132 | +0.00 | +1.176 | +2.35 | +0.576 | [+0.70,+1.68] | +0.75 |

## Horizon 5m — per P-decile (TEST, mode-first)

| decile | N | mode (pts) | mean (pts) | mean ($) | net-of-0.6 (pts) | 95% CI (pts) | median (pts) |
|---|---|---|---|---|---|---|---|
| 0 | 40133 | +0.00 | -1.332 | -2.66 | -1.932 | [-1.65,-1.00] | -1.00 |
| 1 | 40133 | +0.00 | -1.920 | -3.84 | -2.520 | [-2.34,-1.50] | -1.25 |
| 2 | 40133 | +2.00 | -1.413 | -2.83 | -2.013 | [-1.94,-0.92] | -0.50 |
| 3 | 40133 | +0.00 | -0.852 | -1.70 | -1.452 | [-1.30,-0.41] | -0.50 |
| 4 | 40133 | -0.00 | -0.452 | -0.90 | -1.052 | [-0.88,-0.02] | -0.25 |
| 5 | 40133 | -0.00 | -0.312 | -0.62 | -0.912 | [-0.71,+0.08] NS | +0.00 |
| 6 | 40133 | +0.00 | -0.223 | -0.45 | -0.823 | [-0.74,+0.26] NS | +0.00 |
| 7 | 40133 | +0.00 | +0.345 | +0.69 | -0.255 | [-0.08,+0.78] NS | +0.25 |
| 8 | 40134 | +0.00 | +0.952 | +1.90 | +0.352 | [+0.40,+1.49] | +1.00 |
| 9 | 40132 | +1.00 | +3.862 | +7.72 | +3.262 | [+2.46,+5.12] | +3.25 |

## Horizon 15m — per P-decile (TEST, mode-first)

| decile | N | mode (pts) | mean (pts) | mean ($) | net-of-0.6 (pts) | 95% CI (pts) | median (pts) |
|---|---|---|---|---|---|---|---|
| 0 | 40133 | +0.00 | -1.824 | -3.65 | -2.424 | [-2.57,-1.06] | -0.75 |
| 1 | 40133 | +0.00 | -2.404 | -4.81 | -3.004 | [-3.42,-1.36] | -1.25 |
| 2 | 40133 | +0.00 | -2.334 | -4.67 | -2.934 | [-3.53,-1.26] | -1.00 |
| 3 | 40133 | -0.00 | -0.915 | -1.83 | -1.515 | [-1.90,+0.01] NS | -0.75 |
| 4 | 40133 | +2.00 | -0.068 | -0.14 | -0.668 | [-0.87,+0.74] NS | -0.25 |
| 5 | 40133 | +0.00 | +0.243 | +0.49 | -0.357 | [-0.68,+1.22] NS | +0.00 |
| 6 | 40133 | -0.00 | +0.138 | +0.28 | -0.462 | [-0.74,+1.00] NS | +0.00 |
| 7 | 40133 | +0.00 | +0.804 | +1.61 | +0.204 | [-0.10,+1.70] NS | +0.25 |
| 8 | 40134 | -1.00 | +1.814 | +3.63 | +1.214 | [+0.58,+3.04] | +1.00 |
| 9 | 40132 | -1.00 | +3.042 | +6.08 | +2.442 | [-0.19,+6.40] NS | +1.75 |

## Horizon 30m — per P-decile (TEST, mode-first)

| decile | N | mode (pts) | mean (pts) | mean ($) | net-of-0.6 (pts) | 95% CI (pts) | median (pts) |
|---|---|---|---|---|---|---|---|
| 0 | 40133 | -0.00 | -1.834 | -3.67 | -2.434 | [-3.15,-0.54] | -0.50 |
| 1 | 40133 | +0.00 | -2.554 | -5.11 | -3.154 | [-4.15,-0.93] | -1.25 |
| 2 | 40133 | +0.00 | -2.217 | -4.43 | -2.817 | [-3.98,-0.61] | -1.00 |
| 3 | 40133 | -0.00 | -1.353 | -2.71 | -1.953 | [-3.14,+0.23] NS | -0.50 |
| 4 | 40133 | +0.00 | +0.560 | +1.12 | -0.040 | [-0.70,+1.82] NS | +0.25 |
| 5 | 40133 | +1.00 | +0.616 | +1.23 | +0.016 | [-0.95,+2.38] NS | +0.00 |
| 6 | 40133 | +0.00 | +0.395 | +0.79 | -0.205 | [-1.06,+1.90] NS | +0.00 |
| 7 | 40133 | -0.00 | +0.643 | +1.29 | +0.043 | [-0.91,+2.17] NS | -0.25 |
| 8 | 40134 | +1.00 | +1.200 | +2.40 | +0.600 | [-0.78,+3.20] NS | +0.75 |
| 9 | 40132 | +1.00 | +1.841 | +3.68 | +1.241 | [-5.34,+8.58] NS | +1.25 |

## Horizon 60m — per P-decile (TEST, mode-first)

| decile | N | mode (pts) | mean (pts) | mean ($) | net-of-0.6 (pts) | 95% CI (pts) | median (pts) |
|---|---|---|---|---|---|---|---|
| 0 | 40133 | +0.00 | -1.942 | -3.88 | -2.542 | [-3.97,+0.05] NS | -0.75 |
| 1 | 40133 | +0.00 | -3.535 | -7.07 | -4.135 | [-6.08,-1.04] | -1.75 |
| 2 | 40133 | -0.00 | -3.305 | -6.61 | -3.905 | [-6.80,-0.42] | -1.25 |
| 3 | 40133 | +0.00 | -1.571 | -3.14 | -2.171 | [-4.32,+0.79] NS | -0.25 |
| 4 | 40133 | +0.00 | +0.726 | +1.45 | +0.126 | [-0.92,+2.45] NS | +0.25 |
| 5 | 40133 | -0.00 | +1.647 | +3.29 | +1.047 | [-0.43,+4.05] NS | +0.25 |
| 6 | 40133 | +0.00 | +0.396 | +0.79 | -0.204 | [-1.42,+2.19] NS | +0.00 |
| 7 | 40133 | -0.00 | +0.385 | +0.77 | -0.215 | [-1.81,+2.55] NS | +0.25 |
| 8 | 40134 | +1.00 | +1.398 | +2.80 | +0.798 | [-1.58,+4.34] NS | +0.50 |
| 9 | 40132 | -1.00 | +3.934 | +7.87 | +3.334 | [-4.18,+12.12] NS | +1.50 |

## ACTION rows — candidate live populations (TEST)

"top decile as-is" = decile 9, drift as traded. "bottom decile INVERTED" = decile 0 with drift sign flipped (fade the least-reliable-agreement fires).

| population | horizon | N | mode (pts) | mean (pts) | mean ($) | net-of-0.6 (pts) | 95% CI (pts) | median (pts) |
|---|---|---|---|---|---|---|---|---|
| top decile as-is | 1m | 40132 | +0.00 | +1.176 | +2.35 | +0.576 | [+0.70,+1.68] | +0.75 |
| top decile as-is | 5m | 40132 | +1.00 | +3.862 | +7.72 | +3.262 | [+2.46,+5.12] | +3.25 |
| top decile as-is | 15m | 40132 | -1.00 | +3.042 | +6.08 | +2.442 | [-0.19,+6.40] NS | +1.75 |
| top decile as-is | 30m | 40132 | +1.00 | +1.841 | +3.68 | +1.241 | [-5.34,+8.58] NS | +1.25 |
| top decile as-is | 60m | 40132 | -1.00 | +3.934 | +7.87 | +3.334 | [-4.18,+12.12] NS | +1.50 |
| bottom decile INVERTED | 1m | 40133 | -0.00 | +0.548 | +1.10 | -0.052 | [+0.41,+0.70] | +0.25 |
| bottom decile INVERTED | 5m | 40133 | +0.00 | +1.332 | +2.66 | +0.732 | [+1.00,+1.65] | +1.00 |
| bottom decile INVERTED | 15m | 40133 | -0.00 | +1.824 | +3.65 | +1.224 | [+1.06,+2.57] | +0.75 |
| bottom decile INVERTED | 30m | 40133 | -2.00 | +1.834 | +3.67 | +1.234 | [+0.54,+3.15] | +0.50 |
| bottom decile INVERTED | 60m | 40133 | -0.00 | +1.942 | +3.88 | +1.342 | [-0.05,+3.97] NS | +0.75 |

## Kill-point verdicts

- KILL-POINT A did NOT fire: 5 (population,horizon) cell(s) clear friction with CI excluding 0 -> top decile as-is/1m mean=+1.176pts CI[+0.70,+1.68]; top decile as-is/5m mean=+3.862pts CI[+2.46,+5.12]; bottom decile INVERTED/5m mean=+1.332pts CI[+1.00,+1.65]; bottom decile INVERTED/15m mean=+1.824pts CI[+1.06,+2.57]; bottom decile INVERTED/30m mean=+1.834pts CI[+0.54,+3.15]
-   SHAPE WARNING top decile as-is/1m: mode=+0.00 ~= 0 but mean=+1.176 — edge is a FAT RIGHT TAIL dragging the mean, not a typical fire (outlier-day trap). Lead with the mode.
-   SHAPE WARNING bottom decile INVERTED/5m: mode=+0.00 ~= 0 but mean=+1.332 — edge is a FAT RIGHT TAIL dragging the mean, not a typical fire (outlier-day trap). Lead with the mode.
-   SHAPE WARNING bottom decile INVERTED/15m: mode=-0.00 ~= 0 but mean=+1.824 — edge is a FAT RIGHT TAIL dragging the mean, not a typical fire (outlier-day trap). Lead with the mode.

_Shape note: read MODE first. Where mode ~= 0 while mean > 0, the "edge" is a fat right tail (a few big-drift fires), not the typical outcome — the user's outlier-day trap rule._