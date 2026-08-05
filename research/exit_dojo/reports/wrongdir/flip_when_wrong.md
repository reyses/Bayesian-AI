# Flip-when-wrong — mechanical, doc-107 population

N=23,378. never-bail mean abs = +9.36 t/ep ($+4.68). Flip fires at first drawdown <= -X pts.

| X (pts, flip depth) | flip mean | Δ vs never-bail | 95% CI | %flipped | cut@X mean |
|---|---|---|---|---|---|
| 2 | -6.34 | -15.70* | [-29.99, -1.59] | 84% | +2.51 |
| 3 | -5.63 | -14.99* | [-28.97, -1.13] | 82% | +2.85 |
| 4 | -5.60 | -14.96* | [-28.80, -1.37] | 81% | +2.85 |
| 6 | -3.23 | -12.59 | [-26.44, +0.78] | 78% | +3.99 |
| 8 | -0.94 | -10.30 | [-23.12, +2.57] | 74% | +5.10 |
| 12 | +0.95 | -8.41 | [-20.67, +3.79] | 68% | +5.97 |

## By class at X=2pt (earliest flip) — flip mean abs (ticks)
| class | never-bail | flip@2 |
|---|---|---|
| good_clean | +291.6 | +237.3 |
| good_dipped | +221.5 | -327.8 |
| wrong | -198.6 | +78.5 |
| dead_band | -2.9 | -69.4 |

Read: flip should WIN on wrong (ride the real leg) and LOSE on good_dipped/good_clean (whipsawed out of recoveries/wins). Net beats never-bail only if the wrong-class capture > the good-class whipsaw. Earlier X (smaller) = flip sooner = less eaten before the ride. * = CI excludes 0. Caveat holds if Δ rises as X falls.
