# Good-dip horizon — envelope + flip-beyond

N=23,378. Dip DEPTH = MAE (min favorable-signed drift), points.

## 1. Dip-depth envelope (MAE, pts) by class
| class | n | median | p75 | p90 | p95 | p99 | deepest |
|---|---|---|---|---|---|---|---|
| good_dipped | 5,912 | -17.2 | -31.0 | -51.8 | -67.2 | -113.4 | -221.2 |
| wrong | 11,615 | -48.0 | -79.5 | -121.2 | -156.8 | -256.9 | -1219.8 |

good_dipped recovery time (dip→breakeven): median 2m, p90 6m, p95 7m.

## 2. Separation — P(class | dip reached ≤ -D)
| D (pts) | N reached | P(good_dipped) | P(wrong) | P(wrong)/[gd+wrong] |
|---|---|---|---|---|
| 2 | 19,569 | 30.2% | 59.4% | 66.3% |
| 4 | 18,905 | 31.3% | 61.4% | 66.3% |
| 6 | 18,157 | 29.4% | 63.6% | 68.4% |
| 8 | 17,383 | 27.7% | 65.7% | 70.3% |
| 10 | 16,644 | 26.2% | 67.7% | 72.1% |
| 12 | 15,874 | 24.8% | 69.6% | 73.8% |
| 16 | 14,467 | 22.1% | 72.9% | 76.7% |
| 20 | 13,180 | 19.5% | 75.9% | 79.5% |
| 25 | 11,671 | 17.3% | 78.8% | 82.0% |
| 30 | 10,273 | 15.1% | 81.5% | 84.4% |
| 40 | 8,055 | 12.0% | 85.2% | 87.6% |

## 3. Depth-gated flip@D (flip at first dip≤-D) vs never-bail (+9.36 t/ep)
| D (pts) | flip mean | Δ vs never-bail | %flipped |
|---|---|---|---|
| 2 | -6.34 | -15.70 | 84% |
| 4 | -5.60 | -14.96 | 81% |
| 6 | -3.23 | -12.59 | 78% |
| 8 | -0.94 | -10.30 | 74% |
| 10 | -0.15 | -9.51 | 71% |
| 12 | +0.95 | -8.41 | 68% |
| 16 | +3.19 | -6.17 | 62% |
| 20 | +4.25 | -5.11 | 56% |
| 25 | +6.25 | -3.11 | 50% |
| 30 | +7.30 | -2.06 | 44% |
| 40 | +7.85 | -1.50 | 34% |

Best depth-gated flip: D=40pt (mean +7.85 t/ep vs never-bail +9.36). 
Read: if good_dipped MAE has a p95/p99 ceiling and P(wrong|reached -D) climbs to ~100% beyond it, that D is the flip horizon. If P(wrong) plateaus (<100%) due to the survivorship rebound, flipping beyond still whipsaws the deep-dip survivors — the envelope is soft, not a clean gate.
