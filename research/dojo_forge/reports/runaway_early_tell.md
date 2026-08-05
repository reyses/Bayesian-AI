# Is a runaway visible in the first seconds? (1s resolution)

Owner's claim: the mechanical harvest loses only because the decision needs to be split-second. This tests whether the information is actually there below 5s, where we have never looked.

Band touches reproduced exactly as in `oscillation_harvest_test.py` (cubic 5s w90, ±1.5σ, edge-triggered, RTH). Features use ONLY the first H seconds of 1s tape; the outcome is what happens after. **Trades already resolved inside H are dropped** — predicting a runaway that already happened is not prediction.
AUC: 0.5 = coin flip. Label 1 = runaway.

Sessions sampled: **124**

| H (s) | N | runaway rate | drift | mae | frac_adv | rng | accel |
|---|---|---|---|---|---|---|---|
| 5 | 12366 | 19.4% | `0.541` | `0.670` | `0.554` | `0.721` | `0.593` |
| 10 | 12344 | 19.1% | `0.578` | `0.698` | `0.592` | `0.744` | `0.536` |
| 20 | 12144 | 18.5% | `0.618` | `0.718` | `0.637` | `0.752` | `0.540` |
| 30 | 11932 | 17.9% | `0.636` | `0.727` | `0.665` | `0.751` | `0.528` |

**Strongest signal: `rng @ 20s`, AUC `0.752`.**

That is a real separation. A watcher CAN see it coming; the question becomes how much of the 2.56pt it converts.

## If the watcher exits on `rng` at 20s

| cut at percentile | trades cut | of those, runaways | mean net all (pt) |
|---|---|---|---|
| p60 | 40.2% | 32.9% | `-2.68` |
| p70 | 30.5% | 36.2% | `-2.36` |
| p80 | 20.5% | 39.8% | `-1.92` |
| p90 | 10.4% | 45.3% | `-1.39` |

Baseline (no early exit): `-0.39pt`. An early-exit rule only helps if it beats that — and it pays the adverse move already incurred on every trade it cuts, including the ones that would have won.

