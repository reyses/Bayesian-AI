# Exit Dojo -- full-run synthesis (N=200, gate-audited stepwise-blind)

Nonce-chain audit PASS on all 200 (no agent saw a future frame). MODE-FIRST distributions; bootstrap CIs (4000) on the captured-minus-5m-hold delta with an explicit significance call. **Leakage note is moot here** -- play was blind by construction -- but the graduation firewall still holds: a rule confirmed here must pass the sealed 2024/2025-26 harness before belief.

## Per-regime capture (points; mode-first)
| regime | N | cap mode | cap median | cap mean | 5m-hold median | delta mean (cap-5m) | delta 95% CI | beat-5m rate | oracle-ratio median |
|---|---|---|---|---|---|---|---|---|---|
| winner | 60 | +5.0 | +28.88 | +39.30 | +9.50 | +19.51 | [+8.34,+32.13] * | 63% | +0.29 |
| midflip | 60 | -17.0 | +26.75 | +35.12 | +25.88 | +0.04 | [-10.54,+11.00] | 35% | +0.57 |
| instantfail | 40 | -15.0 | -7.62 | -11.06 | +7.75 | -9.57 | [-15.49,-3.65] * | 20% | +0.30 |
| chop | 40 | -5.0 | -3.38 | -1.23 | -0.75 | -0.25 | [-2.92,+2.68] | 32% | -0.28 |
| **ALL** | 200 | -5.0 | +5.62 | +19.87 | +7.50 | +3.90 | [-1.03,+9.34] | 40% | -- |

_`*` = 95% CI excludes 0 (delta significant). delta = agent capture minus the fixed-5-minute-hold capture, per episode._

## Wrong-side (instantfail) exit speed
- N=40; exit-%ile-of-window median **0.33** (mode 0.25); lower = faster bail. Share bailing in the first third of the window: 55%.

## Grammar citation audit (EXIT-frame reasons, N EXIT commits)
Binding-EXIT reasons collected: 192 (episodes that force-held to the end have no EXIT reason).
| signal cited | episodes | share of exits |
|---|---|---|
| against-fires (multi) | 168 | 88% |
| giveback | 131 | 68% |
| PROPP / prop-turn | 114 | 59% |
| HA (heikin) | 93 | 48% |
| KMDR | 88 | 46% |
| ER10 / efficiency | 76 | 40% |
| CLIMAX | 36 | 19% |
| confluence / stack | 35 | 18% |
| vol / volatility | 16 | 8% |
| bar close / extreme | 6 | 3% |

_Which live signals the blind agents actually invoked to justify exits -- the empirical vocabulary of the exit grammar, to seed EXIT-GRAMMAR-01 priors._