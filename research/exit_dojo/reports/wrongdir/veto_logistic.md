# The DISTILLED VETO -- 2024-sealed logistic at the 24-tick stop moment (Task 105)

Decision point: the minute the plain **24-tick** stop first triggers (favorable drift <= -6.0 pts). At that instant the logistic prices P(recover) = P(terminal drift > drift[t*]) from path-derivable features (<= t* only, causality asserted). Policy: **VETO the stop iff P(recover) >= p***.

## Frozen model (trained on 2024, split==train)
- train engagements scanned: 12917; 24t-stop triggered (examples): 9535; recover rate: 0.465
- LogisticRegression (standardized, C=1.0); frozen **p\* = 0.450**; intercept = -0.1423
- **discrimination: AUC in-sample 0.539, 5-fold CV 0.529** -- a 0.5-anchored gap of ~+0.029 is BELOW the 0.05 conditional-signal floor: the 9 cheap path features barely separate recover from no-recover. Coefficients (below) are noise-level.
- **economic asymmetry of the 24t stop on the NATURAL 2024 distribution**: taking the stop nets mean -11.7 t (median +5.0) -- when the trade RECOVERS it costs -133.2 t (huge forgone run), when it does NOT it saves +93.8 t. The recovery tail dominates, so a net-maximizing veto is driven toward **veto-almost-everything** (~ never-bail). The balanced 50/50 test set is the ONLY reason the plain stop looks like +17.7.

### Feature coefficients (standardized; sign = effect on P(recover))
| feature | coef | reading |
|---|---|---|
| minutes_since_entry | -0.0823 | higher -> more likely to KEEP LOSING |
| loss_velocity | -0.0644 | higher -> more likely to KEEP LOSING |
| giveback_velocity | -0.0644 | higher -> more likely to KEEP LOSING |
| acceleration | +0.0578 | higher -> more likely to RECOVER |
| tod | -0.0335 | higher -> more likely to KEEP LOSING |
| drawdown_vs_vol | +0.0295 | higher -> more likely to RECOVER |
| efficiency_ratio | -0.0268 | higher -> more likely to KEEP LOSING |
| entry_P | -0.0207 | higher -> more likely to KEEP LOSING |
| giveback_depth | -0.0064 | higher -> more likely to KEEP LOSING |

## 2024 p\* sweep (mean net-vs-never-bail ticks/ep over triggered train episodes)
| p* | n_veto | mean net | median net |
|---|---|---|---|
| 0.300 | 9520 | +0.05 | +0.00 |
| 0.325 | 9513 | +0.07 | +0.00 |
| 0.350 | 9490 | +0.07 | +0.00 |
| 0.375 | 9447 | +0.03 | +0.00 |
| 0.400 | 9305 | +0.10 | +0.00 |
| 0.425 | 8764 | +0.29 | +0.00 |
| 0.450 | 6546 | +0.59 | +0.00 | <- frozen
| 0.475 | 3065 | -2.91 | +0.00 |
| 0.500 | 1136 | -8.19 | +0.00 |
| 0.525 | 451 | -11.61 | +0.00 |
| 0.550 | 187 | -11.02 | +3.00 |
| 0.575 | 80 | -11.52 | +4.00 |
| 0.600 | 38 | -11.72 | +4.00 |
| 0.625 | 17 | -11.67 | +5.00 |
| 0.650 | 6 | -11.61 | +5.00 |
| 0.675 | 3 | -11.57 | +5.00 |
| 0.700 | 2 | -11.59 | +5.00 |
| 0.725 | 2 | -11.59 | +5.00 |
| 0.750 | 2 | -11.59 | +5.00 |
| 0.775 | 2 | -11.59 | +5.00 |
| 0.800 | 1 | -11.64 | +5.00 |
| 0.825 | 1 | -11.64 | +5.00 |
| 0.850 | 1 | -11.64 | +5.00 |
| 0.875 | 0 | -11.69 | +5.00 |
| 0.900 | 0 | -11.69 | +5.00 |

## TEST frontier (the 198 doc-100 episodes, scored ONCE)
net-vs-never-bail = (drift[exit] - drift[window]) x 4 ticks; mean +/- 95% day-block CI. Absolute = realized drift x 4 - friction (2.4t/RT), one round trip per episode.
| policy | mean net (ticks) | 95% day-block CI | median | mode | mean ABS w/friction |
|---|---|---|---|---|---|
| never-bail | +0.00 | [+0.00, +0.00] | +0.0 | +2.0 | +16.82 |
| plain stop 24t | +17.74 | [-12.36, +46.71] | +0.0 | +2.0 | +34.55 |
| STOP+VETO | -0.99 | [-3.06, +0.57] | +0.0 | +2.0 | +15.82 |
| blind agents (doc 100 ref) | +7.50 | (external) | - | - | - |
| plain stop 24t (doc 100 ref) | +17.70 | (external) | - | - | - |

## Pre-registered bar
**(1) STOP+VETO beats plain stop on net, delta CI excludes 0.** delta (stop+veto - plain stop) = -18.73 ticks/ep, 95% day-block CI [-47.51, +11.15] -> FAIL (CI includes 0).
**(2) dipped-good false-bail < 54% at equal-or-better wrong-catch.** STOP+VETO dipped-FB = 6% (N=48) vs plain-stop 90%; wrong-catch = 6% (N=100) vs agent ref 95% / plain-stop 100%.
  -> dipped-FB<54%: PASS; catch>=agent 95%: FAIL.

## VERDICT: **FAIL**
(at least one bar fails -- see above.)

## Per-class confusion (STOP+VETO vs plain stop, on the 198)
| policy | wrong-catch | dipped-FB | clean-FB | precision | n_bail |
|---|---|---|---|---|---|
| plain stop 24t | 100% | 90% | 0% | 70% | 143 |
| STOP+VETO | 6% | 6% | 0% | 67% | 9 |

## Veto precision / recall (among the plain-stop bails the veto cancels)
- triggered episodes: 143 (43 good, 100 wrong)
- vetoes fired: 134 (40 on GOOD = correct saves, 94 on WRONG = mistaken holds)
- veto precision P(good | vetoed) = 30%; veto recall P(vetoed | good-triggered) = 93%

_Sealed 2024 fit, frozen (p*, coefs), single pass on the 198. A dojo number is never a result until it clears the sealed frontier; this is that frontier for the distilled veto._