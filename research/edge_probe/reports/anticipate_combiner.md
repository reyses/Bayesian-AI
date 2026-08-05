# Anticipate-the-combiner probe (full ATLAS, walk-forward OOS)
203,602 bars, 520 days, 22 streams + z_se. Can the combiner fire/direction be called N bars early?

## A. FIRE anticipation — AUC(fire within next H bars)
| H (bars early) | AUC with P | AUC streams+z_se ONLY (no P) | base rate |
|---|---|---|---|
| 2 | 0.808 | 0.706 | 44.1% |
| 3 | 0.802 | 0.702 | 52.3% |
| 5 | 0.783 | 0.692 | 63.6% |
| 8 | 0.770 | 0.690 | 74.5% |

## B. DIRECTION anticipation — AUC(upcoming fire is LONG) on pre-fire bars
| H | AUC with P | AUC no P | n pre-fire bars |
|---|---|---|---|
| 2 | 0.926 | 0.897 | 76,861 |
| 3 | 0.912 | 0.879 | 90,352 |
| 5 | 0.885 | 0.848 | 108,343 |
| 8 | 0.861 | 0.822 | 125,081 |

Read: AUC>0.6 (no-P) at H>=3 => the streams+regression carry genuine EARLY anticipation of the combiner (not just reading a near-threshold P) => qwen anticipation is worth building. ~0.5 no-P => the combiner is not anticipatable from its ingredients; the fire is the information, and anticipating it is a mirage.
