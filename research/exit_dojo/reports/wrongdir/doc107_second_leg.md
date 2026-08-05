# doc-107 redo — never-bail decomposed by leg/class

N=23,378, 282 days. never-bail terminal (ticks) + stop X=48 net-vs-never-bail.
never-bail mean terminal: **+11.76 t/ep**; never-bail advantage over stop48: **+3.39 t/ep** (doc-107: +3.39).

| class (leg) | N | share | NB mean term | NB SUM term (share of +) | stop net | NB adv from class |
|---|---|---|---|---|---|---|
| good_clean (1st-leg win) | 4,188 | 17.9% | +294.0 | +1,231,067 (48%) | +0.0 | -0.00 |
| good_dipped (2ND-LEG recovery) | 5,912 | 25.3% | +223.9 | +1,323,866 (52%) | -212.4 | +53.72 |
| wrong (rode to loss) | 11,615 | 49.7% | -196.2 | -2,279,308 (—) | +107.3 | -53.30 |
| dead_band (scratch) | 1,663 | 7.1% | -0.5 | -765 (—) | -41.8 | +2.98 |

## The trade-off, by leg
- **2nd-leg (good_dipped) false-profit wins**: 5,912 trades, mean +223.9t, TOTAL +1,323,866t — this is 52% of never-bail's positive terminal.
- **1st-leg (good_clean) genuine wins**: mean +294.0t, TOTAL +1,231,067t (48%).
- **wrong-class tail never-bail rides**: 11,615 trades, mean -196.2t, p5 -556, p1 -933, worst -4501t ($-2250). A stop caps these; never-bail does not.

Read: never-bail's edge over cutting comes from NOT stopping the good_dipped = the SECOND-LEG recoveries. Remove/curb them and the edge collapses. But those second-leg wins are the same oscillation as the wrong-class tail never-bail must ride to keep them — harvesting 2nd-leg false profit while eating the catastrophic runaway tail. Confirms the owner: doc-107's never-bail win is the second leg.
