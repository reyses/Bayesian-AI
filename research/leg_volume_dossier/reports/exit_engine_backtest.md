# LegExitEngine backtest vs never-bail (census ride eps)
148 ride episodes, 22 days.

| FLOOR/TIGHT | mean vs NB | 95% CI | exit rate | worst ep (NB) | worst ep (engine) |
|---|---|---|---|---|---|
| 50/15 | -7.9 | [-16.1, -0.5] | 84% | -132 | -82 |
| 50/20 | -9.2 | [-17.1, -2.0] | 79% | -132 | -82 |
| 40/12 | -10.1 | [-19.4, -1.6] | 91% | -132 | -82 |
| 60/20 | -8.5 | [-15.8, -2.1] | 70% | -132 | -103 |

Best config 50/15: mean -7.9 pts/ep vs never-bail (CI [-16.1, -0.5]).
TAIL: worst single episode never-bail -132 pts vs engine -82 pts — the floor caps the disaster by +50 pts.

Read honestly: if mean CI includes/below 0, the engine does NOT beat never-bail on average (expected — every component did not) but the tail column shows whether the catastrophic floor earns its small average cost as disaster insurance.
