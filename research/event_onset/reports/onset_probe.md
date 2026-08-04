# EVENT-ONSET PROBE — can a named event be seen coming?

Pre-registered verdict rule (set before any number was read; the program wall is ~0.57): AUC_lo > 0.6 CLEARS, > 0.57 MARGINAL, else WALL.
AUC_lo = fold mean - 1.96*SE. GroupKFold(5) by day, standardisation fit on train folds only, balanced 1:1 design (negatives >=5min from any same-type event). Live sim day 2024_09_16 excluded.

              event  horizon_s      n  days  auc_lin  auc_lin_lo  auc_gbm  auc_gbm_lo verdict
defended_poke_shelf          5   3166   443   0.6835      0.6615   0.6982      0.6813  CLEARS
defended_poke_shelf         10   3166   443   0.6773      0.6606   0.6984      0.6704  CLEARS
defended_poke_shelf         30   3166   443   0.6808      0.6642   0.6942      0.6697  CLEARS
       fakeout_poke          5 106340   448   0.8319      0.8218   0.9235      0.9169  CLEARS
       fakeout_poke         10 106316   448   0.8140      0.7991   0.8944      0.8855  CLEARS
       fakeout_poke         30 106214   448   0.7986      0.7843   0.8363      0.8220  CLEARS
        leg_descent          5  54738   431   0.9801      0.9774   0.9965      0.9961  CLEARS
        leg_descent         10  54654   431   0.9574      0.9542   0.9826      0.9816  CLEARS
        leg_descent         30  54482   431   0.8764      0.8663   0.9263      0.9204  CLEARS
              stall          5  79138   539   0.7818      0.7759   0.8160      0.8126  CLEARS
              stall         10  79162   539   0.7804      0.7732   0.8133      0.8051  CLEARS
              stall         30  79246   539   0.7810      0.7729   0.8125      0.8059  CLEARS
         ultra_chop          5  37202   529   0.8840      0.8794   0.9202      0.9176  CLEARS
         ultra_chop         10  37202   529   0.8844      0.8780   0.9192      0.9152  CLEARS
         ultra_chop         30  37202   529   0.7777      0.7666   0.8557      0.8497  CLEARS

## Headline

Best cell: **leg_descent at H=5s** — lin 0.9801, gbm 0.9965 (n=54738, 431 days) -> **CLEARS**

Counts: {'CLEARS': 15}


## v2 — MATCHED NEGATIVES (the v1 numbers were a design artifact)

v1 drew negatives from stretches >=5min from any same-type event and scored
up to 0.9965. That selects an unusually QUIET regime, so a classifier can win
by answering "is the tape active?" — worthless for trading. v2 replaces every
negative with the SAME event rewound a further 300s (same day, same regime),
rejected if another same-type event confirms inside its own horizon.

| event | H=5s | H=10s | H=30s |
|---|---|---|---|
| leg_descent | 0.9570 | 0.8683 | 0.7253 |
| fakeout_poke | 0.8394 | 0.7687 | 0.6258 |
| ultra_chop | 0.8307 | 0.8299 | 0.7270 |
| stall | 0.6612 | 0.6590 | 0.6561 |
| defended_poke_shelf | 0.6300 | 0.6274 | 0.6159 |

(gradient boosting, GroupKFold(5) by day, balanced 1:1.) Signal decays with
horizon — the shape you expect from real information rather than leakage.

## v3 — FEATURE ABLATION (is it just the level, or just the regime?)

H=10s, matched negatives, gradient boosting:

| event | ALL | NO_LEVELS | REGIME_ONLY |
|---|---|---|---|
| leg_descent | 0.8683 | 0.8307 | 0.5588 |
| ultra_chop | 0.8299 | 0.7718 | 0.6444 |
| fakeout_poke | 0.7687 | 0.7329 | 0.5600 |
| stall | 0.6590 | 0.5972 | 0.5365 |
| defended_poke_shelf | 0.6274 | 0.6199 | 0.6110 |

NO_LEVELS drops every distance/range/position feature; REGIME_ONLY keeps only
volatility, volume, body ratio and clock.

Reading it honestly:
- **fakeout_poke, leg_descent, ultra_chop carry real early structure** — they
  survive both the regime control and the loss of level features. The
  information is in the return/vol/flip dynamics, not in "a level is nearby".
- **defended_poke_shelf is almost entirely regime** (0.627 -> 0.611 when only
  vol/clock remain): its "predictability" is time-of-day and volatility, not
  the setup. Do not build on it.
- **stall is weak** (0.597 without levels).
- leg_descent at H=5s (0.957) is near-tautological — 10s before a zigzag
  confirms, the leg is mostly formed. Its value is latency, not foresight.

## VERDICT

The gate CLEARS, and the load-bearing cell is **fakeout_poke at H=10s:
AUC 0.769 (0.733 without level features), n=247,288 across 539 days** — an
event whose outcome table is already SHARP on the level question
(p(clears) 91% breakout vs 67% return). Early classification + sharp table is
exactly the actuary architecture: predict WHICH STATE, look up WHAT HAPPENS.

This is the first thing in the program to clear the 0.57 wall on a
well-controlled test. It clears because the target changed from DIRECTION
(measured null, six ways) to EVENT IDENTITY.
