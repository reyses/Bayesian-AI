# LinReg slope filter — within-tier holdout validation

Generated: 2026-04-29 23:22

## Method

For each tier:
1. Sort trades by timestamp.
2. Split into train (first 70%) and test (last 30%).
3. Pick best `slope_skip_threshold` on train.
4. Apply that threshold to test (no peeking).
5. Compare PnL improvement on train vs test.

## Robustness verdict

- 4/6 tiers improve TOTAL PnL on test (vs baseline).
- 4/6 tiers improve PER-TRADE PnL on test (more honest).

- **Mixed result — only ship for tiers showing per-trade improvement on test.**

## Detailed table

```
        tier  n_total  train_n  train_baseline_pnl  T_picked  train_kept_n  train_filtered_pnl  train_improvement_total  train_improvement_per_kept_trade  test_n  test_baseline_pnl  test_kept_n  test_filtered_pnl  test_improvement_total  test_improvement_per_kept_trade  generalizes_total  generalizes_per_trade
   RIDE_CALM      637      445              2751.0       0.5           315             11866.5                   9115.5                         31.489406     192             1010.5          136             5025.0                  4014.5                        31.685509               True                   True
RIDE_AGAINST     1423      996             27618.0       0.5           758             35768.0                   8150.0                         19.458419     427            11712.0          319            15038.5                  3326.5                        19.714062               True                   True
FADE_AGAINST      352      246             14586.5       1.5           202             17971.5                   3385.0                         29.673106     106             9051.0           90             9860.5                   809.5                        24.174319               True                   True
    BASE_NMP     1195      836             11066.5       0.5           650             13658.0                   2591.5                          7.774868     359             8930.5          279            10501.0                  1570.5                        12.761948               True                   True
   FADE_CALM      365      255             29757.0       3.0           247             29996.5                    239.5                          4.749202     110            12884.0          110            12884.0                     0.0                         0.000000              False                  False
   KILL_SHOT      106       74              1968.0       5.0            74              1968.0                      0.0                          0.000000      32              603.5           32              603.5                     0.0                         0.000000              False                  False
```

## Caveat

This is a within-tier time-split, not a true IS/OOS. The original 'IS' 
and 'OOS' files come from different pipeline runs with different tier mixes,
so naive IS-train-OOS-test isn't possible.

This 70/30 split tests whether the threshold picked on early data still 
works on later data — a basic sanity check for time-stability.
