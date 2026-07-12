# ATR-09 Approach-Ladder, Time-Ordered Forward Split (Claude-executed)
**Doc:** 037 · **Date:** 2026-07-11 · **Author:** Claude (executor) · **Status:** FINAL

## Design (per Moises, null-free; doc-017 "approach" anchor)
- Features: entry bar + lead-in lags [0,3,6,12,24,48] 5s-bars (0s..4min BEFORE t(e))
  across all 6 V2 tiers = 312 dims. Captures HOW the state approaches the entry.
- Both years pooled (799 events, 391 days).
- Validation = TIME-ORDERED forward split (train earliest 60% of days -> 2025-02-21,
  test the forward 40% it never saw). No surrogate null (rejected per MVP §5 / Moises:
  a near-copy surrogate leaks the response then falsely reads as failure).
- Model: L1-CV select on train, freeze thresholds on train, evaluate forward.

## Result (forward test, 319 events, 17 features selected)
| Branch | N (days) | win% | EV pts | day-block CI | mode |
|---|---|---|---|---|---|
| ACT (take the snap-back bet) | 34 (34) | 0.03 | -8.2 | [-15.1, +3.0] | -10 |
| INVERT (ride instead) | 69 (53) | **0.94** | +4.7 | [-8.1, +14.3] | **+12** |

## Read
- **Structure holds FORWARD by frequency**: when entry F-space predicts the snap-back
  will NOT come, riding wins 94% of the time forward, mode +12 pts (above friction).
  The lead-in lags contributed (17/312 selected span multiple lags) -> the APPROACH
  dynamics matter, supporting the fuller multi-TF ladder.
- **But the mean EV CI crosses 0**: a fat LEFT tail (rare rides that get run over)
  eats the average of many +12 wins. High-hit-rate structure WITH tail risk, not a
  clean high-EV edge. Tradable-relevant: take it + manage the tail, don't bank the mean.
- Better than the single-bar snapshot (which failed year-swap) -> lead-in helps.

## Binding next step (kills the last approximation)
INVERT EV here is still the MIRROR approx (-magnitude of the snap-back bet). The money
question: SIMULATE the actual ride trade (enter opposite at t(e) when P(response) low,
exit on a real stop/target), measure its true forward distribution with the tail
managed. If the ride-with-a-stop pays forward, it is takeable.

Artifact: `tools/ag_phase5_approach_atr09.py`. Committed this turn.
