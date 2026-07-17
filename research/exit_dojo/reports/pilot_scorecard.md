# Exit Dojo -- pilot scorecard

**Leakage caveat**: pilot episodes are played single-prompt (the agent receives all frames in one message with a sequential-commitment contract) -- attention CAN see future frames, so these scores are OPTIMISTIC and are used ONLY for hypothesis generation. Any discovered rule must be codified and pass the sealed 2024/2025-26 harness before belief. A true stepwise-blind runner is a later build if measured LLM performance is ever wanted.

| episode | type | agent exit (min) | captured (pts) | 5m-hold ref (pts) | oracle (label-end) ref (pts) | capture ratio |
|---|---|---|---|---|---|---|
| ep_01 | winner | 9 | +5.50 | -19.25 | +27.50 (@t=31) | +0.20 |
| ep_02 | winner | 31 | +40.75 | +8.25 | +55.50 (@t=27) | +0.73 |
| ep_03 | winner | 38 | +8.25 | -1.00 | +18.50 (@t=30) | +0.45 |
| ep_04 | midflip | 17 | +88.25 | +36.50 | +118.75 (@t=15) | +0.74 |
| ep_05 | midflip | 7 | +11.25 | +38.00 | +38.00 (@t=5) | +0.30 |
| ep_06 | midflip | 13 | +7.25 | -23.00 | +26.75 (@t=10) | +0.27 |
| ep_07 | instantfail | 7 | -65.25 | -54.50 | +0.00 (@t=0) | n/a |
| ep_08 | instantfail | 38 | +9.75 | -11.00 | +0.00 (@t=0) | n/a |
| ep_09 | chop | 6 | +2.25 | +4.25 | +4.50 (@t=4) | +0.50 |
| ep_10 | chop | none (forced) | +7.25 | -2.75 | +1.00 (@t=2) | +7.25 |

## Totals (N=10)
- mean captured: +11.53 pts | median: +7.75 pts
- mean 5m-hold ref: -2.45 pts | mean oracle(label-end) ref: +29.05 pts
- mean capture ratio (n=8 with a stable denominator): +1.31 | median: +0.47

_N=10 is a pilot sample for hypothesis generation, not a statistically powered claim -- no CI is reported (would be uninformatively wide at this N). See the leakage caveat above before acting on any of this._