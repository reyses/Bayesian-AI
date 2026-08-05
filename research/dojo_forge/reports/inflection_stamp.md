# Causal inflection stamp — out-of-sample validation

Rule: after **R consecutive seconds of positive acceleration**, stamp the first non-positive one (mirrored for lows). Velocity and acceleration on 1s closes over K-second windows. Strictly causal.

NOT a prediction test — the same probe showed acceleration turns AT the extreme, not before it. The claim under test is only that the stamp RECOGNISES a leg top within ±2s of its formation, earlier than a stall detector can.

Ground truth: close-based zigzag R=15pt on 1s (self-tested on a synthetic sawtooth). Sessions: **60**, RTH only, ~343 hours of tape.
Excluded: 2024_09_16. The live leg that motivated this is NOT in the sample.

| K | R | fires | fires/hr | precision | recall | median timing |
|---|---|---|---|---|---|---|
| 3 | 3 | 248,317 | 723.0 | `0.022` | `0.676` | +1s |
| 3 | 4 | 113,722 | 331.1 | `0.024` | `0.351` | +1s |
| 3 | 5 | 55,016 | 160.2 | `0.026` | `0.182` | +1s |
| 3 | 6 | 25,474 | 74.2 | `0.028` | `0.090` | +1s |
| 3 | 8 | 3,966 | 11.5 | `0.037` | `0.019` | +1s |
| 5 | 3 | 182,908 | 532.6 | `0.021` | `0.496` | +1s |
| 5 | 4 | 153,648 | 447.4 | `0.022` | `0.424` | +1s |
| 5 | 5 | 132,858 | 386.8 | `0.023` | `0.379` | +1s |
| 5 | 6 | 73,963 | 215.4 | `0.026` | `0.247` | +1s |
| 5 | 8 | 25,473 | 74.2 | `0.031` | `0.098` | +1s |
| 8 | 3 | 143,237 | 417.1 | `0.020` | `0.356` | +1s |
| 8 | 4 | 121,624 | 354.1 | `0.020` | `0.305` | +1s |
| 8 | 5 | 105,987 | 308.6 | `0.020` | `0.268` | +1s |
| 8 | 6 | 93,612 | 272.6 | `0.021` | `0.243` | +1s |
| 8 | 8 | 74,838 | 217.9 | `0.022` | `0.208` | +1s |

**Best precision: K=3, R=8 → `0.037` (11.5 fires/hr, recall 0.019).**

That is close to what random stamping would achieve. **The rule does not survive out of sample** — the live leg was a coincidence, exactly as the single-observation caveat warned.

## Baseline

With ~1 pivot per leg and TOL of ±2s, a stamp fired at a uniformly random second would hit at roughly (2·TOL+1)·pivots/seconds — typically well under 0.02. Compare every precision above against that, not against 0.5.

