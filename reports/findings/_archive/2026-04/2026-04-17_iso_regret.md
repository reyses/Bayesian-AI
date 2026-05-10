# Regret Analysis on Isolated Tiers
Date: 2026-04-17
Source: `training/output/isolated/*.pkl` (from `tools/run_tier_isolated.py`)

## How to read this
- **Actual**: what the tier made as-is (from isolated run).
- **Optimal**: sum of best counterfactual per trade — perfect oracle.
- **Capture**: actual/optimal. Low = exits leak, or direction wrong.
- **% counter**: share of trades where regret says flip direction.
  - ~50% = tier has no directional edge (coin flip). DEAD.
  - 25-35% = real edge, CNN can extract it.
  - <20% = direction mostly right; problem is exits.
- **Counter WR**: actual win rate of counter-labeled trades. If regret is sound, should be much lower than 50%.
- **If flipped at peak**: upper-bound oracle $/day if we obeyed regret labels perfectly (peak capture on counters).
- **If flipped at exit**: realistic bound — flip direction but keep the same exit timing (no peak-chasing).

## Per-tier summary

| Tier | N | $/day actual | $/day optimal | Capture % | % counter | Counter WR | $/day if flip@peak | $/day if flip@exit |
|---|---|---|---|---|---|---|---|---|
| RIDE_AGAINST | 39,721 | $-11 | $+35261 | -0% | 49% | 41% | $+17794 | $+850 |
| FADE_CALM | 24,039 | $-16 | $+21078 | -0% | 49% | 38% | $+10818 | $+939 |
| MTF_BREAKOUT | 5,961 | $+4 | $+6409 | 0% | 50% | 39% | $+3240 | $+115 |
| KILL_SHOT | 4,411 | $-2 | $+3861 | -0% | 50% | 45% | $+2025 | $+84 |
| FADE_AGAINST | 4,532 | $+5 | $+4050 | 0% | 46% | 38% | $+1957 | $+190 |
| FREIGHT_TRAIN | 34 | $+61 | $+3752 | 2% | 50% | 35% | $+1846 | $+382 |
| CASCADE | 1,270 | $+6 | $+1228 | 0% | 52% | 39% | $+656 | $+49 |
| MTF_EXHAUSTION | 233 | $+9 | $+593 | 1% | 48% | 36% | $+260 | $+23 |

## Verdict per tier (my read)

- **RIDE_AGAINST**: DEAD — 49% counter-flip means no directional edge. Kill or rebuild entry.
- **FADE_CALM**: DEAD — 49% counter-flip means no directional edge. Kill or rebuild entry.
- **MTF_BREAKOUT**: DEAD — 50% counter-flip means no directional edge. Kill or rebuild entry.
- **KILL_SHOT**: DEAD — 50% counter-flip means no directional edge. Kill or rebuild entry.
- **FADE_AGAINST**: DEAD — 46% counter-flip means no directional edge. Kill or rebuild entry.
- **FREIGHT_TRAIN**: DEAD — 50% counter-flip means no directional edge. Kill or rebuild entry.
- **CASCADE**: DEAD — 52% counter-flip means no directional edge. Kill or rebuild entry.
- **MTF_EXHAUSTION**: DEAD — 48% counter-flip means no directional edge. Kill or rebuild entry.

## Best-action breakdown per tier

### FADE_CALM
- counter_extended: 11772 (49%)
- same_extended: 11730 (49%)
- same_early: 373 (2%)
- same_at_exit: 61 (0%)
- counter_early: 60 (0%)
- counter_at_exit: 43 (0%)

### RIDE_AGAINST
- same_extended: 19814 (50%)
- counter_extended: 19357 (49%)
- same_early: 311 (1%)
- counter_early: 119 (0%)
- same_at_exit: 85 (0%)
- counter_at_exit: 35 (0%)

### KILL_SHOT
- counter_extended: 2196 (50%)
- same_extended: 2142 (49%)
- same_early: 39 (1%)
- counter_early: 14 (0%)
- counter_at_exit: 11 (0%)
- same_at_exit: 9 (0%)

### CASCADE
- counter_extended: 657 (52%)
- same_extended: 586 (46%)
- same_early: 16 (1%)
- same_at_exit: 7 (1%)
- counter_at_exit: 2 (0%)
- counter_early: 2 (0%)

### FADE_AGAINST
- same_extended: 2354 (52%)
- counter_extended: 2052 (45%)
- same_early: 85 (2%)
- same_at_exit: 19 (0%)
- counter_early: 15 (0%)
- counter_at_exit: 7 (0%)

### MTF_BREAKOUT
- counter_extended: 2980 (50%)
- same_extended: 2943 (49%)
- same_early: 21 (0%)
- same_at_exit: 8 (0%)
- counter_early: 7 (0%)
- counter_at_exit: 2 (0%)

### MTF_EXHAUSTION
- same_extended: 121 (52%)
- counter_extended: 111 (48%)
- same_early: 1 (0%)

### FREIGHT_TRAIN
- same_extended: 17 (50%)
- counter_extended: 17 (50%)


## Aggregate — all tiers pooled

- Trades: 80,201 across 348 days
- Actual: $-3,348 ($-10/day)
- Optimal: $+24,363,846 ($+70011/day)
- If flipped at peak (oracle): $+12,362,421 ($+35524/day)
- If flipped at exit (realistic): $+759,804 ($+2183/day)
