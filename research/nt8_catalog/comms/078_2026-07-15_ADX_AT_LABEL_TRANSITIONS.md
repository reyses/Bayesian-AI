# What ADX says at the label transition — turn-confirmer with age-decay + late inversion
**Doc:** 078 · **Date:** 2026-07-15 · **Author:** Claude (executor) · **Status:** RESULTS
Question (Moises): what does the signal say during the label transition?

## Answer (1,359 in-label signals, 576 days, oracle-clocked from each label's start)
| mins since turn | N | agree CURRENT | agree PREVIOUS |
|---|---|---|---|
| 0–1 | 9 | **0.78** | 0.22 |
| 1–2 | 80 | **0.69** | 0.31 |
| 2–5 | 621 | **0.62** | 0.38 |
| 5–10 | 318 | 0.60 | 0.40 |
| 10–20 | 179 | **0.45** | 0.55 |
| 20–60 | 138 | **0.43** | 0.57 |
Late (>10 min) signals: 0.45 vs current but **0.55 vs the NEXT label**, firing a median
**16.4 min before the next turn**. Early (≤5 min): 0.63 current / 0.37 next (mirror).

## Reading
1. **ADX is a turn-CONFIRMER.** Its strongest statement comes right after a transition
   (0.69–0.78 within 2 min) and decays with pivot age. The pooled 0.58 (doc 077) hides
   this: an early 0.62–0.78 population + a late ~0.44 population.
2. **Past ~10 minutes it INVERTS**: late ADX signals point against the current label and
   weakly WITH the coming one — the next turn announcing itself early. Two usable modes:
   confirm the fresh turn (early), fade/anticipate (late).
3. **Caveat (binding):** the transition clock here is ORACLE-defined = hindsight. The live
   version needs a CAUSAL turn-clock — streaming zigzag pivot age — making the combiner
   feature ADX × pivot-age, not ADX alone. (Also 0-1min N=9: directional but thin.)

## Next
Build the causal pivot-age clock (streaming zigzag) and re-cut this table on it; if the
early/late structure survives the causal clock, ADX×age enters the stage-0 combiner.
