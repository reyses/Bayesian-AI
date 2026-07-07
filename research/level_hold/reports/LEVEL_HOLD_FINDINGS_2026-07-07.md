# Level-Hold Study — findings (2026-07-07)

User observation: frozen band levels appear to act as recurring S/R
(examples/Figure_1.png). Causal test: freeze levels at t from data ≤ t,
count forward touches; symmetric barriers (RW null = 0.500 exactly);
jittered phantom levels calibrate the mean-reversion base rate.
6 days (5 IS + 2024_02_27), ~15-17k resolved touches per config.

## Results — P(hold | touch), real vs phantom

| barrier scale | real (pooled) | phantom (pooled) | verdict |
|---|---|---|---|
| 8 ticks (2 pt) | 0.61–0.63 | 0.62–0.64 | real ≈ phantom |
| 20 ticks (5 pt) | 0.58–0.63 | 0.58–0.61 | real ≈ phantom |
| 40 ticks (10 pt) | 0.51–0.59 | 0.50–0.57 | real ≈ phantom, both → 0.5 |

Only repeated positive cell: `slow_extreme` (60-min extreme of the hourly
band): real−phantom ≈ +0.031 (R20) / +0.036 (R40), CIs overlapping, while
`fast_extreme` flips NEGATIVE at R40 — with 8 family×scale cells, one
marginal +0.03 is expected by chance. **No level family clears the 0.05
significance bar against phantom.**

## The decomposition (what the eye actually sees)

1. **The bounce is real**: ~60-62% of touches reject at 2-5 pt barriers —
   massively above the 0.500 random-walk null at N≈17k. Price at 5s-1m scale
   mean-reverts hard (consistent with the oscillation framework's ~91%
   reversion).
2. **The lines are not special**: a phantom line 1-4 pts away from the real
   band level holds *just as often*. The level's specific VALUE (its anchor
   to band extremes / prior S/R) adds nothing measurable.
3. So watching price reject "your" level ~6 times in 10 is a true
   experience — but any nearby line would have delivered the same show. The
   tradeable content is the reversion itself, which the zigzag/L5 system
   already harvests, and at micro scale the 60/40 × 2pt edge is roughly
   eaten by round-trip costs.
4. Hold rates decay with scale (0.62 → 0.59 → 0.54): reversion is a
   short-range force; at 10-pt barriers the market is near coin-flip.

## Caveats

- Bands here are OLS ±2σ approximations of the NT8 1a/1b indicators (exact
  k/window may differ); tested on Databento IS days.
- Levels were tested IN ISOLATION. The user's practice is confluence:
  level + cubic approach-curvature + Markov prob together. Untested here:
  P(hold | touch, approach geometry) — if hold-rate spreads strongly by
  approach curvature, the edge lives in the CONFLUENCE (level = where,
  curvature = when/selection), which is the natural stage-0 detector target.

Raw outputs: level_hold_R8→(level_hold_results.txt initial run), _R20.txt,
_R40.txt. Tool: tools/level_hold_study.py.
