---
name: signed-mfe-pivot
description: For direction prediction on regret-oracle trades, target MUST be signed_mfe (mfe_dollars × direction sign). mfe_dollars alone hides direction signal — features that predict direction look invisible (R²≈0). Discovered 2026-05-16; R² jumped 0.187→0.262 on same features.
metadata:
  type: feedback
---

**Rule:** For ANY direction prediction or direction-discrimination work on
the regret-oracle daisy-chain trades, use
`signed_mfe = mfe_dollars × (+1 if LONG else −1)` as the regression/cluster
target. NOT `mfe_dollars` alone.

**Why:** `mfe_dollars` is the magnitude of forward excursion in the trade's
chosen direction. LONG trades and SHORT trades BOTH produce positive
mfe_dollars (it's the absolute excursion). Direction information is
collapsed out of the target. Features that predict direction (slope sign,
z-score sign, rail-position relative to price) appear invisible:

  - `slope_15s_3m` on `mfe_dollars`: R² = 0.002 (invisible)
  - `slope_15s_3m` on `signed_mfe`: R² ≈ 0.20 (dominant direction predictor)

A coefficient of +$X/σ on signed_mfe means "1σ increase in feature → $X
more LONG-skewed outcome." That's what direction work needs.

**How to apply:**

- For 1D, paired, triplet, k-way regression on direction: target = signed_mfe.
- For clustering by direction-archetype: target column for cluster stats = signed_mfe.
- Cells with `|mean_signed_mfe|` high AND pct_long far from 50% are
  direction-callable.
- `bar_range` and `volume` are MAGNITUDE amplifiers (symmetric across
  direction) — they show up strong on mfe_dollars and weak on signed_mfe
  with the same dataset. Use mfe_dollars when you specifically want to find
  magnitude-amplifier features; use signed_mfe for direction predictors.

**Evidence (2026-05-16 sleep run):**
- k=2 R² jumped 0.187 (mfe_dollars) → 0.262 (signed_mfe) on same features.
- Direction-callable cells (Wilson CI excludes 30/70%) at 43-59% rate.
- Per-cell accuracy 82-86% in callable cells, 93% in extreme cells.

**Where it sits in the arc:** Layer 2 (direction discrimination) of
[[regret-six-layer-architecture]]. Findings doc:
`reports/findings/regret_oracle/2026-05-16_direction_signal_kway.md`.
