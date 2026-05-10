---
name: 5s level is inherently noise — segmentation substrate, not prediction target
description: At 5s/30s scale, market action is dominated by stochastic micro-fluctuations; predictions should anchor at measure (15s) or coarser, not note (5s)
type: feedback
---

# 5s LEVEL IS INHERENTLY NOISE

User direction (verbatim, 2026-05-10 morning): "remember it might not be useful
to go down to 5s TF since it is inherently noise."

Confirmed empirically the same morning: when reclassifying the 9,561
NOISE-after-STEEP_LINEAR_DOWN cell at note level against the full 20-primitive
SeedPrimitiveLibrary, **53.8% remained NOISE residual** even after broadening
the template library from 13 to 20 shapes. P(fwd_up) was also uniform across
all sub-shapes (~0.31-0.45) — inner geometry didn't carry predictive content,
only the parent measure shape did.

## Rule

The 5-level segmentation hierarchy provides note (5s) as a SEGMENTATION
PRIMITIVE — it produces the chord vector that conditions Bayesian-table
lookups — but predictions should NOT be anchored at the note level.
Instead, anchor predictions at:
- **measure (15s)** — first level where simple-shape primitives carry
  signal that survives sub-shape decomposition
- **sub_motif (1m)** — most prediction-rich level (e.g.
  STEEP_CONCAVE_UP within STEEP_LINEAR_UP at sub_motif → 68.4% UP, n=321)
- **motif (5m)** — strategic-context level (FLATLINE-after-rally → 74.6%, n=57)
- **phrase (15m)** — day-shape framing (low n, wide CIs)

## Why

At 5s scale, MNQ price moves ~0.25-1.0 ticks per bar; rolling 30s std
is dominated by tick-quantization and microstructure noise rather than
trader-driven directional flow. A note's "shape" at this scale is
mostly random walk over a smooth parent context. The 50%+ NOISE-residual
rate confirms there's no consistent geometric primitive to extract.

## How to apply

- **DO** use note-level shape as a chord component when looking up
  conditional probabilities at coarser levels
- **DON'T** train classifiers, fit models, or expect direction
  predictability at 5s/30s horizons
- **DON'T** quote tight CIs from large-n cells at note level (n=9,539
  CI=0.005) as if they imply predictability — they reflect ABUNDANCE
  of NOISE, not predictive structure
- **DO** treat "sub-shape distribution within a NOISE bucket is uniform
  across forward-direction" as evidence that the cell is actually about
  parent context, not inner shape

## Implication for substrate use

Drop the note level from prediction tables but keep it for chord
construction. The Bayesian-table lookup at-bar should be keyed by:
    (measure_shape, sub_motif_shape, motif_shape, phrase_shape)
NOT by note_shape directly. The note is a measurement, not a forecast.

When inspecting a "huge n / tight CI" finding at note level, the answer
to "does it survive OOS?" is less interesting than the answer to "does
the same finding hold at measure or sub_motif level?" If the parent
level shows the same effect, the parent is doing the prediction. If only
the note level shows it, it's likely a microstructure artifact.

## Related memories

- `memory/project_5level_segmentation_substrate.md` — the substrate this
  refines; treat note level as primitive, not predictor
- `memory/feedback_quantile_selection_overfit.md` — same large-n /
  tight-CI / fragile-OOS pattern at a different layer
