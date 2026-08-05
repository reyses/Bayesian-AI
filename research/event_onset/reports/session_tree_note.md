# Correction: the "scale-invariant 47%" was a uniform distribution

Reported that the dominant swing sits at 0.47-0.49 of segment duration at
every depth, calling it scale-invariant structure. Checked against the null:

Uniform(0,1) gives median 0.50, IQR [0.25, 0.75]. Observed at depth 5-6:
median 0.470-0.474, IQR [0.23, 0.74]. KS statistic 0.034-0.040 — the
p-values only clear 0.05 because n is 2,552-3,451.

**That is a uniform distribution described as a finding. Withdrawn.**

What survives:
- The SIZE cascade is real: 407 -> 306 -> 212 -> 144 -> 105 -> 82 -> 69pt,
  0.75x span and ~2x count per level.
- Depths 0 and 1 are genuinely non-uniform (KS 0.291 p=3e-9; KS 0.117
  p=0.003; medians 0.62 and 0.56) — at daily and half-daily scale the
  dominant move leans LATE. This is the only positional structure present.
- 2,588 segments remain >= 50pt at max depth: the recursion is truncated by
  MAX_DEPTH, not exhausted by the tape.
