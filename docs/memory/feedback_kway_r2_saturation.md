---
name: kway-r2-saturation
description: Direction-prediction R² saturates around 0.35 on daisy-chain trades; k>3 with full bins is wasteful. Stratified k=2 matches unstratified k=5 with FAR fewer parameters. Don't escalate k blindly.
metadata:
  type: feedback
---

**Rule:** For direction prediction on the daisy-chain regret-oracle trade
set, R² saturates around 0.35. Don't escalate to k=4 or k=5 with full bins
expecting big gains. Stratify instead.

**Evidence (2026-05-16):**

| Method | R² |
|---|---|
| k=1 single feature on signed_mfe (slope_15s_3m) | ~0.20 |
| k=2 (5 bins, all pairs) | 0.262 |
| k=3 (5 bins) | 0.307 (+0.045) |
| k=4 (3 bins) | 0.320 (+0.013) |
| k=5 (2 bins) | 0.348 (+0.028) |
| **Stratified k=2 within bar_range S3** | **0.344** |
| **Stratified k=2 within tod_minutes S5** | **0.342** |

The 4-way and 5-way interaction terms add ~0 R². Signal is fully captured
by 1-way + 2-way + a stratifier.

**Stratified k=2 (2 features within a stratum) matches unstratified k=5
(5 features + all interactions)** — at far fewer parameters, less overfit
risk, easier to interpret. The user's "shaft from seeds" intuition
empirically verified — heterogeneous data masks subgroup-specific
direction signal.

**How to apply:**

For future feature-combination work on the regret oracle:
1. Start with 1D regression per feature (target = signed_mfe per [[signed-mfe-pivot]]).
2. Pair stratification next (k=2 with 5 bins).
3. **Stop escalating bin-count beyond k=3 with full bins.** Either reduce
   bin count (k=4 at 3 bins, k=5 at 2 bins) or stratify first.
4. Pick a primary feature to stratify on (`bar_range`, `tod_minutes`, or
   the regime-2d label). Run pair analysis WITHIN each stratum.
5. Accept ~0.35 R² as the natural ceiling for the V2-direction feature set
   on the daisy-chain trades. Pushing past requires structurally different
   features (e.g., trajectory shape — Layer 3 / [[bayesian-archetypes-pending]]).

**Caveat:** This ceiling is for the LINEAR-AND-LOW-ORDER-INTERACTION model
class on these specific features. Trajectory-based models (Layer 3 of
[[regret-six-layer-architecture]]) could break through.

**Multi-comparison caution:** With ~400k+ cells tested across all the
analyses, the top cells have selection bias. OOS validation is mandatory
before any selector uses them (per MEMORY hard rule).

See also [[signed-mfe-pivot]].
