---
name: signal-threshold-magnitudes
description: "User's signal-magnitude bar: an edge/effect-size >= 0.10 = REAL signal; 0.05-0.10 = conditionally approved; below ~0.05 (e.g. 0.00079) = NOT signal / noise. Applies to edge metrics — AUC-over-0.5 gap, OOS R2, correlation, null-anchored gaps, normalized lift."
metadata: 
  node_type: memory
  type: feedback
  originSessionId: a1bfcbce-37ff-4c61-931f-a5324a849c31
---

User (2026-06-23, Telegram, verbatim): *"Dude .00079 is not a real signal. .1 or higher is, and .05 is conditionally approved."*

**Rule (effect-size / edge magnitude):**
- **>= 0.10** → REAL signal.
- **0.05–0.10** → CONDITIONALLY approved (worth a deeper look, not a green light).
- **< ~0.05** (e.g. 0.00079, or a 0.008 AUC gap) → NOT signal / noise. Don't chase it, don't build on it.

**Why:** real-money system; tiny edges are noise and overfit-bait. Don't get excited by a 0.008 AUC gap or a
0.00079 anything — the bar for committing effort (e.g. building a model) is a CLEAR edge, not statistical hair.

**How to apply:** to edge/effect metrics — AUC above 0.5 (the gap), out-of-sample R², correlation, null-anchored
real-minus-null gaps, normalized $/day lift. Pair with the existing CI + null-anchoring discipline
([[report-distributions-and-mode]]): an edge must be BOTH ≥ the magnitude bar AND beat its null / CI.

**Example application:** the fspace_cadence direction screen's best real-vs-null AUC gap across all 5 variants
(L1-L3 / L4-L5 × wait 0/10/20s × horizons 30-300s) was **+0.008** — far below 0.05 → NOT signal → the Mamba
build was correctly NOT undertaken. (2026-06-23)
