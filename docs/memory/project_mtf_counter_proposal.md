---
name: MTF Two-Layer Counter-Proposal (SUPERSEDED)
description: 29D MTF architecture superseded by nn_v2 79D pipeline with 3-CNN system
type: project
---

## SUPERSEDED (2026-04-03)

The MTF Two-Layer Counter-Proposal (29D features, Direction + Duration CNNs) was
superseded by the nn_v2 pipeline which uses 79D features and a 3-CNN system:
- CNN Flip (direction at entry, 70.6%)
- CNN Hold (hold/exit during trade, 94.8%)
- CNN Risk (cut losers early)

Result: $613/day OOS vs the counter-proposal's untested spec.

**Why:** nn_v2 was built from clean data discovery, not theoretical architecture.
**How to apply:** Spec at `docs/Active/COUNTER_PROPOSAL_MTF_TWO_LAYER.md` is historical only.
