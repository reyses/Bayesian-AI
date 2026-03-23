---
name: PhysicsEngine changes must be surgical — it's the live revenue source
description: PhysicsEngine is running live in sim making money. Any change risks sinking the boat. Treat as production — no refactors, no experiments, only targeted fixes with clear rollback.
type: feedback
---

PhysicsEngine is the LIVE revenue source. First session: $1,495 (93 trades, PF 1.96).
It's fragile — $264 without the outlier. Any wrong change can turn it negative.

**Why:** The system is "hanging on a thread" — thin edge, lots of churn, occasional big wins.
The user explicitly said treat it like a boat that sinks with one wrong move.

**How to apply:**
- NO refactors, NO feature changes, NO "improvements" to PhysicsEngine without explicit approval
- Only surgical fixes: ORPHAN_FLATTEN bug, 0-bar SL entries, clear engineering bugs
- Every change must have a rollback plan (git revert)
- Test on OOS FIRST before deploying to live sim
- All experimental work goes to AdvanceEngine (the rebuild), NOT PhysicsEngine
- PhysicsEngine stays frozen except for bug fixes
- The grounded feature rebuild happens in AdvanceEngine — PhysicsEngine keeps its ugly 12 features
- When AdvanceEngine is proven better on OOS, THEN it replaces PhysicsEngine. Not before.
