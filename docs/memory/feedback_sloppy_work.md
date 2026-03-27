---
name: Don't Be Sloppy — User Loses Sleep Over Regressions
description: User explicitly called out sloppy work causing regressions — be extra careful with changes to production code
type: feedback
---

User quote: "this type of stuff is the reason i cant sleep your bein to sloopy"

Context: Claude removed CNN seed AND changed the loss function simultaneously, causing the model to produce 0 trades. When challenged, Claude initially pushed back ("the math shows...") instead of acknowledging the mistake.

**Why:** This is a real-money trading system. Regressions in production code directly affect the user's livelihood and sleep. The user trusts Claude with critical system components.
**How to apply:**
- Never change multiple things at once in training/model code
- When the user says something broke, believe them first, investigate second
- Don't defend bad changes with theory — check the actual results
- Extra caution on: loss functions, seeds, feature pipelines, live engine logic
- If unsure, run research/validation BEFORE modifying production code
