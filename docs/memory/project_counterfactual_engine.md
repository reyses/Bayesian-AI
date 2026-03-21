---
name: Counterfactual Engine (Goat Brain)
description: Every trade and skip spawns phantom trades with alternative parameters. The optimization surface emerges from real-time what-if analysis. Foundation for the goat brain.
type: project
---

Every decision spawns phantom trades testing alternative thresholds:
- Skip → phantom enters anyway with different gate settings
- Trade → phantoms test different exit thresholds (giveback 20%-70%)
- Each phantom tracked for 80 bars (20 min), same exit cascade

After N phantoms complete, optimal parameter values emerge from the data.
System adapts thresholds in real-time. No offline optimization needed.

**Why:** The cat/crow/monkey learn WHAT to trade. The goat learns HOW to tune
ALL parameters simultaneously from continuous counterfactual evaluation.

**How to apply:**
- IS: synchronous (data in _states_map). Report at end of IS.
- OOS: validates IS-learned parameters hold out of sample.
- Live: async background workers. Auto-tune or flag for human review.
- Spec: `docs/specs/COUNTERFACTUAL_ENGINE.md`

**Evolution:** Cat (regime) → Crow (seeds) → Monkey (CNN) → Goat (counterfactual).
The goat doesn't replace the others — it TUNES them.
