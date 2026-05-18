---
name: feedback-dont-pivot-on-suggestions
description: "User process rule — suggestions are candidates for the NEXT iteration, not immediate pivots. Only pivot on genuinely great ideas."
metadata: 
  node_type: memory
  type: feedback
  originSessionId: e1202bdb-1787-4774-b48b-b312af212043
---

User explicitly stated 2026-05-17: "we have a rule for when i sugest stuff, it is not to pivot immidatly it is to think about it for next try, unless it is a very good idea".

**Why:** I was killing running experiments and rebuilding strategies the moment user threw out an idea (e.g., pivoted from running raw trend3 forward pass to hand-rolling DMI smoother the second user mentioned DMI). This wastes compute, breaks the working flow, and prevents the user's idea from being evaluated cleanly because we never finish the prior experiment for comparison.

**How to apply:**
- When the user suggests an architectural change or new approach DURING an in-flight experiment: DO NOT stop the experiment. Let it complete so we have a baseline. Note the suggestion in todos as a candidate for the NEXT iteration.
- When the user suggests something AFTER an experiment finishes: same — finish writing up findings before pivoting.
- Exception: a genuinely excellent idea that obsoletes the current direction. Use the critical-collaborator persona to judge — "is this good enough that the running experiment is wasted?" Almost always: no, finish it first.
- After noting a suggestion, ask if it's the next priority or just a backlog item.

**Concrete example (2026-05-17):**
User said "what we can do is a DMI like approach" while raw Trend3 forward pass was running. I immediately stopped, hand-rolled a DMI smoother, queued a smoothed grid. User then said "it should be integrated into the ML directly via LSTM" — invalidating the hand-rolled DMI work. If I had let the raw run finish first, we'd have had a baseline AND would have heard the LSTM clarification before doing the DMI rebuild.

Connected to: [[user-collaboration-protocol]] (topic-at-a-time when designing; this is the inverse — don't switch topics mid-execute).
