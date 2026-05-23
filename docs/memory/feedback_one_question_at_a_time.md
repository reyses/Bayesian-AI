---
name: feedback-one-question-at-a-time
description: "Ask the user one question at a time, never batch multiple questions in a single turn — they process sequentially, batched questions overwhelm"
metadata: 
  node_type: memory
  type: feedback
  originSessionId: 0e6e30c4-bf89-4817-9ea3-0e4056c5e720
---

Present questions to the user **one at a time**, never as a numbered list of multiple questions in one response.

**Why:** User stated directly (2026-05-23): "i can only process 1 question at a time cuz thats how my brain works." Batching questions causes either incomplete answers (user picks one and forgets the others) or cognitive overload that stalls the conversation.

**How to apply:**
- When a decision-flow has N questions, ask question 1, wait for the answer, then ask question 2.
- If a topic genuinely requires N parallel decisions (e.g. greenlighting an analysis with multiple parameters), order them by gating priority and ask only the first.
- Internal reasoning / pre-commitment proposals (e.g. "I propose decision rule X — accept?") still count as a question. One per turn.
- Exception: yes/no confirmation paired with a single follow-up clarification (e.g. "Run it? If yes, do you want variant D included?") is borderline — prefer to split unless the second is trivial.
- This applies to ALL conversations on this project going forward, not just the current session.

Linked context: [[project_parity_b9_horizon_2026_05_20]], live-deploy decision flows where I previously batched 4–5 questions in one turn.
