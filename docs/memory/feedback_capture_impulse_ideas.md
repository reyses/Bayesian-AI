---
name: Impulse ideas go to the backlog, not the build queue
description: When the user fires a new idea mid-session, capture it as a TODO with full context, return to it at the right phase. Don't context-switch the active build path on every impulse.
type: feedback
---

**Rule** (locked 2026-05-09 evening):

When the user introduces a new idea or direction during an active build,
the default is: **capture it as a TODO with full context** and return
to the current foundation work. Switching the active build on every
impulse fragments the foundation and means none of the ideas get
properly grounded.

**Why**: every idea sounds urgent in the moment, but the value of an
idea depends on what's already built. A "what if we use CRM as the
substrate for geometric primitives?" idea can't be built well until
the underlying probability table substrate is solid — otherwise we
build geometric primitives on top of unverified foundations and have
to redo the work.

**How to apply**:

1. When user fires a new idea, first sentence: did they ask me to
   build it now, or did they think out loud?
2. Default reading: "thinking out loud" — capture and continue.
3. If user wants it built now, they'll say so explicitly ("let's
   do it", "build it", or interrupt with another action).
4. The TODO entry must capture:
   - the idea ITSELF (one-line summary)
   - the CONTEXT (why it came up, what triggered it)
   - the PRECONDITIONS (what foundation needs to be solid first)
   - the COST estimate (rough; small / medium / large)
   - the TRIGGER for revisit (when in the build phase to come back)
5. Acknowledge the idea was captured so the user knows it isn't lost.

**Why this matters more than it seems**: in an auto-mode session
the impulse-to-execute is biased high (the agent wants to build).
That bias makes the agent context-switch into every new direction,
which produces a fragmented session where 5 things are half-done
and nothing is shipped. Capturing first and building second
preserves session focus.

**Counter-rule**: If the new idea is BLOCKING the current build
(e.g., reveals an error in the substrate), drop everything and
address it. The capture-first rule is for ADDITIONS, not
CORRECTIONS.

**Oshit-moment override** (user-named exception, locked 2026-05-09):

User is self-aware about chaotic-neutral / AuDHD impulse-firing
tendencies. The dual rule below is designed for that working style:

  - Default = capture (so ideas don't fragment the session)
  - Oshit pivot = allowed when I (Claude) judge it necessary, but
    pivoting MUST preserve the link between current work and the
    new direction.

**When I judge "Oshit moment"** and pivot:

| trigger                                                  | Oshit? |
|---------------------------------------------------------|:-----:|
| New idea reveals error/bias in current substrate         | YES   |
| Current work would be invalid without addressing it      | YES   |
| Current work is solving the wrong problem               | YES   |
| New idea adds value but doesn't invalidate current work  | NO    |
| User wants to chase a curiosity right now                | NO unless explicit |
| Current work is nearly done; new idea is small extension | NO (finish first) |

Examples this session:
- regime_labels_2d circular bias → YES, pivoted (correct call)
- z_high V2 column was wrong metric → YES, pivoted (correct call)
- "what if we use CRM for geometric primitives?" → NO (captured)
- "what about psychohistory framing?" → NO (added vocabulary, didn't pivot)

**The non-negotiable when we DO pivot — preserve the connection**:

When pivoting on an Oshit moment, the response MUST cover:

1. WHAT the current work was, where it stood (so we don't lose it)
2. WHY the new direction supersedes it (the Oshit reasoning)
3. HOW the new work connects to / replaces / completes the current
4. WHAT becomes of the in-flight artifacts (kept as research artifact?
   trashed? superseded by the new build?)
5. WHEN (if ever) we resume the original

Without #3 we death-spiral: jump to new shiny thing, lose context, the
new thing also fragments before completion, repeat. With #3 we have
a thread to pull back to even after multiple pivots.

This is the stabilizer. The user FIRES ideas; my job is to be the
connector that keeps them threaded together so we ship something.

## Examples

GOOD (capture):
- User: "what if we use CRM for geometric primitives?"
- Me: capture as TODO with context, continue current work.

GOOD (act):
- User: "you're using regime labels which we found are flawed"
- Me: stop, fix the substrate, then continue.

BAD (act on impulse):
- User: "what about psychohistory style modeling?"
- Me: spends 600 lines building a "psychohistory framework" mid-session
  while the probability table substrate is still half-built.
