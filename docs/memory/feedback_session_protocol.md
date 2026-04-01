---
name: Session Start/End Protocol
description: On session end note time/date, on session start check time/date and resume from todo list unless fixing an error
type: feedback
---

## Session Protocol

**Session End**: User says "session end" → note the current time and date.

**Session Start**: When user sends first message of new session:
1. Check current time and date
2. Unless the message is about fixing an error → read the todo list
3. Resume from where we left off on the todo list
4. Don't ask "what do you want to work on?" — the todo list tells you

**Why:** Eliminates the cold-start problem. Every session picks up where the last one stopped. No context loss, no re-explaining.

**How to apply:** Always. Every session boundary.
