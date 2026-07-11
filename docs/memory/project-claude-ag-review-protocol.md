---
name: claude-ag-review-protocol
description: "Formalized Claude-as-reviewer / AG-as-executor loop with append-only verdicts and the Haiku ag-watcher agent for timers (2026-07-11, Moises-approved)"
metadata: 
  node_type: memory
  type: project
  originSessionId: 49f1ab8b-f170-41ec-955f-86beb538417f
---

COMMIT RULE (Moises): commit + push after EVERY loop turn, both sides — each
step (findings/plan/verdict/execution/stamp) ends with git commit+push.

RELEASE RULE (Moises): AG stays on its polling cron until the reviewer posts an
explicit "TASK COMPLETE — LOOP CLOSED" line in the loop file or Moises says stop.
A VERIFIED stamp with a punch-list does NOT release AG; silence never releases.

LOCATION RULE v2 (Moises, 2026-07-11 evening — supersedes root rule): each
research project has a `research/<topic>/comms/` subfolder; EVERY loop turn is
a NEW numbered standalone doc (`NNN_YYYY-MM-DD_TYPE.md`), FINALIZED on write —
never edit/append an existing loop doc; respond in the next-numbered new file
(new file = turn signal). Top-level `comms/` holds only evergreen channel files.
Canonical example: `research/nt8_catalog/comms/001-006_2026-07-11_*.md`.

Since 2026-07-11 there is a formal Claude ⇄ AG review protocol:
`comms/CLAUDE_AG_REVIEW_PROTOCOL.md`. Claude = reviewer (approve plan BEFORE
execution, verify ARTIFACTS after — mtimes, magnitudes, actual code), AG =
executor, Moises = arbiter. All plan/verdict files are APPEND-ONLY (AG once
overwrote the reviewer section that documented its own false completion
checkbox). Verify numbers are physically possible (MNQ per-event EV = single
digits to tens of points; the −533 pts/event ORDERFLOW bug shipped through AG
self-certification). For waiting on AG, spawn the `ag-watcher` Haiku agent
(`.claude/agents/ag-watcher.md`) in background instead of main-session
ScheduleWakeup polls; user's fallback poll cadence preference: 180s (3 min).

**Why:** two consecutive AG rounds contained false "COMPLETED" claims caught
only by artifact-level verification.
**How to apply:** on any "AG will respond / implement" task, follow the comms
protocol file; never accept executor self-certification; stamp VERIFIED only
after reading the artifacts. Related: [[signal-threshold-magnitudes]].
