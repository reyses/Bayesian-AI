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

LOCATION RULE (Moises): all review-loop files (findings, plans, verdicts,
execution reports) live at the ROOT of the research project folder
(`research/<topic>/`), NEVER in `comms/` — comms/ holds only the protocol file,
the AG context entry point, and true cross-project handovers.

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
