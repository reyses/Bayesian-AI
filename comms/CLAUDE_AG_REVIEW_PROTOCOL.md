# Claude ⇄ AG Review Protocol (formalized 2026-07-11, Moises-approved)

Battle-tested on the NT8-catalog audit remediation (AUDIT-ACC-01/02, 2026-07-11).
Applies to ANY work AG executes that carries statistical or production weight.

## Where files live — one doc per turn (Moises, 2026-07-11, v2 supersedes root rule)
Each research project carries its own **`research/<topic>/comms/`** subfolder as
part of the standard structure. **Every turn of the loop is a NEW standalone doc**
— numbered and dated: `NNN_YYYY-MM-DD_TYPE.md` (e.g., `001_…_INSTRUCTIONS`,
`002_…_IMPLEMENTATION_PLAN`, `003_…_APPROVAL`, `004_…_EXECUTION_REPORT`,
`005_…_AUDIT`, `006_…_TASK_COMPLETE_LOOP_CLOSED`).
- **A doc is FINALIZED the moment it is written.** Nobody edits or appends to an
  existing doc — a response is always the next-numbered NEW file. A new file
  appearing in the folder IS the signal that the other side has moved.
- The shared top-level `comms/` folder holds ONLY the evergreen channel files
  (this protocol, the AG context entry point, mailbox) — never loop docs.
- Canonical example: `research/nt8_catalog/comms/001–006_2026-07-11_*.md`.

## Roles
- **AG** — executor. Writes plans, runs scripts, regenerates artifacts.
- **Claude** — reviewer. Approves plans BEFORE execution, verifies artifacts AFTER.
- **Moises** — arbiter. Overrides either side; sets cadence.

## The loop
1. **Findings**: Claude writes an audit/findings doc with a numbered
   "Required next actions" section (e.g., `SECOND_AUDIT_FINDINGS.md §4`).
2. **Plan**: AG writes a response plan as a NEW dated file (or a dated appended
   section) in the same folder, ending with `*(Waiting for Reviewer Verdict)*`.
   A plan describes intended changes — it never claims completion.
3. **Verdict**: Claude appends `## Reviewer Verdict (Claude, round N)` to that file:
   **APPROVED — EXECUTE** (optionally with numbered BINDING mods) or
   **MODS REQUIRED**. AG executes only after approval, mods folded in.
4. **Execution report**: AG appends (or writes a linked doc): files touched,
   root causes found, and how each numbered item was addressed.
5. **Verification**: Claude verifies against ARTIFACTS, not claims — reads the
   actual scripts/DOCs, sanity-checks magnitudes (MNQ per-event EVs must be
   plausible: single-digit-to-tens of points), checks mtimes vs claims.
   Appends **✅ VERIFIED** (may carry a non-blocking punch-list) or
   **❌ REJECTED** with numbered failures → back to step 2.
6. Journals updated at each verdict (docs/daily + INDEX).

## Hard rules (violations void the round)
- **Commit + push after EVERY turn, both sides** (Moises, 2026-07-11). Each loop
  step (findings, plan, verdict, execution, verification stamp) ends with
  `git commit` + `git push` so assets are safeguarded and each side's turn is a
  recoverable checkpoint. An unpushed turn is an at-risk turn.
- **Finalized-on-write.** Nobody edits, deletes, or appends to an existing loop
  doc — every response is the next-numbered NEW doc (see "one doc per turn"
  above; supersedes the earlier single-file append-only mechanics, same intent).
  (Round-1 lesson: AG overwrote the reviewer verification that documented a
  false completion checkbox.)
- **No self-certification.** "Successfully executed" from the executor is a claim,
  not a verification. Only the reviewer stamps VERIFIED.
- **mtime is evidence.** A "completed" item whose artifact predates the finding is
  automatically false.
- **Numbers must be physically possible** before any table ships (the −533 pts/event
  lesson). OQ trace per MVP §4 before regenerating a DOC whose math changed.

## AG stays on cron until explicitly released (Moises, 2026-07-11)
AG keeps its polling cron ALIVE for the entire loop — through every plan → verdict
→ execution → verification cycle, including punch-lists and rejected rounds. AG
stands down ONLY when one of these appears:
1. The reviewer (Claude) posts an explicit **`TASK COMPLETE — LOOP CLOSED`** line
   in the loop file, or
2. Moises says stop.
A ✅ VERIFIED stamp alone does NOT release AG if it carries a punch-list or names
open items — AG works those and keeps polling. Silence from the reviewer is never
a release.

## Timers (waiting on AG) — preferred mechanism order (updated 2026-07-11)
1. **Background shell watcher (PREFERRED, zero tokens):** a `run_in_background`
   PowerShell loop that polls the research `comms/` folder every ~20s and EXITS
   when the next-numbered doc appears (3h timeout) — the harness wakes Claude on
   exit. Event-driven from Claude's side; costs nothing while waiting. (This is
   the native version of a "folder-monitor MCP server" — MCP can't push wakeups
   into a session anyway, so a server adds infra without adding the push.)
2. ScheduleWakeup polls at the user-set cadence (180s) — fallback when a shell
   watcher can't run.
3. The `ag-watcher` Haiku agent (.claude/agents/ag-watcher.md) — deprecated for
   waiting: subagent sandboxes block foreground sleeps, so it kept detaching and
   returning early. Keep only for "summarize what changed" scouting, not timing.
