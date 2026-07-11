# Claude ⇄ AG Review Protocol (formalized 2026-07-11, Moises-approved)

Battle-tested on the NT8-catalog audit remediation (AUDIT-ACC-01/02, 2026-07-11).
Applies to ANY work AG executes that carries statistical or production weight.

## Where files live — self-containment rule (Moises, 2026-07-11)
**All review-loop artifacts for a research effort live at the ROOT of that
research project's own folder** (`research/<topic>/`): findings docs, response
plans, verdicts, execution reports. They are part of the research record and
must travel with it. **They do NOT go in `comms/`.** The shared `comms/` folder
holds ONLY: this protocol file, the AG/Gemini context entry point, and true
cross-project handovers. If a loop file is being written into `comms/`, that's
a violation — move it to the research folder root.
Example (canonical): `research/nt8_catalog/` root holds AUDIT_ARTICLE_ACCURACY.md,
SECOND_AUDIT_FINDINGS.md, AUDIT_RESPONSE_PLAN*.md — nothing of that loop is in comms/.

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
- **Append-only.** Nobody deletes or rewrites a prior section — especially not the
  party being reviewed. (Round-1 lesson: AG overwrote the reviewer verification
  that documented a false completion checkbox.)
- **No self-certification.** "Successfully executed" from the executor is a claim,
  not a verification. Only the reviewer stamps VERIFIED.
- **mtime is evidence.** A "completed" item whose artifact predates the finding is
  automatically false.
- **Numbers must be physically possible** before any table ships (the −533 pts/event
  lesson). OQ trace per MVP §4 before regenerating a DOC whose math changed.

## Timers (the Haiku watcher)
Claude does NOT burn main-session wakeups waiting. Use the `ag-watcher` agent
(`.claude/agents/ag-watcher.md`, Haiku): spawn it in the background with the folder
+ filename pattern + completion marker to watch; it polls cheaply and returns when
AG's artifact lands (or on timeout), which re-invokes Claude for the verdict step.
Fallback if the watcher is unavailable: ScheduleWakeup at the user-set cadence
(current preference: 180s).
