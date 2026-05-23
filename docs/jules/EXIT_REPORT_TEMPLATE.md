# Jules Exit Report — TEMPLATE

> Copy this file to `YYYY-MM-DD_<task_name>_exit_report.md` and fill in. Required for every Jules run on this project.

---

## 1. Task identifier
- **Brief file**: `docs/jules/YYYY-MM-DD_<task_name>.md`
- **Branch / commit range**: `<branch_name>` from `<base_sha>` to `<head_sha>`
- **Wall-clock duration**: `<HH:MM>`
- **Jules model version**: `<version>`

## 2. Outcome summary
One paragraph. What was the task, what got done, did it succeed.

## 3. Deliverables status
Mirror the checklist from the brief. One row per deliverable.

| # | Deliverable | Status | Notes |
|---|---|---|---|
| 1 | `<verbatim from brief>` | ✅ done / ❌ skipped / ⚠️ modified | one line |
| 2 | ... | | |

Legend:
- ✅ **done** — completed as specified
- ❌ **skipped** — not done, see Deviations Log for why
- ⚠️ **modified** — done but differently than specified, see Deviations Log

## 4. DEVIATIONS LOG (most important section)

**Any time an instruction was not followed verbatim or was changed mid-task, document it here.**

For each deviation:

### Deviation #N
- **Original instruction** (quote verbatim from the brief):
  > "..."
- **What was actually done**:
  > ...
- **Reason** (what circumstance forced the change — code state, broken dependency, ambiguity, conflicting constraint, etc.):
  > ...
- **Risk assessment** (what could break because of this deviation):
  > ...
- **Reversibility** (how to undo this deviation if the human disagrees with the call):
  > ...

Repeat for every deviation. **If zero deviations, write "No deviations." in this section — do not omit it.**

## 5. Smoke test results
Mirror the smoke tests from the brief. For each:

| # | Test | Result | Output excerpt or error |
|---|---|---|---|
| 1 | `<command from brief>` | ✅ pass / ❌ fail | `<one line of stdout/stderr>` |

If any failed, this run does **not** qualify as complete — the deviation must be documented in §4 with a remediation path.

## 6. File-change statistics
| Operation | Count | Notable files |
|---|---|---|
| Created | N | top-3 most important |
| Modified | N | top-3 most important |
| Deleted | N | top-3 most important |
| Moved/Renamed | N | top-3 most important |

Also report:
- Lines added: N
- Lines deleted: N
- Number of test files touched: N
- Number of model artifacts (`*.pkl`, `*.pt`) modified: **must be 0 unless brief explicitly authorized**

## 7. Open questions / blockers for human review
Bullet list. Anything Jules couldn't decide on, anything that smelled wrong but was out of scope, anything that needs the user's eyes before next steps.

## 8. Suggested follow-up
Bullet list of recommended next tasks. Distinguish:
- **(immediate)** — needed before this work can be considered safe to merge
- **(soon)** — should be picked up within the current sprint
- **(eventual)** — nice-to-have cleanup

## 9. Sign-off

- [ ] All sections above filled in (no placeholders left)
- [ ] Deviations log complete (or "No deviations." stated)
- [ ] No model artifacts modified (or modification explicitly authorized)
- [ ] No hooks bypassed (`--no-verify` not used)
- [ ] Exit report committed in the same PR as the work

**Final status**: ✅ COMPLETE / ⚠️ PARTIAL / ❌ ABORTED

---

## Why this template exists

Silent deviation from a spec is the highest-risk failure mode in delegated work. The user finds out weeks later when something doesn't behave as documented. This template forces deviations to surface immediately, with reasoning attached, so the human can either accept or reverse the call while context is still fresh.

**The principle**: a Jules run that disagrees with the brief is fine; a Jules run that hides the disagreement is not.
