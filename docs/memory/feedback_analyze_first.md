---
name: Analyze First, Then Plan, Then Execute
description: Never iterate blindly. Always present findings and plan before making changes.
type: feedback
---

## Rule: Analyze → Present → Plan → Execute

When working on system improvements:

1. **ANALYZE** the current results — read the data, run the numbers
2. **PRESENT** findings to the user — what's working, what's not, why
3. **PLAN** the change — propose what to modify and expected impact
4. **EXECUTE** only after the user agrees

Do NOT iterate through code changes without presenting findings first.
Do NOT make multiple edit → run → edit → run cycles without stopping to analyze.

**Exception:** If the user explicitly requests "iterate until we achieve X" — then rapid iteration is allowed.

**Why:** Blind iteration wastes time and breaks things. Each change should be informed by data from the previous step. The RCA process (feedback_rca_process.md) requires understanding before action.

**How to apply:** After every run, stop. Read the output. Present what changed and why. Then ask what the user wants to do next — don't assume.
