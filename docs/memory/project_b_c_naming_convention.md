---
name: project-b-c-naming-convention
description: "B-prefix for production candidates (validated/active in stack), C-prefix for failed candidates (research artifacts only, kept for audit but not in production). Numbering is contiguous within each prefix."
metadata: 
  node_type: memory
  type: project
  originSessionId: e1202bdb-1787-4774-b48b-b312af212043
---

Naming convention for regret-oracle models, established 2026-05-17.

**B-prefix**: production candidates. Validated, integrated into the
forward pass / composite pipeline, or actively under validation.
Numbering contiguous (B1, B2, B3, ... no gaps).

**C-prefix**: failed candidates. Research artifacts kept for audit but
NOT in production. Marked as failed in their docstring/header so future
sessions don't accidentally promote them. Numbering can match the
original B-slot they once occupied (so C9 was once B9 before being
demoted).

**Rule**: when a B-model fails validation:
  1. Rename ALL its files (tool .py, cache .parquet/.pt, report .txt)
     using `git mv` from b{N}_* -> c{N}_*
  2. Update all internal references (paths, var names, class names,
     prose) to c{N} prefix
  3. Mark docstring with "(failed candidate)" suffix
  4. Free the B{N} slot for the next production candidate
  5. Save the rename in the daily journal + finding report

**Current state (2026-05-18)**:
  B1  pivot-imminent (HGB)                     ACTIVE
  B2  fakeout (HGB)                            ACTIVE
  B3  time-to-pivot (HGB regressor)            ACTIVE
  B4  pivot-region (HGB)                       ACTIVE
  B5  leg-phase 3-class (HGB)                  ACTIVE
  B6  directional-pivot 3-class (HGB)          ACTIVE
  B7  leg-sizer (HGB regressor)                ACTIVE -- production entry sizer
  B8  hour-risk (HGB regressor)                ACTIVE
  B9  during-trade remaining-amplitude         ACTIVE -- FIRST L5 model (validated 2026-05-17 OOS +$67/day CI [+$32,+$106])
  -----
  B10 RESERVED for next during-trade promotion (C12 candidate)
  B11 RESERVED for next during-trade promotion (C13 candidate)
  -----
  C9  LSTM leg-sizer                           FAILED (was B9 originally, lost to B7 GBM)
  C10 LSTM direct-trade                        FAILED (was B10 originally, OOS Pearson -0.02)
  C11 bad-trade binary-cut detector            FAILED (was B9 briefly 2026-05-17; both v1 + v2 walk-forward 0/N significant)
  C12 RESERVED for next experimental retarget (B6 during-trade?)
  C13 RESERVED for next experimental retarget (B5 during-trade?)

**Two-stage workflow (effective 2026-05-18)**:
  STAGE 1 - candidate: new during-trade retarget research takes lowest available C-slot.
    Build, walk-forward IS, OOS-sealed test. If validates, STAGE 2; else stays as C-artifact.
  STAGE 2 - promotion: validated candidate gets renamed to lowest available B-slot.
    File rename (git mv c{N}_* b{M}_*), internal reference update, memory update.

**Why:** Contiguous B-numbering preserves a clean operational view of
the production stack. C-prefix preserves research artifacts (we don't
delete failed experiments — they teach us what doesn't work) without
polluting the production numbering.

**How to apply:** When the user mentions B{N}, assume production
candidate unless prefixed with C. When proposing a new model, take the
LOWEST available B-slot (i.e., reclaim freed slots from C-renamed
predecessors before extending the sequence).

Related: [[project_during_trade_b_stack.md]] (B9 = first during-trade
model in development).
