---
name: time-assessment-calibration
description: "Time estimates in autonomous mode are systematically too pessimistic. Track actual time (start/end) and use to calibrate. When over-budget, prefer doing it RIGHT to handing back a cut-corner partial."
metadata: 
  node_type: memory
  type: feedback
  originSessionId: e1202bdb-1787-4774-b48b-b312af212043
---

In autonomous (sleep-run) mode, my time estimates are consistently too
pessimistic. I budget conservatively, cut corners "for time," and end up
with hours of unused budget at the end.

**Why:** Recorded 2026-05-19. Cut corner on L5Decider's zigzag pivot
detector during overnight build -- wrote a simpler running-extreme
state machine instead of porting the training pipeline's
`tools/build_zigzag_pivot_dataset.py` (min_bars=36 + ATR-on-pivot-series).
Result: live mock captured only 30% of OOS leg count, killing the
expected edge. User: "you always end up with extra time, improve your
time assessments."

**How to apply:**
  1. At the **start of every code-write or analysis task**, note the
     wall-clock start time (`date +%H:%M`).
  2. At the **end**, note the end time and compute actual duration.
  3. Compare to my initial estimate.
  4. Use the calibration to revise future estimates:
     - If actual is consistently <50% of estimate -> estimates are 2x too pessimistic
     - Use the surplus to do the job RIGHT instead of cutting corners
  5. **Don't cut corners on technical correctness because of a perceived
     time crunch.** A wrong-but-shipped implementation is worse than a
     right implementation that takes longer. The user can't deploy
     something they can't trust.
  6. Journal the start/end/actual/estimated for each task -- builds the
     calibration dataset over time.

**Concrete revision (2026-05-19):**
  - Build "single component end-to-end" estimate: was 30-60 min, actual
    ~15-30 min. Cut by 50%.
  - Build "research tool + run on data + chart" estimate: was 60-90 min,
    actual ~30-45 min. Cut by 50%.
  - Multi-day mock sweep: estimated 60 min wall clock, actual ~50 min.
    Fine.

**Anti-pattern to avoid:**
  - "I'll do the simpler version because I might run out of time" --
    no, do the right version. If it takes longer, deliver less but
    correct.
