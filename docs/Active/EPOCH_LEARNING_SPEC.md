# Epoch Learning Spec — Book Gets Schooled
> Date: 2026-04-05
> Status: SPEC — review before implementing

## Problem
Single-pass pipeline: NMP → regret → tree → book → AI → done.
No iteration. No learning from mistakes. The book is built once from NMP regret
and never updated. Result: $41/day IS, -$30/day OOS.

## Solution: Per-Day Epochs with Book Learning

### Four phases:

```
TRAIN:     NMP → NMP regret → tree → book (baseline book v0)
BASELINE:  Clean AI forward pass IS + OOS (book v0) → AI regret → report (H0)
LEARN:     Per-day epochs (book refined day by day with retries → book vN)
VALIDATE:  Clean AI forward pass IS + OOS (book vN) → AI regret → report (H1)
COMPARE:   H0 vs H1 — did epoch learning actually help?
```

### H0 (null hypothesis):
The book straight from NMP regret is the best we can do.
Measured by: clean forward pass before any epoch learning.
This is the control. If epochs don't beat this, they're noise.

### H1 (alternative):
Per-day epoch learning improves the book.
Measured by: clean forward pass after all epoch learning.
Must beat H0 on BOTH IS and OOS to be accepted.

### LEARN phase detail:

```
For each IS day (1 to 277):

    epoch = 0
    while epoch < MAX_EPOCHS:
        1. Run AI on this day using current book
        2. Run regret on this day's AI trades
        3. For each leaf that traded:
           - Compare actual vs optimal (from regret)
           - If regret says "hold longer" → adjust exit bar in book
           - If regret says "counter was better" → adjust regret profile weights
           - If regret says "exit timing wrong" → adjust expected path
        4. If any adjustments were made:
           - RETRY: run AI on same day with updated book
           - Compare: did PnL improve?
           - If yes → keep adjustments, epoch++
           - If no → revert adjustments, stop epochs for this day
        5. If no adjustments needed → day is maximized, stop

    Record final book state → carry to next day
```

### What the book adjusts per epoch:

| Field | Adjustment | Source |
|-------|-----------|--------|
| `same_exit_bar` | Extend if regret says same_extended | `regret.same_best_bar` |
| `counter_exit_bar` | Extend if regret says counter_extended | `regret.counter_best_bar` |
| `regret_profile` weights | Shift toward action that won this day | Day's regret action distribution |
| `same_path` / `counter_path` | Blend with actual observed path | Day's trade paths |

### Non-interference rule:
Adjustments from Day N must not break Days 1 to N-1.
Enforcement: after each day's epochs, spot-check a random sample of
previous days. If degradation > threshold, revert Day N's changes.

### Stopping criteria per day:
- No regret-driven adjustments possible (day is maximized)
- PnL did not improve between epochs (diminishing returns)
- Max epochs reached (default: 5)

## VALIDATE phase:

After all 277 days have been through the LEARN phase:

```
1. Freeze the book (no more adjustments)
2. Clean AI forward pass on ALL IS days → save trades
3. AI regret on IS trades → compare to pre-learning baseline
4. Clean AI forward pass on ALL OOS days → save trades
5. AI regret on OOS trades → this is the real test
6. Report: IS vs OOS, before vs after learning
```

### Success criteria:
- IS $/day should increase (learning worked)
- OOS $/day should increase (learning generalized)
- IS-OOS gap should be small (not overfit)
- Winning day % should increase on both

### Failure modes:
- IS up, OOS flat → memorization, not learning
- IS up, OOS down → overfit, lever-pulling created false patterns
- Both flat → book adjustments don't affect AI behavior (lever disconnected)

## Pipeline (updated):

```
python training/run.py pipeline

TRAIN:
  Step 1:  NMP on IS                                (generates trades)
  Step 2:  NMP Regret                               (what NMP did wrong)
  Step 3:  Train Tree                               (frozen classifier)
  Step 4:  Build Book v0                            (baseline from NMP regret)

BASELINE (H0):
  Step 5:  AI Forward Pass IS (book v0)             (predict)
  Step 6:  AI Regret IS                             (evaluate)
  Step 7:  AI Forward Pass OOS (book v0)            (predict)
  Step 8:  AI Regret OOS                            (evaluate)
  Step 9:  Baseline Report (H0)                     (save as control)

LEARN:
  Step 10: Per-Day Epochs (IS, book v0 → vN)        (book gets schooled)

VALIDATE (H1):
  Step 11: AI Forward Pass IS (book vN)             (predict)
  Step 12: AI Regret IS                             (evaluate)
  Step 13: AI Forward Pass OOS (book vN)            (predict)
  Step 14: AI Regret OOS                            (evaluate)
  Step 15: Final Report (H1 vs H0)                  (did learning help?)
```

## Files to modify:

| File | Change |
|------|--------|
| `training/per_day.py` | Rewrite: per-day epochs with book updates + retry |
| `training/book.py` | Add `update_leaf()` method for epoch adjustments |
| `training/run.py` | Pipeline step 5 calls new per_day |

## Files NOT modified:
- `training/nightmare.py` — data source, correct
- `training/regret.py` — analysis tool, correct
- `training/tree.py` — frozen classifier, correct
- `training/ai.py` — executor, correct (reads book, doesn't modify it)
- `training/gate.py` — classifier + exit checker, correct
- `training/report.py` — reporting, correct
- `training/memory.py` — brain accumulator, correct

## Risk assessment:
- **Overfitting**: each day's lever-pulling could create day-specific adjustments
  that don't generalize. Mitigated by: non-interference check, OOS validation.
- **Book divergence**: after 277 days × 5 epochs, the book could drift far from
  the tree's original classification. Mitigated by: book only adjusts exit timing
  and weights, not entry classification.
- **Slow**: 277 days × up to 5 epochs × AI run time. May need progress bars.
