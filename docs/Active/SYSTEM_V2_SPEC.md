# System V2 Spec — Full Implementation Plan
> Date: 2026-04-05
> Status: APPROVED SPEC — implement in next session

## Context
training pipeline built. NMP baseline established. Strategy tree + AI continuous
positioning tested. Honest sequential OOS = $45/day (bulk was $711 — lookahead).
Need: per-day iteration with calibrated exits, chain trades, dense brain.

## Architecture (3 layers)

### Layer 1: PROCESS (learning, offline)
Goes day by day through IS. Builds the strategy book.

### Layer 2: AI (execution, runtime)
Reads 79D every bar. Follows the book. Manages positions.

### Layer 3: REPORTING
Produces the reality check. Drives production risk decisions.

---

## PROCESS LAYER — Per-Day Iteration

### For EACH IS day (all 277, not just losers):

```
1. Run all existing branches on this day
2. For each trade outcome:
   a. Compute regret (same early/extended, counter early/extended)
   b. Compare to branch's expected path from book
   c. Score: how much PnL gap remains?
3. If gap > threshold:
   a. TRY existing branches first (adjust exit timing, flip direction)
   b. ONLY if no existing branch covers it → add new specialist
4. Re-run this day with updates
5. Repeat until this day is maximized or max rounds
6. Record ALL evidence to brain (dense per branch)
7. Move to next day (brain carries forward)
```

### Non-interference rule:
New branch must NOT break previously fixed days.
Each branch is a specialist with narrow activation.

### Branch creation protocol:
1. Exhaust ALL existing branches first
2. If existing branch needs exit adjustment → create sub-branch, don't modify original
3. If no branch covers the condition → create new specialist
4. Verify: new branch only activates on its specific condition

---

## AI LAYER — Execution Logic

### Every bar:
```
1. Gate evaluates 79D → returns calibrated playbook
   (branch, strategy, direction, expected path, exit conditions)

2. If FLAT:
   - If gate says trade → ENTER (direction from branch)
   - Record entry decision (branch, expected path)

3. If IN POSITION:
   a. New signal SAME direction as current position:
      → STAY in trade
      → UPDATE expected path to new branch's path
      → Brain records: "branch updated from X to Y at bar N"
   
   b. New signal DIFFERENT direction:
      → EXIT current trade (record to brain with chain history)
      → ENTER opposite direction
      → This counts as 2 trades (close + open)
   
   c. Calibrated exit triggered (from book):
      - Path divergence: actual PnL deviates > 2x from expected
      - Optimal bar reached: branch's exit timing hit
      - Exit 79D matched: features look like the exit signature
      → EXIT to flat (or flip if new signal ready)
   
   d. None of the above → HOLD
```

### Chain recording:
```
Trade starts as branch 68 LONG at bar 0
  Bar 3: new signal = branch 83 LONG → stay, update path
  Bar 7: new signal = branch 142 SHORT → exit LONG, enter SHORT
  
Brain records:
  Trade 1: branch 68→83 LONG, bars 0-7, PnL=$XX
    - Chain point at bar 3 (branch 68 PnL at that point: $YY)
  Trade 2: branch 142 SHORT, bars 7-...
```

---

## BAYESIAN BRAIN

### Records per trade:
- Entry branch + any chain updates
- PnL at each chain point
- Final PnL and exit reason
- Expected path vs actual path (divergence)
- Entry match score (how well 79D matched branch signature)

### Accumulates per branch:
- Win rate (from actual execution, not tree training)
- Average PnL
- Chain frequency (how often this branch chains into others)
- Typical exit reason
- Path adherence (do trades follow the expected path?)

### Evidence grows day by day:
Day 1 brain has sparse data. Day 277 brain has dense data.
Later days' decisions are informed by all previous evidence.

---

## REPORTING

### End-of-run report table:
```
                        IS (277 days)    OOS (34 days)
---------------------------------------------------------
Total PnL               $X,XXX           $X,XXX
$/day                   $XXX             $XXX
Total trades            X,XXX            X,XXX
Trades/day              XX               XX

WINNING DAYS:
  Count                 XXX              XXX
  Mode PnL              $XXX             $XXX
  Avg PnL               $XXX             $XXX
  Avg trades/day        XX               XX

LOSING DAYS:
  Count                 XXX              XXX
  Mode PnL              $-XXX            $-XXX
  Avg PnL               $-XXX            $-XXX
  Avg trades/day        XX               XX

LOW TRADE DAYS (0-10 trades):
  Count                 XXX              XXX
  Avg PnL               $XXX             $XXX
  Win/Loss              XX/XX            XX/XX

CHAINS:
  Single trades         XX%              XX%
  2-chain               XX%              XX%
  3+ chain              XX%              XX%

TYPICAL MONTH ESTIMATE:
  Winning days: XX × mode $XXX = $X,XXX carry
  Losing days:  XX × mode $-XXX = $-X,XXX drag
  Net typical month: $X,XXX
```

### Mode over mean:
Mode shows what a TYPICAL day looks like. Mean is dragged by outliers.
The mode × count gives the realistic monthly expectation.

### Drawdown analysis (for production band-aids):
- Max consecutive losing days
- Worst single day loss
- Worst weekly PnL
- Max drawdown from peak equity

These are NOT strategy fixes. They become risk management guardrails:
- Daily stop loss = worst typical losing day × 1.5
- Per-trade max loss = from branch expected drawdown
- Weekly equity pause = if weekly loss > X

---

## FILES TO MODIFY/CREATE

### Modify:
| File | Changes |
|------|---------|
| `training/gate.py` | Already updated — uses book for calibration + should_exit() |
| `training/ai.py` | Chain logic: same dir = stay + update path, diff dir = flip |
| `training/per_day.py` | Every day iterated, exhaust branches before adding new |
| `training/memory.py` | Record chain history, path adherence, entry match |
| `training/run.py` | Pipeline command updated, report at end |

### Create:
| File | Purpose |
|------|---------|
| `training/report.py` | End-of-run reporting with IS/OOS comparison table |

### Do NOT modify:
- `training/ticker.py` — dumb pipe, correct
- `training/aggregator.py` — TF aggregation, correct
- `training/build_dataset.py` — sheet music builder, correct
- `training/sfe_ticker.py` — test mode replay, correct
- `training/nightmare.py` — NMP seed generator, correct (but no longer used at runtime)
- `training/regret.py` — counterfactual analysis, correct
- `training/tree.py` — strategy tree, correct
- `training/book.py` — strategy book generator, correct

---

## EXECUTION ORDER

1. Fix `ai.py` — chain logic + calibrated exits from gate
2. Fix `memory.py` — chain recording + path adherence
3. Fix `per_day.py` — every day, exhaust branches first, brain carries forward
4. Build `report.py` — IS/OOS comparison with modes + typical month
5. Update `run.py` pipeline to use all of the above
6. Run full pipeline on honest sequential IS
7. Run AI on honest sequential OOS
8. Read the report — that's the reality check

---

## PARKED (future, not this session)
- Segment regression (redundant with path divergence)
- 15s resolution dataset
- 5s/1s resolution for execution layer
- Live deployment + NT8 bridge
- ACL in production (brain adjusts strategies based on live evidence)

---

## CURRENT STATE
- IS honest: $32/day, 48% winning (pre-iteration)
- OOS honest: $45/day, 50% winning (pre-iteration)
- OOS bulk: $711/day, 84% winning (lookahead inflated)
- Strategy book: 75 branches, ~200 after iteration
- Brain: sparse (needs dense evidence from per-day iteration)
- Sheet music: IS built (277 days), OOS built (34 days)
