# Cycle 02 — Forward pass on RM pivots

> PDCA cycle inside the RM Pivot research project.
> Ref: [project.md](project.md) §IMPROVE · built on Cycle 1 STANDARDIZED findings

---

## PLAN

### Hypothesis

Using the Cycle 1 standardized rule (confirmed RM zigzag pivot at R=$4, direction = pivot type: LOW→LONG, HIGH→SHORT), a simple forward pass that enters at pivot confirmation and exits at the next RM pivot will be **net-positive** on both IS and OOS, with mean $/trade ≥ $10 net (pre-slippage, pre-commission — those come in Control).

### Why this should work

- Cycle 1 showed the direction rule is right 83.5% of the time
- RM cord ceiling is $2.5-3K/day (from journal 2026-04-21)
- ~23 IS pivots/day, ~27 OOS pivots/day
- Each pivot-to-pivot leg has 2R=$8 retracement cost (enter $R late, exit $R early)
- Only legs > $8 amplitude can net positive; in RM space most legs exceed this

### Change

New script `tools/forward_pass_rm_pivot.py`. Standalone. No engine modification (engine still has the wrong direction rule; Cycle 3 will port if Cycle 2 passes).

### Predicted outcome

| Metric | Predicted (IS) |
|---|---|
| Trades/day | 20–25 |
| Trade WR = (∑profit/\|∑loss\|)−1 | ≥ +0.5 (profit ≥ 1.5× loss) |
| Mode $/trade | ≥ $10 |
| Mean $/trade | ≥ $10 |
| Daily mean PnL | ≥ $200 |
| Daily median PnL | ≥ $100 |
| Day WR (%winning days) | ≥ 60% |
| Daily mode bucket (∆$25) | ≥ $0 (positive or break-even) |

### Success gate

**All 7 predicted metrics hit on IS.** OOS must match within 30% of IS on mean $/day (OOS/IS ≥ 0.7).

### Kill gate

- Trade WR < 0 (losses exceed profits) → RM-pivot trade idea dead, pivot away
- IS mean $/trade < $5 → signal doesn't survive 2R retracement tax at R=$4
- OOS/IS < 0.3 → overfit

### Method

Per day (IS 2025 + OOS 2026):
1. Load 1m closes + 1m_z_se residuals (FEATURES_5s nearest-ts lookup)
2. Compute rolling 60-bar OLS → RM series
3. Live-safe zigzag on RM at R=$4
4. Walk pivots in chronological order:
   - On pivot confirm: **enter** in pivot direction (LOW→LONG, HIGH→SHORT) at current 1m close price
   - On next pivot confirm: **exit** at current 1m close price
   - Record: entry_ts, exit_ts, direction, entry_price, exit_price, pnl, held_min
5. EOD force-close open positions at 20:55 UTC
6. No stop-loss, no take-profit, no max-hold (v1 = pure pivot-to-pivot)

No chains in v1 (one open position at a time).

### Output

- Trades pickle: `training_RM_physics/output/trades/rm_is.pkl`, `rm_oos.pkl`
- Findings: `research/rm_pivot/findings/2026-04-22_fwd_pass_rm_pivot.md`
- Chart: same folder, PNG
- All metrics computed on post-EOD-close trades

### Reproduction command

```
python tools/forward_pass_rm_pivot.py
```

---

## DO

Ran `python tools/forward_pass_rm_pivot.py` on 2026-04-22. Also swept R ∈ {$2,$4,$6,$10,$15,$20} with `--sweep`.

Outputs:
- Pickles: `training_RM_physics/output/trades/rm_is.pkl` (2,982 trades), `rm_oos.pkl` (846)
- Report: `research/rm_pivot/findings/2026-04-22_fwd_pass_rm_pivot.md`
- Chart: same folder, PNG

---

## CHECK

### Primary (R=$4, the Cycle-1-chosen R)

| Metric | Predicted | Actual IS | Pass |
|---|---|---:|---|
| Trades/day | ≥ 20 | 12.9 | ✗ |
| Trade WR (PnL ratio) | ≥ +0.5 | **−0.06** | ✗ |
| Mode $/trade bucket low | ≥ $10 | −$20 | ✗ |
| Mean $/trade | ≥ $10 | −$2.10 | ✗ |
| Daily mean PnL | ≥ $200 | −$27 | ✗ |
| Daily median PnL | ≥ $100 | −$47 | ✗ |
| Day WR (%) | ≥ 60% | 42% | ✗ |

**All 7 gates failed.** Cycle 2 hypothesis was WRONG.

### R sweep (IS)

| R | N | $/day | $/tr mean | $/tr med | tradeWR | DayWR | %≥$10 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| $2 | 4,416 | −$21 | −$1.08 | −$16 | −0.04 | 45% | 28% |
| $4 | 2,982 | −$27 | −$2.10 | −$20 | −0.06 | 42% | 30% |
| $6 | 2,050 | −$7 | −$0.74 | −$23 | −0.02 | 48% | 33% |
| **$10** | **1,132** | **+$7** | **+$1.06** | **−$25** | **+0.02** | **50%** | 35% |
| $15 | 654 | −$50 | −$9.17 | −$31 | −0.15 | 45% | 37% |
| $20 | 337 | −$114 | −$24.72 | −$40 | −0.31 | 42% | 35% |

Only R=$10 prints positive, and barely — mean $/tr = $1.06 (gate was $10). None meet criteria.

### Why the signal didn't translate

Cohen d = 1.96 at pivots, yes — but that's about the RESIDUAL DISTRIBUTION matching the pivot type. The zigzag definition already gives direction at a confirmed pivot. The residual just confirms.

Where the edge dies: **the 2R retracement tax.**
- Entry: we buy $R above the actual low (confirmation requires $R bounce)
- Exit: we sell $R below the next high (confirmation requires $R retracement)
- Net trade capture = (leg amplitude) − 2R
- At R=$4: mean leg barely exceeds $8, so mean trade ≈ $0 before commission
- At R=$10: fewer, bigger legs — but tax grows to $20; still only $1/trade net

**Accuracy ≠ tradeable edge.** 83.5% of pivots have direction matching the next leg; but most legs are too short to survive 2R tax.

### What the data says about the RM cord ceiling

Even at R=$10 (best case), the strategy captures ~$7/day IS against the $2.5–3K/day RM cord ceiling = **0.24%** capture. The 20–30% target is 100× away with this geometry.

---

## ACT

**Decision: ADJUST → open Cycle 3 with new hypothesis targeting the EXIT rule.**

Rationale:
- The entry trigger works (signal verified, pivot detection fires reliably)
- The direction rule works (83.5% of pivots lead to correct-direction legs)
- The EXIT is broken (waiting for next pivot costs $R in tax AND often exits in noise)

**Cycle 3 hypothesis candidates (user to pick before building):**

**3a. Asymmetric R**: enter at large R (e.g., $10 — strong pivots), exit at small R (e.g., $4 — faster). Saves half the tax.

**3b. Mean-cross exit**: exit when residual returns to zero (price reaches RM). No retracement tax on the exit — pure mean-reversion target. This is the "cusp to mean cross" idea from earlier in the session.

**3c. Fixed TP + max hold**: exit at +$X PnL or N minutes, whichever first. Decouples exit from zigzag geometry.

**3d. Leg-amplitude filter**: only enter when entry features predict leg > threshold. Skips noise pivots.

My vote: **3b (mean-cross exit)**. It's the cleanest physics — enter at the pivot (price extreme vs RM), exit when price returns to RM (residual = 0). No second zigzag confirmation needed. Matches the user's original sketch ("cusp → mean cross").

Pre-Cycle-3 measurement to inform the Plan:
- Distribution of leg amplitudes at each R (already have data — needs one chart)
- Fraction of RM-pivot-entered trades where residual returns to zero within N bars

Closing Cycle 2. Commit note: `tools/forward_pass_rm_pivot.py`, `training_RM_physics/output/trades/rm_is.pkl`, `research/rm_pivot/findings/2026-04-22_fwd_pass_rm_pivot.{md,png}`.
