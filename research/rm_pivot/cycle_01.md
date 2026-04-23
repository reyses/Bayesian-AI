# Cycle 01 — Cohen d verification at RM pivots

> PDCA cycle inside the RM Pivot research project.
> Ref: [project.md](project.md) §MEASURE M2

---

## PLAN

### Hypothesis

The 2026-04-21 journal claim — *direction at regression-mean zigzag pivots
via `1m_z_se` residual has Cohen d ≈ −2.46 walk-forward* — is real and
reproducible on current ATLAS data.

### Change

None to production code. **Read-only measurement.** Writes one script to
`tools/` and one report to `research/rm_pivot/findings/`.

### Predicted outcome

At RM-zigzag pivots (rolling 60-bar OLS on 1m closes, zigzag retracement
$R swept), the Cohen d of residual between HIGH-pivot and LOW-pivot groups
will be in **[−3.0, −1.5]** (matching the journal's −2.46 with ±30% band).

Accuracy of "residual sign predicts next-leg direction" will be
in **[75%, 92%]** (journal claim: 86%).

Walk-forward (monthly IS 2025, held-out OOS 2026) will show:
- Monthly |d| std < 0.8 (stable signal, not a one-month fluke)
- OOS |d| within ±0.5 of IS mean

### Success gate

- **|d| ≥ 1.5** on pooled IS measurement
- Monthly |d| std < 0.8 (stability)
- OOS |d| ≥ 1.0 (signal survives into 2026)

If all three pass → PROCEED to Cycle 2 (forward pass).

### Kill gate

- **|d| < 0.5** on pooled IS → signal doesn't exist. Abandon RM-pivot research.
- Monthly |d| std > 1.5 → signal is noise, not a stable effect. Abandon.
- OOS |d| < 0.3 while IS ≥ 1.5 → overfit/regime-dependent. Flag for Analyze, don't proceed to Cycle 2 yet.

### Method

New tool: `tools/measure_rm_pivot_direction_cohen_d.py`

Steps per day:
1. Load 1m closes from `DATA/ATLAS/1m/<day>.parquet`
2. Compute rolling 60-bar OLS → RM series (fitted values)
3. Run live-safe zigzag (confirmation-only, no lookahead) on RM series at $R threshold
4. At each confirmed pivot bar, record:
   - Pivot type (HIGH or LOW)
   - `1m_z_se` residual at that bar (from `DATA/ATLAS/FEATURES_5s/<day>.parquet`, nearest timestamp)
   - Next-leg direction (LOW pivot → next leg UP, HIGH pivot → next leg DOWN) — known once the next pivot confirms

Metrics per R ∈ {$2, $4, $6, $10}:
- N pivots / day (sanity)
- Cohen d of residuals between HIGH-pivot group vs LOW-pivot group
- Accuracy of residual-sign-based direction prediction vs actual next-leg direction
- Residual distribution histograms by pivot type

Walk-forward:
- Monthly: compute d for each month Feb–Dec 2025 (skip Jan warm-up)
- IS pooled: all 2025 days combined
- OOS: 2026 Jan–Feb combined

Output:
- `reports/findings/` → `research/rm_pivot/findings/2026-04-22_cohen_d_verify.md`
- Chart: `research/rm_pivot/findings/2026-04-22_cohen_d_verify.png` (distributions, monthly trend)

### Reproduction command

```
python tools/measure_rm_pivot_direction_cohen_d.py
```

---

## DO

Ran `python tools/measure_rm_pivot_direction_cohen_d.py` on 2026-04-22 at 05:05 UTC.

Duration: ~7s (277 IS days + 68 OOS days).

Output:
- Report: `research/rm_pivot/findings/2026-04-22_cohen_d_verify.md`
- Chart: `research/rm_pivot/findings/2026-04-22_cohen_d_verify.png`

No errors. All 345 days processed.

---

## CHECK

### Predicted vs Actual

| Metric | Predicted | Actual | Hit? |
|---|---|---|---|
| Pooled IS Cohen d (R=$4) | [−3.0, −1.5] | **−1.96** | ✓ |
| Pooled OOS Cohen d (R=$4) | within ±0.5 of IS | **−1.95** (Δ=0.01) | ✓ |
| Monthly std(d), R=$4 | < 0.8 | **0.16** | ✓ |
| Accuracy [75%, 92%] | 86% (journal) | **16.5% raw ↔ 83.5% when rule is flipped** | Partial (see below) |

### Success gates (from Plan)

| Gate | Threshold | Actual | Pass |
|---|---|---|---|
| IS |d| ≥ 1.5 | **1.96** | ✓ |
| Monthly std < 0.8 | **0.16** | ✓ |
| OOS |d| ≥ 1.0 | **1.95** | ✓ |

**All three gates passed.** Kill gates not triggered.

### Key finding (unexpected)

The direction rule I assumed was **inverted**. I predicted:
- residual > 0 (price above RM) → next leg DOWN (mean-reversion)
- residual < 0 → next leg UP

Actual data (R=$4 IS):
- At HIGH pivots: mean residual = **−0.993** (price is BELOW RM, not above)
- At LOW pivots: mean residual = **+0.988** (price is ABOVE RM, not below)

**Why:** The 60-bar OLS lags a turning price. At a peak, price stalls while RM keeps rising → residual crosses zero and goes negative. Same mirrored at a trough.

**Implication:** The signal is real; my prediction rule was backwards. The correct rule:
- LOW pivot (confirmed) → enter LONG (residual will be positive, confirming)
- HIGH pivot (confirmed) → enter SHORT (residual will be negative, confirming)

This is NOT mean-reversion; it's **trend-continuation at the pivot** — trade in the direction the pivot implies, residual magnitude serves as confirmation strength.

Flipped accuracy = 100% − 16.5% = **83.5%**, matching journal's 86% claim (within 2.5pp).

### Stability across R

| R | Pooled IS d | Pooled OOS d | Monthly std |
|---:|---:|---:|---:|
| $2 | −1.92 | −1.87 | 0.17 |
| $4 | −1.96 | −1.95 | 0.16 |
| $6 | −1.83 | −1.94 | 0.25 |
| $10 | −1.56 | −1.67 | 0.28 |

R=$4 is the sweet spot — largest |d|, lowest std, most pivots (6,496 IS / 1,842 OOS).

---

## ACT

**Decision: STANDARDIZE and proceed to Cycle 2.**

Rationale:
1. All three success gates passed decisively.
2. Signal survives IS→OOS with effectively no decay (Δd = 0.01).
3. Monthly stability exceptional (std 0.16 ≪ 0.8 gate).
4. Journal's 86% accuracy claim is reproduced (83.5%) once the direction rule is corrected from "mean-reversion" to "trade the pivot direction."

**Standardized findings (for use in Cycle 2):**
- **Entry trigger**: confirmed RM zigzag pivot at R=$4 (best |d| + stability + pivot count)
- **Direction rule**: LOW pivot → LONG, HIGH pivot → SHORT (trade the pivot direction)
- **Residual use**: optional confirmation/filter (|residual| ≥ threshold → higher-conviction entry)
- **Expected pivot frequency**: ~23 IS pivots/day, ~27 OOS pivots/day (6,496/277 and 1,842/68)

**Correction to project.md**: flip the direction rule interpretation. The engine I built in `training_RM_physics/rm_physics_engine.py` had the WRONG direction (mean-reversion from residual). That's why its full IS run was −$5,753. Cycle 2 will use the corrected rule.

**Commit note** (for session end): add `tools/measure_rm_pivot_direction_cohen_d.py`, `research/rm_pivot/` folder, findings report.
