# AG Phase 4 — Conditioning Sweep ("make each proposal more effective")

**Context**: all 18 dossiers are complete. Verdict of Phase 3B: NO concept
shows a replicated positive edge as its article states it, unconditionally.
That was expected — the measured precedent is that raw signals are noise
until conditioned (the house fade signal went PF 0.94 → 1.26 inside 9–13 CT).
Phase 4 finds each signal's OPERATING WINDOW.

## Prerequisite fixes (before any sweep)
- **P0 — Units**: FIB-17, PIVOT-16, VP-01, ORDERFLOW-14, SCALP-18 report
  magnitudes of ±10–19 while the rest clamp at ±2.05σ — they are not on the
  §7 σ-standard/clamping. Re-run those five to the standard before they
  enter any comparison.
- **P1 — Master index**: regenerate `AG_cat_00_INDEX.md` from the 18 DOCs
  (current one has 5 stale rows, a copy-paste "what it measures" column, and
  a degenerate Squeeze row).

## The sweep
For EVERY dossier setup, recompute the Phase-3B tables inside each bucket of
FOUR standard conditioners (same four for all signals, no bespoke ones):
1. **Hour-of-day (CT)**: {pre-7, 7–9, 9–11, 11–13, 13–15, 15+} — the
   conditioner already proven once.
2. **Regime**: efficiency ratio over trailing 60m — {churn (ER low),
   mixed, trend (ER high)} terciles. Do-not-lump rule.
3. **Volatility state**: trailing 1m regression σ percentile (day-relative)
   — {low, mid, high} terciles.
4. **Event depth**: how extreme the event itself was, in σ — terciles.

## Discipline (this is where sweeps die or lie)
- **Replication rule = the multiple-comparisons police.** ~18 signals ×
  ~2 setups × 4 conditioners × ~4 buckets ≈ 550 cells; at 5% chance ≈ 27
  fake "significant" cells PER YEAR. Requiring same-direction, comparable-
  magnitude effects in BOTH years cuts expected fakes to ~1. Only replicated
  windows are reported as findings; everything else is a table row.
- Per cell: N, PF-WR, EV(σ) + bootstrap CI, both years side by side. Cells
  with N < 30 per year: print but grey out (no interpretation).
- NO economics yet; NO verdicts beyond "replicated window found / none".
- Also carry forward, into the same conditioning grid, the Phase-3B
  **inversion-candidate flags** (ORDERFLOW-14 delta-div, RSI-06 bearish-div,
  VP-01 trend-runner) and the **stable-positive flag** (SEASON-12 Mon/Tue,
  +0.31 both years): does conditioning strengthen or dissolve them?

## Output
- Per dossier: `COND_<ID>.md` appended to its test folder — the 4 conditioner
  tables + a one-line summary ("replicated window: <desc>" or "none").
- Master: `reports/AG_cat_00_CONDITIONING.md` — every replicated window
  found, ranked by both-year EV, with its N. Expect a short list; empty is a
  valid outcome.
- These windows (if any) become interaction terms for the joint model and
  candidates for the ONE economics test that follows.

## Unchanged
σ-standard (§7), magnitude-to-resolution, 8:30 CT session context, GDP
dossier discipline, label-free, NT8 sealed.
