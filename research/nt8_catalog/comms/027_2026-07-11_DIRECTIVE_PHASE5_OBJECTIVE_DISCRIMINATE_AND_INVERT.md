# Phase-5 Objective Amendment — discriminate good/bad trades AND complete/invert the signal
**Doc:** 027 · **Date:** 2026-07-11 · **Author:** Claude (reviewer), directive from Moises · **Status:** FINAL
**Amends:** doc 017 (purpose section). All other Phase-5 mechanics (docs 023/024/026) unchanged.

## The point of the regression (Moises, 2026-07-11)
The logistic is not just a characterization exercise. Its purpose is:
1. **Separate good trades from bad** within each signal's event population —
   P(registered response | F-space ladder at t(e)).
2. **Complete the signal where the raw claim is dead or backwards.** Exhibit:
   ATR-09 (DOC-ATR-09_distributions.png) — the fade loses ~90% of events in a
   tight −10pt spike while rare winners pay +50..+200; ≈ breakeven both ways
   round. The tradable object is CONDITIONAL: F-space states where the fade
   catches the turn vs states where the extension keeps running (= the NMP
   master equation: |extension| extreme + λ<0 → fade; λ>0 → ride).

## Deliverable per dossier: a three-way policy, not an AUC
From the entry-anchor model, derive two thresholds on P(response):
- **ACT** (P ≥ p_hi): trade the signal as the article states it.
- **INVERT** (P ≤ p_lo): trade the OPPOSITE side — a reliably-failing signal
  is itself a signal.
- **SKIP** (between): no trade.
Report per branch, per year: N, PF-WR, EV in RAW points with day-block
bootstrap CI, plus the branch's magnitude histogram (mode first).

## Anti-self-deception rules (binding)
1. **Thresholds p_hi/p_lo are selected on 2024 ONLY and frozen**, then
   evaluated ONCE on 2025. Threshold shopping on 2025 = the run is void.
   (MVP §6 warning: inversion-picking on the same data manufactures
   inversions by chance.)
2. An ACT or INVERT branch is a FINDING only if its 2025 EV CI excludes 0 AND
   the branch direction matches 2024. Otherwise it is a table row.
3. The magnitude model (raw points) rides along to size the claim: a branch
   with significant hit-rate but mode ≈ friction (< ~2 pts) is flagged
   SUB-FRICTION, not tradable.
4. Priority order for first results: ATR-09 (the exhibit), FIB-17, VA-13,
   then the rest of the 24.

## Sequencing unchanged
Doc-026 fixes (exit anchors, depth semantics, bar-level corruption filter)
remain the gate before any Phase-5 model run.
