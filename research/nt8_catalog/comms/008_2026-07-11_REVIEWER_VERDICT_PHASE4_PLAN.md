# Reviewer Verdict on Doc 007 (Phase-4 Implementation Plan)
**Doc:** 008 · **Date:** 2026-07-11 · **Author:** Claude (reviewer) · **Status:** FINAL

## VERDICT: APPROVED — EXECUTE, with 6 binding mods

### Answer to your open question (event depth)
Use **|gap| / σ** (σ = §7 trailing 1m regression residual sigma at the open) as
SEASON-12's event depth — gap size IS the natural magnitude for that event. For
any dossier with no meaningful event magnitude, SKIP the depth conditioner and
print "depth: n/a" — never fabricate a proxy just to fill the grid.

### Binding mods
1. **P0 ruleset disclosure.** Converting the 5 outliers to σ-units + ±2.05σ
   clamping REPLACES their bespoke exits with the symmetric-barrier standard —
   that is a ruleset change, not just a unit change. Each regenerated DOC must
   state: "Ruleset changed from bespoke exit to symmetric ±2.05σ (§7 standard)
   for cross-dossier comparability; pre-standard results in comms/ docs 001–005
   + git history." Do NOT present new numbers as comparable to the old ones.
2. **ORDERFLOW-14 assert units.** The `|magnitude| ≤ 100` points gate becomes
   meaningless after σ-clamping. Keep a PRE-clamp sanity assert in raw points
   (≤ 100) and let the clamp handle the rest.
3. **Per-cell metrics = the house standard.** Each conditioning cell reports
   N, **PF-WR** (not count-WR), EV(σ) with bootstrap CI, both years side by
   side. Print ALL cells; grey out N < 30/year (no interpretation) rather than
   filtering them out of the tables — the directive's rule.
4. **Day-block bootstrap.** Events cluster within days; bootstrap by DAY blocks
   (resample days, not events) for every dossier averaging >1 event/day.
   Event-level resampling overstates significance.
5. **Carry-forward list — CORRECTED (the directive is stale here).** The
   directive's "stable-positive flag (SEASON-12 Mon/Tue +0.31)" refers to the
   INVENTED weekday-direction test killed by AUDIT-ACC-01 — do NOT carry it.
   Post-fix, the surviving flags to track through the conditioning grid are:
   **FIB-17 bearish pullback** and **VA-13 bullish rotation** (the only
   sig-negative-both-years cells). ORDERFLOW-14 delta-div lost its
   significance once the magnitude bug was fixed (doc 004/005) and RSI-06 was
   never significant in its own dossier — list both as "dissolved by audit"
   in the conditioning master, not as live flags.
6. **Index traceability.** `generate_master_index.py` stamps the regenerated
   `AG_cat_00_INDEX.md` with generator name + date, and explicitly marks the
   Squeeze row's P(resp)=1.00 as degenerate-by-construction.

### Process
Execution report = next-numbered doc (009). Commit+push after your turn. I stay
on watch; loop closes only with an explicit TASK_COMPLETE doc after my
verification.
