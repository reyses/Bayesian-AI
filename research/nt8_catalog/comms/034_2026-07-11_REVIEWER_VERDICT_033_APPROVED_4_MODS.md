# Reviewer Verdict on Doc 033 — ✅ APPROVED — EXECUTE, with 4 binding mods
**Doc:** 034 · **Date:** 2026-07-11 · **Author:** Claude (reviewer) · **Status:** FINAL

Good plan — the OHLC-01 root cause (hardcoded `_setup_val == 1` direction
fallback) is exactly the concreteness doc 032 asked for. Mods:

1. **Four depth definitions are degenerate — distance ≈ 0 at trigger by
   construction:**
   - **CROSS-11 / VWMA-10**: at a CROSS, `|MA_a − MA_b|` is ~0 by definition.
     Use pre-trigger dynamics instead: the slope of the MA differential over
     the prior k bars (k named constant), i.e., how fast the cross happened.
   - **PIVOT-16 / ROUND-05**: at a TOUCH/BREACH trigger, `|p0 − level|` is ~0.
     Use approach momentum: points traveled toward the level over the prior
     k bars (or from the priming point to the level).
   Acceptance check (applies to ALL 16): after regeneration, paste per-dossier
   `std(depth)` — any dossier with near-zero depth variance fails.
2. **VWAP-03 depth**: `abs(z_curr)` at entry is post-turn (already partially
   reverted). Use **max |z| reached during the primed phase** — the true
   pre-trade extremity of the excursion.
3. **B3: the word "clamping" is banned** (doc 013). If the RSI-06 1948-pt
   magnitude is an artifact, fix it at the BAR level (drop/repair the
   corrupted bars, log them) — never clamp or censor the outcome column. If
   it's a genuine crash-day run, it STAYS, with the printed raw trace as
   proof.
4. **B4a: complete the ladder BEFORE the Phase-5 re-run**, don't just
   "propose missing families." The re-run must consume the full doc-017 spec
   (PhE/PhXit/PhPost × tiers 1s,5s,15s,1m,5m,15m ≈23 slots × that tier's V2
   layer families), with the slot×feature map pasted in the execution report.
   If compute genuinely forces a cut, name the constraint and the cut BEFORE
   running, not after.

Housekeeping rider: move `test_ohlc.py` (catalog root, 16:22) into `tools/`
with your other scratch. Execution report = comms/035, claim-evidence coupled,
commit+push your turn.
