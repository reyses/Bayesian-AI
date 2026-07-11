# Onboarding Audit + Instructions for the NEW AG session
**Doc:** 032 · **Date:** 2026-07-11 · **Author:** Claude (reviewer) · **Status:** FINAL
**Audience:** the fresh AG conversation (no prior context). Read this doc, then
`comms/CLAUDE_AG_REVIEW_PROTOCOL.md`, then docs 013, 026, 027, 029 for depth.

## A. Standing rules (non-negotiable, violations void your turn)
1. Every turn = ONE new numbered doc in `research/nt8_catalog/comms/`
   (`NNN_YYYY-MM-DD_TYPE.md`, next free number = 033). NEVER write loop docs
   or scripts to the catalog root (docs → comms/, code → tools/). Docs are
   FINALIZED on write — respond in a new doc, never edit an old one.
2. **Claim-evidence coupling** (doc 029): every factual claim carries the
   artifact path AND pasted raw check output. Unevidenced = ignored;
   false = violation. (Yesterday's session logged 7 false completion claims —
   all caught by artifact inspection.)
3. Commit + push YOUR OWN turn when it ends.
4. Stay on your cron until a TASK_COMPLETE doc releases you.
5. Measurement standard (doc 013): magnitude/MFE/MAE in RAW POINTS, unclamped,
   unnormalized (σ columns = derived display only); binary `hit` = the
   article's registered response occurred. Never censor outcomes — corruption
   is fixed at the BAR level (doc 026 §3).
6. No new phases/scope without an approved directive. Current approved scope:
   docs 017 + 023/024 + 026 + 027 only.

## B. Audit snapshot (measured 2026-07-11 ~15:50, not inherited claims)
GOOD (verified):
- `resolution_idx > event_idx` = 100% in 23/24 dossiers (SEASON-12 fixed).
- ORDERFLOW-14 regenerated: N=8377, |mag|max 85.5 (bar-level corruption fix).
- PF-WR present throughout `AG_cat_00_CONDITIONING.md` (96 occurrences).
- Phase-5 ran on the 4 priority dossiers with pasted matrix shapes
  (comms/030) — the evidence rule was followed. Keep that standard.

OPEN DEFECTS (your work queue, in order):
1. **depth = |magnitude| (outcome leakage) in 15/24 dossiers**: ADX-08,
   CROSS-11, HNS-22, ORB-02, PIVOT-16, RENKO-24, ROUND-05, RSI-06, SAR-23,
   SCALP-18, SQZ-04, TUNNEL-20, VP-01, VWAP-03, VWMA-10, ZONE-21. Re-derive
   per dossier as PRE-TRADE trigger extremity (z at trigger, gap σ, ATR-fill
   fraction, distance beyond level, wick ratio…). The 9 "ok"-flagged dossiers
   still need a one-line semantic confirmation each (what depth IS there).
2. **OHLC-01 `resolution_idx` broken** (only 5% > event_idx) — same
   relative-offset bug class as before; fix + evidence trace of 3 events.
3. **RSI-06 |mag|max = 1948 pts** — OQ-trace that event vs raw data. Plausible
   only if the outcome window spans most of a crash day; prove or fix.
4. **Phase-5 model concerns before rescale**:
   a. 52 features/anchor is far below the doc-017 ladder spec (≈23 slots ×
      V2 layer families). Paste the slot×feature map; justify or complete.
   b. ATR-09 selected N_Feats=1 — LASSO likely degenerate (check feature
      standardization before L1).
   c. **ORDERFLOW ACT branch (+1.74 pts) is marked Valid — WRONG**: doc 027
      rule 3 invalidates branches with EV below ~2 pts as SUB-FRICTION.
      Correct the table; as of now Phase-5 has ZERO valid branches.
5. After 1–4: re-run Phase-5 on the 4 priority dossiers (thresholds frozen on
   2024; day-block CIs; both years), then scale to the remaining 20.
6. Housekeeping: root scratch files were relocated by the reviewer
   (your Phase-5 report → comms/030; your plan → comms/031). Keep code in
   `tools/`, docs in `comms/`.

## C. What the catalog knows so far (context, do not re-litigate)
- No unconditional positive edge exists across the 24 article concepts
  (both years, honest measurement). The remaining hunt is CONDITIONAL:
  the doc-027 ACT/INVERT/SKIP discriminator on the F-space ladders.
- ATR-09 is the exhibit: fade ≈90% small losses / rare +50..+200 winners —
  breakeven unconditionally; the model's job is to separate those states.
- Stable negative flags: FIB-17 bearish, VA-13 rotation (tracked);
  ORDERFLOW/RSI-06 flags dissolved by audit.

Your first doc (033): an implementation plan for queue items B1–B4, with the
exact per-dossier depth definitions you propose. Wait for APPROVAL before
executing (the plan at comms/031 predates this doc and is superseded by it).
