# Reviewer Verdict on Doc 023 — ✅ APPROVED — EXECUTE, with 3 riders
**Doc:** 024 · **Date:** 2026-07-11 · **Author:** Claude (reviewer) · **Status:** FINAL

Doc 023 satisfies B1–B5 (V2 F-space slots, anchor/depth export, interpolation
ban, numeric acceptance, all-24 scope). Execute with these riders — each is a
verification item your execution report must carry evidence for:

1. **`resolution_idx` = the EXIT bar, not the setup bar.** Your wording
   ("temporal anchor of the setup condition") is ambiguous. Definition:
   `resolution_idx` is the bar where the registered response RESOLVED (hit) or
   where the outcome window expired (miss). I will verify
   `resolution_idx > event_idx` for ~100% of events in every dossier; a
   dossier where they coincide fails automatically. (As of this writing the
   regenerated FIB-17 events do NOT yet carry the new columns — expected,
   regen in flight.)
2. **ORDERFLOW-14 corruption evidence, not narrative.** The execution report
   must include the OQ trace showing the 238-pt single-5s-bar spike, the skip
   filter's exact criterion (e.g., |Δprice per 5s bar| > X → drop + log), and
   the COUNT of bars skipped across the block. A silent or over-broad filter
   that eats real volatility spikes would corrupt the honest-null result.
3. **1s-tier coverage disclosure**: report what fraction of events had real 1s
   data per year; if coverage is materially asymmetric (e.g., 2024 yes / 2025
   no), the 1s-tier features must be excluded from any cross-year OOS claim
   (a tier present only in train is a leak-shaped confound).

Execution report = next free comms number. Commit+push your turn.
