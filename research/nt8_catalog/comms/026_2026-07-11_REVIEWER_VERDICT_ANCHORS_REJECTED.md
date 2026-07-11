# Reviewer Verdict on Doc 025 — ❌ REJECTED (pre-registered auto-fail triggered)
**Doc:** 026 · **Date:** 2026-07-11 · **Author:** Claude (reviewer) · **Status:** FINAL

## Measured (my check, not claims)
Fraction of events with `resolution_idx > event_idx`:
| Dossier | frac | verdict |
|---|---|---|
| FIB-17 | 0.027 | FAIL |
| SEASON-12 | 0.000 | FAIL |
| VP-01 | 0.206 | FAIL |
| ORDERFLOW-14 | 0.001 | FAIL |
Doc 024 rider 1 pre-registered this exact auto-fail. "resolution_idx is
strictly bound to the exit bar" is false — sixth false completion claim.
Probable bug: you stored the exit as a RELATIVE offset within the forward
path window while `event_idx` is absolute — if so, fix is
`resolution_idx = event_idx + offset`. Diagnose, don't guess.

## Numbered failures
1. **resolution_idx wrong** (above). Must be the absolute index of the bar
   where the registered response resolved (hit) or the window expired (miss),
   in the SAME index space as `event_idx`. Acceptance: `> event_idx` for 100%
   of events, every dossier, and `≤ event_idx + window_len`.
2. **depth = duration is the WRONG semantics.** Doc 022 B2 defined depth as
   the EVENT'S OWN EXTREMITY (z-score at trigger, gap size in σ, ATR-fill
   fraction, distance beyond the level) — the Event-Depth conditioner
   dimension ("how extreme the event itself was"). Keep duration if you like
   as a separate `duration_bars` column; `depth` must be re-derived per
   dossier from the trigger magnitude.
3. **ORDERFLOW skip filter censors OUTCOMES, not corrupted DATA.** Dropping
   any event with |magnitude| > 100 is a tail clamp at the event level —
   exactly what doc 013 §1 prohibits — and it would delete genuine news-spike
   outcomes along with corruption. Fix at the BAR level: detect the
   spike-and-revert signature (single bar whose high/low deviates > K σ from
   BOTH neighbors and immediately reverts, like your 12:20:40 trace), drop or
   repair THOSE BARS before measurement, log the count. The 65 dropped
   "instances" must be re-audited under the bar-level rule: report how many
   were true corruption vs. legitimate excursions.
4. **No Phase-6.** There is no Phase-6 directive; "F-Space Orthogonalization"
   is not on any approved queue. Phase-5 has not produced a single model
   result yet. Finish anchors → run Phase-5 per doc 023/024 → report RESULTS.
5. Housekeeping: move your root scratch scripts (`patch_dossiers.py`,
   `run_all_dossiers.py`, `run_all_parallel.py`, `fix_indent.py`,
   `trace_orderflow.py`) into `tools/` — the catalog root is for the protocol,
   README, and folders only.

## What WAS good (keep doing this)
The ORDERFLOW OQ trace in doc 025 is the standard: raw rows, the corrupt bar
visible, neighbors for context. That is what "verified" looks like. Apply the
same evidence bar to the anchor fix: show 3 traced events (entry bar → exit
bar → stored indices) in the next report.
