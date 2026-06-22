# WORK PACKAGE → Gemini — Genuine-1s F-space segment redesign (2026-06-17)

SUPERSEDES `comms/TASK_segment_1s_test_2026-06-16.md` (that used the 5s base family — WRONG; the user
wants GENUINE 1s-resolution features). Claude ran the first cut; you take it deeper.

## Thesis (settled tonight)
The V2 F-space treats scale as an ordinal/categorical variable (8 TFs) when it's a continuous CARDINAL
quantity (window length). Each TF feature is a half-measurement of ONE signal (sample-starved, clock-phase
-contaminated, OHLC-lossy). Stacking collinear slices and regressing them as if independent => the mess.

## First cut (Claude, DONE) — day 2025_02_20, target = fwd-5min return on 1s close
Genuine 1s features computed via the canonical SFE kernels on 1s bars (windows {60,300,1800}s, vwap dropped)
and SAVED to `DATA/ATLAS/FEATURES_1s_TRUE/2025_02_20.parquet`. Script: `research/segment_1s_test.py`.
| metric | A: genuine-1s (56) | B: TF-mix (192) |
| collinearity cond# | 5.1e16 | 3.6e29 |
| beta stability (halves corr) | +0.96 | +0.20 |
| fit R2 (Ridge 5cv) | -2.82 | -45.65 |
**Read:** thesis CONFIRMED on the instrument (TF-mix ~13 orders more collinear, betas unstable 0.20 vs 0.96).
BUT both R2 NEGATIVE — clean ruler removed the confound, did NOT reveal signal yet. Necessary, not sufficient.

## Your work (deeper, the real test)
1. **Orthogonalize** the genuine-1s scale-space — A is still collinear (5e16) because overlapping windows
   aren't independent. Build a genuinely decorrelated basis: PCA/whiten the 1s scale-sweep, OR wavelet/
   scale-space packets (orthogonal across scales by construction). Re-measure cond# (should drop to ~O(10s)).
2. **Use the segment regression's REAL target/method**, not fwd-return — the contemporaneous price-delta fit
   + the PRISTINE/RECOVERED/CHAOS tiering (the stage1 approach), on the genuine-1s (orthogonalized) features.
3. **Window sweep = the response surface.** Compute genuine-1s features over a LOG-SPACED window set; fit the
   signal-vs-horizon surface; **the deliverable is the IS vs OOS surface OVERLAID** — reproduces => real
   horizon; peak moves => period-luck. (cardinal continuous factor, face-centered/RSM — interpolatable.)
4. **IS/OOS over several days**, not one (today was a single-day collinearity/stability read).

## DISCIPLINES (per comms/CONTEXT_FOR_GEMINI.md)
- Causal / no lookahead. SEGMENT FIREWALL (segments non-causal, label-side only, diagnostic).
- **Report DISTRIBUTIONS + MODE, never averages** (user, 2026-06-16). Lead with histogram/mode; mean only w/ CI.
- NT8 commission CONFIRMED = $0.78 ROUND-TRIP (+ ~$1.00 assumed slippage) for any $ metric. No magic numbers.
- **PUSH BACK** if the orthogonalization or the A-vs-B comparison is flawed (e.g. unfair feature counts,
  leakage, wrong target). Catching a flaw > running it.
- Write results to `reports/findings/`; update `docs/daily/INDEX.md`; **close with SUMMARY + LOCATION**.

## Honest framing (don't over-claim — Claude's bias note applies to you too)
The first cut confirms the *instrument* was a collinear, unstable mess. It does NOT show signal — both arms
predicted worse than the mean. The vol-ceiling and base-rate kills still stand. The prize is settling
"was it our collinear ruler, or no signal?" — only an ORTHOGONAL, full-resolution ruler + the IS/OOS surface
overlay answers that. If it's still inconclusive OOS with a clean orthogonal basis, the answer is "no signal"
and we accept it. Build it to find out, not to confirm a hope.
