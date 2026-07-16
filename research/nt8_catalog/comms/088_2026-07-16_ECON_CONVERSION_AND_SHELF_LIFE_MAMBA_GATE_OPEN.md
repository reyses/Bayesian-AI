# Economic conversion + shelf-life — VERIFIED; the Mamba gate is OPEN
**Doc:** 088 · **Date:** 2026-07-16 · **Author:** Claude (reviewer) · **Status:** FINAL
**Executors:** Opus worker (econ conversion), Sonnet worker (overfit-decay) — ladder
trials #2 and #3, specs per doc 087 §4 queue.

## 1. Reviewer verification
- **Econ conversion**: headline cells reproduced EXACTLY from econ_drift_rows.parquet
  with independent code + day_block_ci(4000): D9@5m median +3.25 / mean +3.86
  [+2.46,+5.12]; D0-inverted@5m +1.00 / +1.33 [+1.00,+1.65]. Raw-data spot-check:
  15/15 fires' drift_5m match the 5s closes (1 apparent mismatch was the 15:15
  truncation cap working as specified — fire at 15:12 CT). ✓
- **Shelf-life**: window 2024-01-01 recomputed from overfit_decay_rows.parquet:
  initial_edge 0.2169 ✓, shelf-life 81 weeks ✓ (exact). ✓
- Both workers: deviations declared, kill-points evaluated honestly, nothing
  committed, no tuning past a gate. Ladder verdict: 3 trials, 3 passes.

## 2. ECONOMIC CONVERSION (test 2025+26, N=401,330 fires, day-block CIs, NO stops)
**P(right) converts to points — monotonically, with the sign flipping at the
calibration midpoint.** P was fit to LABEL agreement, never to price → the price
linkage is independent confirmation, not circularity.
- **The clean cell: top decile @ 5m — mode +1.0, median +3.25, mean +3.86 pts
  [+2.46,+5.12] (+$7.72/fire gross; +3.26 pts net of the 0.6-pt friction line;
  N=40,132).** A genuine distributional shift — mode AND median positive — not an
  outlier tail. ~63 such fires/day on average.
- Also significant: top decile @ 1m (+1.18 [+0.70,+1.68]); bottom decile INVERTED
  @ 5m/15m/30m (+1.33/+1.82/+1.83, all CIs > 0) — though these are tail-driven
  (modes ≈ 0); the typical inverted fire barely covers friction.
- **Horizon fragility is the trade-management story**: every top-decile cell at
  15m+ is NOT significant (30m CI [−5.34,+8.58]). The drift must be harvested in
  the 1-5m window — exactly the job the Mamba owns. 60m numbers additionally
  biased low by 13.5% session-end truncation.
- Anti-doom framing: these are exploration-level drift numbers (no stops, no
  management, pseudo-replicated fires mitigated by day-block CIs) — NOT a $/day
  claim. No deployment claim is made here.

## 3. SHELF-LIFE (doc 075 standard, 27 windows × 2 training lengths)
- 8-week windows: uncensored median 37 weeks (mode 7, N=14); **13/27 windows
  NEVER decayed** below 70% of initial edge within their horizon (right-censored
  → population truth is LONGER). 16-week: median 41, mode 57, 12/27 censored.
- Initial edge stable across every window (0.14-0.24) — the combiner is not a
  fragile overfit. Practical rule: **monthly retune is amply safe**; even the
  worst observed window (7 weeks) is covered.
- Watch item: 2025-fit windows skew shorter among the uncensored — partly a
  censoring artifact (short horizons), revisit at the next quarterly pass.

## 4. GATE DECISION
**OPEN.** The funnel Moises specified (extract → mix → hand to Mamba) has cleared
its evidence bar: honest calibrated P(right) at scale (AUC 0.689, diagonal
calibration, N=714k), economically convertible tails (top decile +3.86 pts @5m,
sig), and a ~9-month shelf-life. Next artifact: the **Mamba state-vector handoff
spec** — per-stream time-since-fire × direction × P + pooled P(right) as RL input
features, short-horizon (1-5m) management objective. Design doc first (per
protocol), then implementation.

## 5. Artifacts
reports/econ_conversion.md + econ_drift_rows.parquet + econ_run.log;
reports/overfit_decay.md + overfit_decay_rows.parquet + overfit_decay_run.log;
tools/econ_conversion.py + tools/overfit_decay_sweep.py.
