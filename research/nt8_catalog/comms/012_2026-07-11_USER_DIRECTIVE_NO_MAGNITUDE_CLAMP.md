# USER DIRECTIVE + Reviewer Amendment — σ-normalize YES, clamp NO
**Doc:** 012 · **Date:** 2026-07-11 · **Author:** Claude (reviewer), directive from Moises · **Status:** FINAL
**Amends:** doc 008 mod #1 and the P0 wording in `reports/AG_PHASE4_CONDITIONING_SWEEP.md`.

## Moises' ruling
Clamping magnitudes at ±2.05σ caps the magnitude distribution and is
detrimental to the downstream binary logistic layer. Do not cap.

## Why he's right (reviewer concurrence)
- The clamp truncates the fat tails — the ONLY validated edge family in this
  repo is continuous sizing on unclamped amplitude (B9); tails are the signal.
- Magnitude-weighted logistic fits degenerate under a clamp (every resolved
  winner's |magnitude| ≈ 2.05 → weights collapse to a constant).
- MVP §5 explicitly mandates winner/loser magnitude histograms/KDE "to visually
  expose fat tails and skews" — clamping defeats the protocol's own requirement.
- The Event-Depth conditioner (terciles of event magnitude in σ) needs the
  continuous distribution.

## Amended P0 standard (supersedes "σ-standard/clamping")
Two separable layers per event — record BOTH:
1. **Binary resolution outcome** (for cross-dossier hit-rate comparability):
   first touch of symmetric ±2.05σ barriers, exactly as in OHLC-01/ORB-02.
   This is a RESOLUTION RULE, not a magnitude cap.
2. **UNCLAMPED σ-normalized magnitudes** (for EV, distributions, logistic /
   F-space layers): `magnitude_sigma`, `mfe_sigma`, `mae_sigma` — divided by
   the §7 trailing 1m regression residual σ, NOT clipped. Causality unchanged:
   the measurement window still ends at the resolution bar (§7 no-post-
   resolution-peeking rule stands — that rule prevents lookahead, it does not
   require capping).

All P0 re-runs (the correct five: FIB-17, PIVOT-16, VP-01, ORDERFLOW-14,
SCALP-18) and any already-converted dossiers must ship BOTH layers in
`events.parquet`. The conditioning sweep's EV tables use the UNCLAMPED σ
magnitudes; hit-rate columns use the binary resolution outcome. Everything
else in docs 008/011 stands.
