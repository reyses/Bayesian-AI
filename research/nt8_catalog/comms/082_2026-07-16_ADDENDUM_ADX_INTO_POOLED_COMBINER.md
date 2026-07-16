# ADDENDUM — ADX-08 folded into the pooled combiner (13 streams)
**Doc:** 082 · **Date:** 2026-07-16 (overnight, follows 081) · **Author:** Claude · **Status:** FINAL

Completes the "mix ALL the signals" directive: ADX-08's doc-079 feature rows
(N=1,359) converted to the shared schema (`signal_rows_ADX08.parquet`) and pooled.

## Result — honest null on incremental lift
- 13-stream pooled OOS AUC **0.685** vs 0.687 without ADX (N 156,119 vs 154,760).
  No incremental lift: ADX's fires are 0.9% of the pool and its predictive content
  (leg alignment, pivot age, the age×alignment inversion) is already carried by the
  shared causal features — `is_ADX08` coef +0.017 ≈ zero.
- Decile calibration unchanged and still monotone-honest (0.28→0.25 ... 0.82→0.80);
  tails still 0.25 [0.23,0.26] / 0.80 [0.79,0.81].
- Interpretation: ADX-08 remains a good STANDALONE calibrated signal (doc 079, AUC
  0.660) but is REDUNDANT inside the mixer. This is the expected behavior of a pooled
  model when a small stream's features are shared — not a retraction of doc 079.

## State at close of the autonomous night
- League (12 streams + ADX): `reports/dossier_signal_league.md` (doc 081 §2).
- Combiner (13 streams): `reports/combiner_preview.md` — pooled P(right), OOS-honest.
- All tools committed; row parquets regenerable via `dossier_signal_pipeline.py`.
- Safety cron deleted (task completed before the 03:28 net was needed).
- Open for Moises' morning review: doc 081 §7 proposals (economic conversion,
  phase-conditioned zigzag, overfit-decay shelf-life, Mamba state-vector handoff).
