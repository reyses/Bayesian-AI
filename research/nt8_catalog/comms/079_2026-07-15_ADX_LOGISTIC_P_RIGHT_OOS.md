# ADX binary logistic — calibrated P(right), OOS: AUC 0.66, tercile spread 39%→74%
**Doc:** 079 · **Date:** 2026-07-15 · **Author:** Claude (executor) · **Status:** RESULTS
Tool: `tools/adx_prob_logistic.py` · rows: `reports/adx_signal_features.parquet`
The stage-0 pipeline (overlap → transition profile → features → logistic), first signal.

## Setup
One row per ADX signal (doc-074 setting). Features: causal zigzag pivot-age (streaming
1m-ATR(14)x4 v1), signal-with-leg alignment, ADX value, time-of-day, age×alignment
interaction. Target = agreed with active AI label. Train 2024 (N=688), test 2025+2026
(N=671). No tuning on test.

## OOS results (2025+2026, day-block CIs)
- **AUC = 0.660** (predicting per-signal direction correctness)
| P-tercile | N | observed agreement | CI | mean predicted P |
|---|---|---|---|---|
| low | 224 | **0.39** | [0.30, 0.49] | 0.30 |
| mid | 223 | 0.62 | [0.56, 0.67] | 0.63 |
| high | 224 | **0.74** | [0.65, 0.83] | 0.79 |
- **35pp OOS spread** high-vs-low, non-overlapping CIs. Calibration decent (predicted ≈
  observed per tercile).
- **Low tercile is BELOW 0.5** ⇒ inverting those signals = ~61% right — the doc-027
  ACT / INVERT / SKIP policy realized with a probability, exactly as Moises framed it.

## The pre-registered check PASSED
Doc-078 predicted the age×alignment interaction would carry the inversion. Coefs:
`sig_with_leg -1.24, inter(age×align) +0.553, pivot_age -0.388` — precisely that
structure, found by the regression on 2024 and REPLICATING on 2025/26.

## Honest limits
- Target = agreement with oracle-label DIRECTION, not realized P&L. Strong proxy
  (labels are the ground-truth good trades); economic conversion still pending.
- ~2.4 signals/day total; OOS N=671. Pivot-age = v1 light streaming zigzag (canonical
  ATR(14)x4 spec) — will be superseded by the parity-ported zigzag detector.
- Single detector. The point of stage-0 is the COMBINER; this is one column of it.

## Next (Moises' direction, doc 075/this thread)
Apply the SAME 4-step pipeline to the other dossiers and to ZIGZAG itself — each contributes
feature columns + its own P(right); the combiner stacks them. Zigzag first (it is the
turn-clock everything else conditions on).
