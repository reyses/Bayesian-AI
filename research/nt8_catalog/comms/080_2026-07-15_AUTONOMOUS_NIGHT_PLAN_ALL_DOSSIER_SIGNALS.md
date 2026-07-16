# AUTONOMOUS NIGHT PLAN — standard signal pipeline across all dossiers + zigzag
**Doc:** 080 · **Date:** 2026-07-15 · **Author:** Claude (autonomous, Moises asleep) · **Status:** PLAN (execution follows immediately)

## Directive (Moises, before sleep)
Continue autonomously: apply the SAME approach (signals → AI-label overlap → transition
profile → feature rows → binary logistic P(right)) to ALL the dossiers and to ZIGZAG —
see if more signal can be extracted. Safety cron set (+5h) with resume instructions.

## Scope tonight — 12 signal streams, article-faithful, each citing its verified spec
ZIGZAG (pivot-confirmation signals; the causal turn-clock), ORB-02, SEASON-12, VWAP-03,
OHLC-01, PIVOT-16, ROUND-05, CROSS-11, VWMA-10, DOW-19, TUNNEL-20, ATR-09 (50% fill).
SKIPPED tonight (cannot port faithfully fast — skip rather than fabricate, the audit
lesson): MACD-07/RSI-06 (pivot-matched divergence), FIB-17 (daily zone logic — exists in
batch_b but needs daily context wiring), SQZ-04, SAR-23, HNS-22, VP-01/VA-13 (volume
profile), ZONE-21, SCALP-18, ORDERFLOW-14 (alt data), RENKO-24 (brick clock). These get
the treatment when their native detectors are wired into the shared harness.

## Method per stream (constant, = docs 077-079 template)
- Signals generated CONTINUOUSLY (tail-carry across files; no cold start, doc 073).
- Shared causal features at each signal: zigzag pivot_age_min, sig_with_leg, detector
  value, tod; interaction sig_with_leg×age.
- Target: direction agreement with the ACTIVE AI label (ground truth, 576 days).
- Train 2024 → test 2025+2026, day-block CIs, tercile calibration table, OOS AUC.
- League table at the end + pooled combiner preview (all signals, detector dummies).

## Rules in force
Article-faithful triggers only (cited); no nulls; FPS core FROZEN (this is a raw-stream
research harness, FPS untouched); one comms doc per milestone; commit+push each; honest
sizing vs the house signal bar; %>0=1.00-class impossibilities = auto-fail.
