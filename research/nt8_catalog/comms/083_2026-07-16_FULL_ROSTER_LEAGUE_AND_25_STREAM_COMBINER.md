# FULL-ROSTER RESULTS — 13 new streams ported, 25-stream league + combiner
**Doc:** 083 · **Date:** 2026-07-16 · **Author:** Claude (executor; Moises directive
"do 4 and 5, easiest to hardest") · **Status:** FINAL

## 1. What was built (all specs cited line-level in the generator docstrings)
Easiest→hardest, per directive: RENKO-24 (Batch A detector verbatim), SAR-23
(PSAR 0.02/0.2), SQZ-04 (BB20/2.0 vs Keltner 1.5×proxy-ATR), RSI-06 + MACD-07
(30m-extreme divergences), SCALP-18 (VWAP+EMA240+RSI pullback), FIB-17 (daily
ADX(7)>25 + 10-day-swing fib zone), ZONE-21 (virgin supply/demand), VP-01 + VA-13
(prior-day volume profile POC/VAH/VAL), HNS-22 (top pattern + volume divergence),
**CURVE** (doc-075 roster: causal ENDPOINT variant of the labeler's centered cubic
N=20 on 1m — trailing fit, edge slope-flip classified by curvature).
**ORDERFLOW-14 = honest skip**: delta exists only 2025-07-30→2026-01-29 (no 2024
train year; 2024 GLBX trades not on disk). Documented deviations: one-shot latches
removed with the legacy 60-bar cooldown on SQZ/RSI/MACD/SCALP (frequency knob;
conditions verbatim).

## 2. League (complete 25 streams) — `reports/dossier_signal_league.md`
New graded entries (train 2024 / test 2025+26, day-block CIs):
```
VP-01     N=   283 AUC 0.732 base 0.36 || 0.12 / 0.43 / 0.52   (low tercile inverted = 88%)
SAR-23    N= 37184 AUC 0.618 base 0.44 || 0.33 / 0.44 / 0.56
RENKO-24  N=198560 AUC 0.611 base 0.55 || 0.44 / 0.55 / 0.65
CURVE     N= 26368 AUC 0.606 base 0.55 || 0.45 / 0.54 / 0.66
ZONE-21   N=  3451 AUC 0.584 base 0.63 || 0.58 / 0.61 / 0.72
MACD-07   N=  9781 AUC 0.552 base 0.05 || inverter
RSI-06    N= 14967 AUC 0.515 base 0.04 || inverter
```
Low-frequency (raw agree): SCALP-18 0.02 (N=53, extreme inverter), FIB-17 0.33,
VA-13 0.33, SQZ-04 0.58, HNS-22 0.57.
**Structural finding hardened:** every fade/divergence/pullback article premise is an
INVERTER (PIVOT 0.05, ATR 0.01, RSI 0.04, MACD 0.05, SCALP 0.02, FIB/VA 0.33) — they
fire mid-move while the label rides the move. The articles' bounce ideas are, on MNQ
5s, continuation confirmations when flipped.

## 3. 25-stream pooled combiner — `reports/combiner_preview.md`
N=447,433 fires (187,636 train / 259,797 test), features = shared causal set +
consensus + per-stream one-hots.
- **OOS AUC 0.675** on test base 0.510 (12-stream was 0.687 on base 0.533 — harder,
  more balanced pool; tails matter more than AUC here).
- **Calibration near-perfect OOS**: pred 0.16→obs 0.17, 0.38→0.38, ..., 0.75→0.74;
  all decile CIs ±0.01 on ~26k fires each.
- **Tails strengthened**: bottom decile obs 0.17 [0.16,0.18] → INVERT = **83% right**;
  top decile 0.74 [0.73,0.75] → ACT = 74% right. ≈52k OOS fires/2yrs in the two
  actionable tails (was ~17k at 12 streams).
- Model auto-inverts the inverters: is_RSI06 −0.526, is_MACD07 −0.422 (most negative
  coefs); sig_with_leg +0.458 and is_ZIGZAG +0.321 still carry the momentum side.
- **Consensus INVERTED in the full pool**: raw agreement 0.56-0.57 at 0-2 co-fires →
  0.48 at 6+. With dense inverter streams pooled, many simultaneous fires mark
  chop/extreme moments — crowding is now a warning, not conviction (coef −0.073).

## 4. Perf note (Moises asked "can't we use CUDA?")
The bottleneck was algorithmic, not hardware: the day-block bootstrap re-scanned the
array per sampled day (~10⁹ ops/decile). Replaced with precomputed per-day sums +
vectorized gather (`day_block_ci` in dossier_signal_pipeline.py; consensus loop
vectorized with prefix sums). Combiner runtime: **55+ min (unfinished) → ~3 min**,
identical statistic. CUDA unnecessary at this scale; the same trick applies to the
coming overfit-decay sweeps.

## 5. Ops incidents
- Windows `python` (hermes-agent venv — AG's) began hanging at interpreter start;
  two combiner runs lost. Routed to the WSL venv (`/home/reyses/venvs/bayesian-ai`).
  The hermes mailbox_watcher (polling the legacy pre-protocol root mailbox since
  07-13) was killed with Moises' approval — the numbered-doc protocol does not use it.

## 6. Next (per the new delegation ladder, specs to follow)
1. **Opus worker**: economic conversion — P-decile fires → forward drift at label
   horizons, no stops; turns P(right) into $/fire honesty.
2. **Sonnet worker**: overfit-decay shelf-life sweep (doc 075) on the combiner.
3. Zigzag phase-conditioned split; Mamba state-vector handoff spec after economics.
