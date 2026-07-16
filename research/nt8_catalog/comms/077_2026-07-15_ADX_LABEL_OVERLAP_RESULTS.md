# ADX -> AI-label overlap — first stable directional signal (weak but real)
**Doc:** 077 · **Date:** 2026-07-15 · **Author:** Claude (executor) · **Status:** RESULTS
Tool: `tools/adx_label_overlap.py`. Setting: doc-074 candidate (SMA, N_adx=84/7min,
thr=25, cross-MA 240), CONTINUOUS windows, RTH triggers. Ground truth = 576 days of
AI labels (25,680 labels). No P&L — detection quality only (north star, doc 076).

## Results
- Label coverage of RTH ≈ 1.00 (labeler chains trades) -> "inside a label" is trivial
  (0.99); **DIRECTION AGREEMENT is the real metric** (baseline exactly 0.50).
| year | N signals | direction agreement | day-block CI |
|---|---|---|---|
| 2024 | 688 | **0.57** | [0.56, 0.59] |
| 2025 | 526 | **0.58** | [0.56, 0.60] |
| 2026 | 145 | **0.61** | [0.57, 0.65] |
- **Phase-in-label**: most signals fire in the FIRST HALF of the label (2024: 514/688
  before phase 0.5) — the move is still ahead; the signal is actionable, not late.
- **Label-side detection: 3.0%** (459/15,362 labels get an agreeing ADX signal).
  ADX sees a small SUBSET of the oracle's ~47/day — it is a FEATURE, not a labeler.
- Rate ~2.4 signals/day, consistent with the doc-074 sweep (2.0/day).

## Honest sizing (house signal bar, MEMORY §2)
Agreement-over-coinflip = +0.07/+0.08/+0.11 -> the CONDITIONAL band (0.05-0.10), touching
"real" in 2026. Replicated in all 3 years with tight CIs and NO tuning on 2025/2026
(setting chosen on 2024Q1 frequency only) -> genuinely out-of-sample stability. Weak but
real, and it fires early. This is exactly a STAGE-0 WEAK SIGNAL for the calibrated
combiner (project north star) — not a standalone strategy.

## Next options
(a) Same overlap harness on the other detectors (zigzag first, doc 075) -> build the
    weak-signal feature matrix for the combiner.
(b) Doc-075 overfit-decay on ADX with THIS metric (label-direction agreement): overfit
    windows on 2-month IS, measure time-to-decay below Moises' 70% line.
FPS untouched.
