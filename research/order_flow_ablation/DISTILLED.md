---
name: distilled-order_flow_ablation
description: True Delta (Databento tick order flow) tested as an entry/exhaustion feature across 4 escalating gauntlets — DEAD every time; tick-data purchase rejected twice over.
metadata: {type: distilled, topic: order_flow_ablation, status: dead}
---
## Verdict
Tested whether Databento True Delta (aggressor buy-sell volume) beats the free
OHLCV wick facsimile as a predictive/RL-reward feature, to decide whether to
buy 2024 tick-level MBO data. No README/project.md exists in this folder —
distilled from `reports/*.md` only. Result across 4 escalating tests: DEAD.
Note the internal reversal: on 2026-06-27 night the ablation briefly *passed*
the Fourier null and tick-data purchase was called "greenlit" (per
`docs/daily/2026-06-27.md` 23:52 entry) — 6 minutes later the SAME session
re-graded it against the project's magnitude gate (+0.05 lift), found the
lift an order of magnitude short, and reversed to DEAD/REJECTED
(`03_FINAL_EXPERIMENT_EXIT.md`). The next morning (2026-06-28) a stricter
5-fold Fourier gauntlet made `inflection_verdict.md` fail outright (BREAK,
lift **-0.0043** vs null) — final answer, reconfirmed twice.

## Key numbers (with CIs where they exist)
- `delta_verdict.md`: 30m fwd-return R² Baseline **-0.1860** vs Baseline+Delta **-0.3396** (**-0.1537**, degraded). Purchase gate FAILED (regression framing).
- `01_absorption_discovery_report.md`: True Delta vs OHLCV facsimile corr **+0.5660**; wick corr: upper wick **+0.063**, lower wick **-0.064**; MTF: body @5s **+0.47** → wick @1h **-0.41**.
- `stage_0B_signal_test.md`: corr(true_delta, price_delta) **-0.0018**; corr(facsimile, price_delta) **0.0645**; disagreement rate **49.67%** of bars.
- `absorption_analysis.md`: fwd 5m return (×10000): Absorption DN **+37746.6**, Absorption UP **-38317.3**, Confirm DN **+24030.0**, Confirm UP **-24729.1** (no CIs — exploratory only, per report's own §10).
- `02_predictive_decay_verdict.md`: t+1 AUC Baseline 0.5523, Full 0.5542, Fourier null 0.5530 → lift **+0.0012** (negligible). Verdict: DEAD.
- `stage_1_inflection_verdict.md` (early pass): Baseline 0.5973, Delta 0.6001, Fourier null 95th **0.6156** → Verdict **BREAK**.
- `03_FINAL_EXPERIMENT_EXIT.md` (1.5M rows, roll-weeks excluded): L1 416D **0.6265**, L2 +wicks **0.6247** (-0.0018), L3 +True Delta **0.6326** (**+0.0079**); passed Fourier null (0.6273) but **+0.0079 << +0.05** conditional-approval threshold. Verdict: DEAD.
- `inflection_verdict.md` (final, rewritten 2026-06-28): L1 0.6156, L2 0.6179, L3 0.6214 (lift over L2 +0.0036); Fold-5 L3 AUC 0.6363 vs Fourier null 95th (N=20) **0.6406** → true lift **-0.0043**. Verdict: BREAK. Purchase Decision: NO.

## Graveyard / never-retry
- **True Delta as directional feature**: DEAD at every framing (R² regression -0.1537; AUC lift capped +0.0079, then -0.0043 on stricter gauntlet). Do not retry without a fundamentally different feature construction.
- **Databento tick/MBO data purchase**: rejected twice (2026-06-27 night, reconfirmed 2026-06-28 morning). Cost/latency/complexity not justified by the measured edge.

## Reusable assets
- `pipeline/ablation_study.py`, `ablation_study_v2.py` — R²/AUC ablation harness (L1/L2/L3 layering).
- `pipeline/stage_1_inflection_ablation.py`, `stage_1_predictive_decay.py` — Fourier phase-randomization null gauntlet (reusable pattern for any new-feature magnitude gate).
- `pipeline/stage_0C_absorption_analysis.py`, `stage_0D_wick_correlation.py` — quadrant/wick-correlation EDA + plots.
- `pipeline/cumulative_delta_builder.py`, `extract_baseline_features.py` — feature builders (last touched 2026-07-16, may still be live-referenced elsewhere — check before archiving).

## Data locations
- `DATA/ATLAS/baseline_features_416D.parquet` (5s V2 grid baseline).
- `DATA/ATLAS/order_flow_delta_5s.parquet` (Databento true delta/volume; corruption trace for this file documented separately in `research/nt8_catalog/comms/025_...PHASE5_READY.md`).

## Open threads
- `research/reward_design/THESIS_reward_design.md` was edited alongside this conclusion (Exhaustion_Penalty walked back) — verify reward_design's DISTILLED reflects the same DEAD verdict, no dangling RL reward hook to a rejected feature.

## Sources
- research/order_flow_ablation/reports/delta_verdict.md
- research/order_flow_ablation/reports/delta_verdict_v2.md
- research/order_flow_ablation/reports/01_absorption_discovery_report.md
- research/order_flow_ablation/reports/02_predictive_decay_verdict.md
- research/order_flow_ablation/reports/stage_1_inflection_verdict.md
- research/order_flow_ablation/reports/03_FINAL_EXPERIMENT_EXIT.md
- research/order_flow_ablation/reports/inflection_verdict.md
- docs/daily/2026-06-27.md, docs/daily/2026-06-28.md (session-exit narrative, incl. the pass/fail reversal)

## Archive recommendation
ARCHIVE (concluded/DEAD, reconfirmed twice over 2 days; no README/project.md ever existed for this folder — pure reports dump, safe to move as-is).
