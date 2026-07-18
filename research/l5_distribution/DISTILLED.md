---
name: distilled-l5_distribution
description: L5 intra-bar distribution layer built + causality-tested, but 2024 forward-return edge test found NO separation — stalled at Stage A, never promoted to FEATURE_NAMES.
metadata: {type: distilled, topic: l5_distribution, status: dead}
---
## Verdict
Built a new SFE feature layer (L5_{tf}_ldist_*, 12 within-bar distribution stats per
TF) to test whether intra-bar 1s distribution shape adds NMP entry-filter edge on top
of L1-L4. Implementation, causality, and unit tests all passed; the GATE edge-test
(forward snap-back separation at the |z|>1.8481 tail, 2024, day-block CI) found ZERO
of 12 L5_1m features separate from noise. Never advanced to Stage B (FEATURE_NAMES)
or live integration.

## Key numbers (with CIs where they exist)
- Baseline snap-back return at |z|>1.8481 (fwd 5min/K=60), 2024, per-day mean: **+0.074 pts**, 95% day-block CI **[-0.124, +0.277]** (259 days, 324,479 tail entries) — not even +EV itself.
- Per-feature Spearman(L5_1m feature, snap-back return), all 12 features, all CIs include 0 (e.g. `n` +0.0131 CI[-0.0008,+0.0270]; `skew` +0.0111 CI[-0.0016,+0.0234]; `level` -0.0077 CI[-0.0221,+0.0064]; full table in report). **No feature separates.**
- Feature count check: N_FEATURES=297 with L5 in Stage A (registered in LAYER_FAMILIES, NOT in FEATURE_NAMES) per `L5_IMPL_PROGRESS_2026-06-13.md`.

## Graveyard / never-retry (if any)
- L5_1m intra-bar distribution battery (min/q1/median/q3/max/mean/std/skew/kurtosis/n/level/outlier_pct) as an NMP entry filter on the forward snap-back proxy — well-powered null (259 days), consistent with "close≈ldist_level" wedge check and the broader NMP-entry-unsolved finding. 1h/4h L5 untested (flagged as natural next probe, never run).

## Reusable assets
- `core_v2/statistical_field_engine.py::compute_L5_ldist` — the L5 layer implementation (kept, materialized, Stage A only).
- `research/l5_distribution/tools/test_l5_ldist.py` — unit test vs independent numpy reference.
- `research/l5_distribution/tools/test_l5_causality.py` — verifies step-fill never uses a forming bar.
- `research/l5_distribution/tools/test_l5_edge.py` — the wedge-methodology edge-test harness (day-block Spearman CI), reusable for any new feature vs forward snap-back.
- `research/l5_distribution/tools/test_l5_smoke_2024.py` — real-data one-day smoke test of the build path.

## Data locations
- `DATA/ATLAS/FEATURES_5s_v2/L5_{tf}/{day}.parquet` — 8 TFs (5s,15s,1m,5m,15m,1h,4h,1D), materialized for 2024 (259 days) and presumably later builds; NOT in FEATURE_NAMES so not consumed by the assembled 185D grid.

## Open threads
- 1h/4h L5 std/skew vs macro vol untested (2024 report flagged this as the natural next probe, never executed).
- BUG-LIVE-LOOKAHEAD in `core_v2/live_features.py get_v2_vector` was found+fixed+parity-verified during this work (2026-06-14, user-approved) — orthogonal to L5 itself but recorded in the same progress doc.

## Sources
- research/l5_distribution/reports/L5_IMPL_PROGRESS_2026-06-13.md
- research/l5_distribution/reports/L5_edge_2024_preliminary.md
- research/l5_distribution/tools/test_l5_edge.py
- research/l5_distribution/tools/test_l5_causality.py
- research/l5_distribution/tools/test_l5_ldist.py
- research/l5_distribution/tools/test_ldist_wedge.py

## Archive recommendation
ARCHIVE (gate failed — well-powered null on the 2024 forward-return proxy; layer materialized at Stage A but never earned promotion; no live/FEATURE_NAMES dependency to preserve).
