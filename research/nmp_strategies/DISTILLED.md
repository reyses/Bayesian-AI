---
name: distilled-nmp_strategies
description: 17 NMP curve/kinematic/inflection entry-exit variants tested — every one net-negative; KT1 oracle-ceiling test found no regime separation.
metadata: {type: distilled, topic: nmp_strategies, status: dead}
---
## Verdict
Broad sweep of NMP-derived curve-fit entry/exit variants (pure curve-velocity
reversal at 1s/5m/60m, kinematic velocity/acceleration, quadratic regression,
EMA dual-timeline, HDBSCAN-clustering exits, rmean/segment-R inflection,
delta-peak, consecutive-inflection, trend-pullback, triple-timeline v2,
z-round-trip, z+5m-slope gating) plus a KT1 "oracle ceiling" regime test.
Every strategy variant tested here is net-negative after costs. z-round-trip
is the least-bad (beats the R-trigger baseline, CI still straddles 0). KT1
found no PnL separation across hindsight regime cells (GAP/PRISTINE/CHAOS),
though the KT1 test itself is flagged invalid-as-run (join bug, mislabeled
"oracle", OOS-vacuous) — corrected re-run still showed no separation but a
TRUE oracle-ceiling test is documented as still pending.

## Key numbers
- KT1 oracle ceiling (corrected, IS-only, 5s): PRISTINE 49.0%, CHAOS 23.0%, GAP 10.6% of trades; mean PnL flat -$4.6 to -$5.6 across all statuses — no separation. (`reports/NMP_KT1_Oracle_Ceiling_Test_2026-06-13.md`)
- Trend-pullback 2024 (honest $2/pt costs): best gross Z_ENTRY=2.5 gross +$1.89/tr vs $2.00/tr cost — can't cover cost. Net $/day -0 to -22 across thresholds. (`reports/nmp_trend_pullback_2024.md`)
- Triple-timeline v2: IS net $/day -1190 [-1239,-1144], OOS -871 [-939,-807], ALL -1008 [-1052,-964], 0/259 winning IS days. (`reports/nmp_triple_timeline_v2.md`)
- z-round-trip vs R-trigger (Feb 2024): z-round-trip $/day -43 [-144,+54] (10/21 winning days) vs R-trigger -421 [-469,-378] (0/21) — beats baseline, still not significant. (`reports/nmp_z_roundtrip_2024_02.md`)
- z 5m-slope-gate structural split: DROP sloping-counter-trend cell +2 [-73,+76]; KEEP flat-or-with-trend -41 [-104,+20] (pre-cost, IS Feb-only). (`reports/nmp_z_5m_slope_gate_2024_02.md`)
- Geometric inflection (2024, full-cost): net $/day -1093 [-1506,-699], 94/259 winning days. (`reports/nmp_geometric_inflection_2024.md`)
- Pure curve-velocity reversal, no filters: all timeframes net-negative — 1s/10m -$65.09/tr, 1s/1h -$102.85/tr, 5m/60m -$72.26/tr, 1m quadratic -$21.82/tr. (`reports/nmp_pure_*_rcurve*.md`, `nmp_quadratic_1week.md`)
- Kinematic/EMA/HDBSCAN/rmean/segment-R/delta-peak/consecutive-inflection (1-week tests): net $/trade -$14 to -$123, PF 0.37-0.95, all net-negative. (`reports/nmp_kinematic_1week.md`, `nmp_ema_dual_1week.md`, `nmp_hdbscan_1week.md`, `nmp_rmean_inflection_1week.md`, `nmp_segment_r_inflection_1week.md`, `nmp_delta_peak_1week.md`, `nmp_consecutive_inflection_1week.md`)

## Graveyard / never-retry
- Pure curve-velocity reversal (any TF, no filter): structurally negative, whipsaw-dominated.
- Kinematic velocity/acceleration exits (EMA, HDBSCAN, rmean, segment-R2, delta-peak): all net-negative, PF 0.37-0.79.
- Trend-with-pullback: gross can't clear $2/tr cost at any Z_ENTRY tested.
- Triple-timeline v2 (3-sign-agree entry): sig. negative both IS and OOS, CI excludes 0.
- KT1 regime-conditioned fade: dead flat, no separation PRISTINE vs CHAOS (mathematically falsified per corrected re-run, direction only, not the full oracle-ceiling claim).

## Reusable assets
- `tools/analyze_nmp_lambda_stratified.py` — lambda-stratified analysis.
- `tools/nmp_z_roundtrip.py`, `tools/nmp_z_5m_slope_gate.py` — least-bad variants; worth a base for future OOS-gated retest.
- `tools/nmp_triple_timeline_v2.py`, `nmp_triple_timeline_forward.py`, `nmp_geometric_inflection.py`, `nmp_trend_pullback.py`, `nmp_1s_cross_entries.py` — other tested variants, all dead.

## Data locations
- `DATA/ATLAS` (L0 1s), `artifacts/stage2_year_segments.json` (112,289 empirical segments) used by KT1.
- Raw trade CSVs per variant in `reports/` (e.g. `nmp_fade_raw_is_atr4.csv`, `nmp_fade_raw_oos_atr4.csv`).

## Open threads
A TRUE oracle-ceiling test (best-selectable subset over fine regime cells /
daisy-chain best-trade oracle, IS-only, day-block CI) was never completed —
KT1 as-run only computed group means, not a ceiling. Flagged pending in the
KT1 report correction but not picked up since.

## Sources
research/nmp_strategies/reports/NMP_KT1_Oracle_Ceiling_Test_2026-06-13.md,
nmp_trend_pullback_2024.md, nmp_triple_timeline_v2.md, nmp_z_roundtrip_2024_02.md,
nmp_z_5m_slope_gate_2024_02.md, nmp_geometric_inflection_2024.md,
nmp_kinematic_1week.md, nmp_quadratic_1week.md (README.md and project.md are empty stubs)

## Archive recommendation
ARCHIVE (every tested variant is net-negative; program pivoted to
λ-completion / RL entry work per ROADMAP_LAMBDA_COMPLETION.md; only the
unfinished true-oracle-ceiling thread is a loose end, not worth keeping live).
