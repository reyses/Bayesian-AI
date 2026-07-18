---
name: distilled-cusp_launch_detector
description: Cubic-regression + classifier cusp/launch detector — shipped random-split AUC 0.63 was same-day leakage; honest day-disjoint pooled AUC 0.453 (noise) on human-match target, 0.654 (real but not-yet-trustworthy) on MFE-paid target; only 4 labeled days, underpowered.
metadata: {type: distilled, topic: cusp_launch_detector, status: live}
---
## Verdict
Asked: does the `tools/viz/` cubic-regression + classifier pipeline (`cubic_utils.find_raw_turns`
→ `extract_pick_primitives` → `train_picks_classifier`) actually detect cusp/launch turns, vs the
shipped random-split AUC 0.63? Found: that 0.63 was same-day leakage. Day-disjoint (leave-one-day-out)
eval shows the human-match target is NOISE and the MFE-paid target looks real but has a coverage bug
and a trivial base rate, so it isn't trustworthy yet. Status: underpowered (4 labeled days) — "unproven,"
not "disproven."

## Key numbers (with CIs where they exist)
- Shipped random 80/20 split: AUC 0.630 — same-day leakage, not trustworthy (README.md, eval_oos.md).
- Day-disjoint, target='target' (human-match): n=1580, pos=85 (5.4%), 4 days — pooled OOS AUC **0.453**,
  day-block 95% CI **[0.419, 0.697]** → NOISE (CI includes 0.5, not significant). Per-day AUC spread
  0.466-0.737 (mean 0.597, k=4). (eval_oos.md)
- Day-disjoint, target='paid' (objective forward MFE ≥ 5pt/60min): n=379, pos=323 (85.2%), 3 days —
  pooled OOS AUC **0.654**, day-block 95% CI **[0.571, 0.740]** → CI excludes 0.5 but flagged NOT
  trustworthy: 85% base rate (near-constant target) + 24% coverage bug (2024-01-01 holiday parquet
  missing 234 rows; MFE loader read only the single start-date parquet vs candidates spanning
  multi-day windows → biased subset). (eval_oos.md, README.md)
- Per-day detail: the two days with real sample size (44, 27 picks) sit at chance (AUC 0.47, 0.49);
  the "good" AUCs (0.74, 0.69) come from days with only 4 and 10 positives — small-sample noise.
  (README.md)
- 1610 candidate cubic turns total; 4 labeled days: 2024-01-01, 2025-01-06, 2025-06-06, 2025-09-08.
  (eval_oos.md)

## Graveyard / never-retry (if any)
- Random 80/20 split evaluation of this pipeline — proven same-day leakage, do not reuse as a metric.

## Reusable assets
- `research/cusp_launch_detector/tools/eval_oos.py` — day-group CV (LODO ≤6 days else k=5) + day-block
  bootstrap CI harness with dual targets (`target` human-match, `paid` objective MFE); built to stay
  honest as labeled-day count grows — rerun as-is once more days are labeled.

## Data locations
- `DATA/cusp_picks/` — 330 human ground-truth picks.
- `DATA/cusp_picks/features/candidate_primitives.csv` — 1610 candidate cubic turns w/ features
  (z_15s, z_1m, z_15m, slope_*, curv_15m, band_width, band_rank_60, sigma_15m_rank_60, fan_width,
  align_up_count, align_down_count).
- `DATA/ATLAS/1m` — price series read by the MFE loader (currently only single start-date parquet
  per candidate — the identified coverage bug).

## Open threads
- Fix coverage bug: load continuous multi-day price series per candidate window instead of single
  start-date parquet; skip/curate holidays (2024-01-01 gap).
- Harden the 'paid' target: raise MFE threshold (~15-20pt) or require MFE/MAE ≥ 2 so base rate isn't
  ~85%.
- Label more days (≥15-20 disjoint), then re-run `eval_oos.py` for a conclusive CI on both targets.

## Sources
- research/cusp_launch_detector/README.md
- research/cusp_launch_detector/reports/eval_oos.md
- research/cusp_launch_detector/tools/eval_oos.py

## Archive recommendation
KEEP-LIVE (reason: verdict is "unproven, not disproven" with a concrete, already-scoped fix list and
a scale-ready harness sitting idle — this is a paused-not-dead thread; archiving would bury the exact
fix-before-retry checklist needed to unblock it).
