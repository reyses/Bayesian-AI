# research/cusp_launch_detector

Honest evaluation of the cubic-regression + classifier cusp/launch detector (the `tools/viz/`
pipeline: `cubic_utils.find_raw_turns` → `extract_pick_primitives` → `train_picks_classifier`).
Cusps = the convex/concave turns / trend-launch points from the `recovery_dynamics` arc; 330 human
picks (`DATA/cusp_picks/`) are ground truth.

## tools/
- `eval_oos.py` — day-disjoint (leave-one-day-out) evaluation vs the shipped random 80/20 split,
  held to the signal bar. Reads `DATA/cusp_picks/features/candidate_primitives.csv`.

## Findings (`reports/eval_oos.md`)
- **The shipped random-split AUC 0.63 was same-day LEAKAGE.** Day-disjoint (test on unseen days):
  **pooled OOS AUC 0.453 — below chance → NOISE** by the signal bar.
- Per-day: the two days with real sample (44, 27 picks) are at chance (0.47, 0.49); the only "good"
  AUCs (0.74, 0.69) come from days with 4 and 10 positives = small-sample noise.
- **No cross-day edge is demonstrated.** But only **4 labeled days** in the CSV → underpowered;
  "unproven," not "disproven."

## Scale-ready harness (current)
`eval_oos.py` now: day-group CV (LODO ≤6 days, else k=5), **day-block bootstrap CI**, and **two
targets** — `target` (matches a human pick) and `paid` (objective forward MFE from price). Built so
that when the labeler grows the day count, the SAME run yields a conclusive CI.

### Honest scorecard (4 days — still underpowered; NEITHER is a green light)
- **human-match**: OOS AUC 0.453, CI [0.42, 0.70] → **NOISE** (full coverage; genuine non-result).
- **MFE-paid**: OOS AUC 0.654, CI [0.57, 0.74] → looks "REAL" but is **NOT trustworthy**:
  - 85% base rate (5-pt MFE in 60 min is trivially hit — an easy, near-constant target);
  - **24% coverage bug**: 2024-01-01 is a holiday (parquet missing, 234 lost) and candidates span
    multi-day windows while the MFE loader read only the single start-date parquet → biased subset.

### Fixes before any MFE number is trustworthy
1. **Coverage**: load a continuous multi-day price series spanning each candidate's window (not the
   single start-date parquet); skip/curate holidays.
2. **Harder target**: MFE ≥ ~15-20 pt, or MFE/MAE ≥ 2 (a real launch), so base rate isn't ~85%.
3. **Label more days** — ≥15-20 disjoint. Then re-run (day-disjoint stays the rule).
