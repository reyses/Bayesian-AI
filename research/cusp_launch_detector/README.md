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

## Next (to make it conclusive)
1. **Label more days** — ≥15-20 disjoint days before any AUC here is trustworthy.
2. **Retarget to "the launch paid"** (MFE/MAE already recorded) instead of "matches a human pick."
3. Then re-run `eval_oos.py` (day-disjoint stays the rule).
