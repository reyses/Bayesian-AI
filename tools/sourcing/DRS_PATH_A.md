# DRS Path A — canonical IS hardened forward pass

Run this AFTER Phase 1B (feasibility) shows the cross-day features have
signal. Produces the real `target_day_pnl` (not the peeky proxy) for DRS
training, by running the deliverable's hardened forward pass on IS days.

**Read-only on the deliverable.** All outputs go to
`DATA/CROSS_DAY/predictions_IS/`. The trained B-model pkls are loaded but
never modified. The deliverable's caches stay untouched.

## Expected runtime

| Step | Approximate wall time | Why |
|------|----------------------|-----|
| 1 — B-model inference (B1, B2, B4, B5, B6, B8) on IS | 5–10 min | 282k bars × 6 models |
| 2 — Build IS pivot probability cloud | ~30 s | join + density bins |
| 3 — Hardened forward pass over 277 IS days | 10–20 min | per-leg R-trigger walk on 5s closes |
| 4 — Aggregate to per-day and join into features | ~5 s | groupby + parquet write |
| **Total** | **~15–30 min** | |

## Run order

From the project root (`C:/Users/reyse/OneDrive/Desktop/Bayesian-AI/`):

```bash
# Step 1 — apply B1/B2/B4/B5/B6/B8 to IS (B7 already produced by Phase 1B)
python tools/sourcing/drs_a_step1_predict_b_models_is.py

# Step 2 — build IS pivot probability cloud
python tools/sourcing/drs_a_step2_build_cloud_is.py

# Step 3 — hardened forward pass on IS days
python tools/sourcing/drs_a_step3_forward_pass_is.py

# Step 4 — aggregate per-leg to per-day, join target into cross_day_features
python tools/sourcing/drs_a_step4_aggregate_day_pnl.py
```

After step 4, the file `DATA/CROSS_DAY/cross_day_features_with_target.parquet`
has the populated `target_day_pnl` column (gbm_ev hardened, per-day).

## What gets written (and where)

```
DATA/CROSS_DAY/predictions_IS/
  b1_proba_IS.parquet                        # step 1
  b2_proba_IS.parquet                        # step 1 (per-pivot)
  b4_proba_IS.parquet                        # step 1
  b5_proba_IS.parquet                        # step 1
  b6_proba_IS.parquet                        # step 1
  b8_proba_IS.parquet                        # step 1
  b7_leg_sizer_IS_with_preds.parquet         # already produced by Phase 1B
  pivot_probability_cloud_IS.parquet         # step 2
  pivot_probability_cloud_IS.txt             # step 2
  composite_forward_pass_hardened_IS.csv     # step 3 (per-leg)
  composite_forward_pass_hardened_IS.txt     # step 3 summary

DATA/CROSS_DAY/
  cross_day_features.parquet                 # already exists (peeky proxy target)
  cross_day_features_with_target.parquet     # step 4 (with real hardened target)
```

## What changes between Phase 1B and Phase A

| Aspect | Phase 1B (peeky proxy) | Phase A (canonical) |
|--------|------------------------|---------------------|
| Entry detection | Oracle pivot bar | R-trigger fire on 5s close |
| Exit price | Pivot ± R formula | Next R-trigger fire close |
| Friction | $6/leg | $6/leg (same) |
| IS day count | 277 | 277 (same) |
| Absolute $/day | ~2× hardened | Real hardened number |
| Day ranking | Similar to hardened | Identical to hardened |
| Compute | seconds | 15–30 min |

The peeky proxy overstates absolute $/day by ~2× per the deliverable's
SESSION_LOG Phase 8. The DAY-RANKING (which days are good vs bad) is
similar, so Phase 1B is a valid feasibility check. Phase A is needed
for the actual training target.

## If a step fails

- **Step 1 fail** (B-model inference): usually pickle / pandas / sklearn version
  mismatch. Re-pin to the deliverable's `requirements.txt`.
- **Step 2 fail** (cloud builder): often due to missing `b5_proba_IS` schema columns
  if step 1 didn't complete. Re-run step 1.
- **Step 3 fail** (forward pass): missing 5s/1m bar files. Check that
  `DATA/ATLAS/5s/<day>.parquet` and `DATA/ATLAS/1m/<day>.parquet` exist
  for every day in the truth dataset.
- **Step 4 fail** (aggregate): pred_amp_R_hardened or pnl_usd missing in CSV. Step 3
  output is malformed.

## What this produces for DRS training

`cross_day_features_with_target.parquet` is a single dataframe with:
- 15 cross-day feature columns (session-start, lookahead-honest)
- `target_day_pnl` — real per-day P&L under hardened gbm_ev pipeline
- `source` — ATLAS (IS) vs NT8 (OOS)
- `date_label` — for chronological walk-forward CV

Then the DRS trainer can run:
- Walk-forward CV with date-disjoint folds (same logic as `drs_feasibility_gbm.py`)
- Train on this real target instead of peeky proxy
- Bootstrap CI on Pearson R, MAE vs persistence baseline
- If Pearson R > 0.20 with CI > 0, the DRS is real-money deployable as
  a day-level size multiplier
