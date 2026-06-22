# Stage 1/2 Overhaul: Build the F-Space MAP

Build the diagnostic F-space **map** — not a predictor. Record the full R-curve
per segment, direction/flip descriptors, and aggregate a real-vs-null composite
with four curves: explainability, survival/hazard, direction, and flip-signature.

## Execution order (as specified in §7 of spec)
`§2 Stage1 dump` → `§3 Stage2 + composite` → `§4 null ensemble` → `§5 predictors (deferred)`

Smallest first runnable slice (spec §7 note):
> "§2 r_curve dump + §3 composite on the ALREADY-BUILT B2C/B2T 2024_02_20 + the 3 existing series
> (real/brown/four) — gives the first distortion map with no new heavy builds."

---

## Proposed Changes

### Component 1 — Stage 1 (`stage1_speed_pass.py`)

#### [MODIFY] [stage1_speed_pass.py](file:///C:/Users/reyse/OneDrive/Desktop/Bayesian-AI/research/fspace_experiment/stage1_speed_pass.py)

**New constants** at top of file:
```python
FLIP_TAIL_BARS     = 20   # bars to trace R-curve PAST the break (distortion-through-flip)
FLIP_LOOKBACK_BARS = 15   # residual tail kept before break (pre-flip signature)
```

**R-curve recording:** Currently `batched_ols_scan_pytorch` produces `tiers_exp/max_res_exp/betas_exp`
for every candidate L but they are immediately consumed and discarded. We will:
1. Compute `r2` at each L: `r2 = 1 − SS_res/SS_tot` on `Y = close − close[start]`.
2. Count `n_off` and `consec_off` at each L using the existing break rule.
3. Accumulate a `r_curve` list of `{L, r2, max_resid, n_off, consec_off}`.
4. Continue the scan for `FLIP_TAIL_BARS` beyond `L_break` to capture post-flip distortion.

**Direction descriptors** added to every PRISTINE segment record:
```python
slope_pts_per_s  : OLS slope (pts/second) — simple linregress on close[start:end]
net_move_pts     : close[end] − close[start]
direction        : sign(net_move_pts)  ∈ {-1, 0, +1}
```

**Flip descriptors** added to every PRISTINE segment record:
```python
break_reason          : 'consec_off' | 'end_of_data' | 'chaos_giveup'
consec_off_at_break   : the consec run that triggered the break
resid_tail            : last FLIP_LOOKBACK_BARS residuals before L_break
tier_at_break         : tier of the final accepted bar before break
```

**`raw_start_idx` / `raw_end_idx`** already present — unchanged (causal bridge pointer).

---

### Component 2 — Stage 2 (`stage2_parallel_chaos.py`)

#### [MODIFY] [stage2_parallel_chaos.py](file:///C:/Users/reyse/OneDrive/Desktop/Bayesian-AI/research/fspace_experiment/stage2_parallel_chaos.py)

Currently chaos-recovered segments emit a minimal schema. After the overhaul they
will emit the **same enriched schema** as Stage 1 PRISTINE segments (r_curve,
direction, flip descriptors). No structural change to the chaos re-segmentation
algorithm itself.

---

### Component 3 — Map Builder (new file)

#### [NEW] [build_map.py](file:///C:/Users/reyse/OneDrive/Desktop/Bayesian-AI/research/fspace_experiment/build_map.py)

Driver that:
1. Accepts `--rep` (B2C or B2T), `--day`, `--k_surrogates` (default 30).
2. Loads the Stage 1 + Stage 2 enriched JSONs for the three existing series: REAL, BROWN, FOUR.
3. Builds the **four composite curves** per series:
   - **Explainability curve:** median + q25/q75 of `r2` at each forward offset k across all regimes of length ≥ k.
   - **Survival / hazard curve:** fraction alive at offset k; hazard(k) = P(flip at k | alive at k).
   - **Direction map:** histogram of `direction`; serial dependence P(next=same); slope vs duration scatter.
   - **Flip signature:** average `resid_tail` trajectory per break_reason.
4. Computes **rank-based p-values** per summary statistic: `p = (1 + #{null ≥ real}) / (1 + K)`.
5. Writes `artifacts/map_{rep}_{day}.json` (all four curves + per-regime table).
6. Writes `reports/findings/map_{rep}_{day}.md` — mode-first, null bands + p-values.

---

## Verification Plan

### Automated
- Run `stage1_speed_pass.py` on `2024_02_20` for `B2Cv2_REAL` — confirm output JSON contains
  `r_curve`, `direction`, `net_move_pts`, `resid_tail`, `break_reason` keys on every PRISTINE segment.
- Run `build_map.py --rep B2Cv2 --day 2024_02_20 --k_surrogates 3` (fast sanity check with K=3).
- Confirm output MD exists with explainability and survival curve tables and p-values.

### Manual Review
- Open one segment's `r_curve` — verify L rises monotonically from SEED_BARS to L_break + FLIP_TAIL_BARS.
- Check `resid_tail` has exactly `min(FLIP_LOOKBACK_BARS, L_star)` entries.
- Check `direction` ∈ {-1, 0, +1} for all records.
- Verify null maps look noisier than real map on the explainability curve (sanity check).

---

## Open Questions

> [!IMPORTANT]
> The spec says to run on the **already-built** `B2Cv2_REAL/BROWN/FOUR` Stage 1 JSONs
> (`stage1_B2Cv2_REAL_segments_2024_02_20.json` etc.) as the first runnable slice.
> However these files were produced by the **old** Stage 1 (without r_curve / direction / flip fields).
> **We must re-run Stage 1 on those three series to produce enriched output.**
> Confirm: should we re-run Stage 1 on B2Cv2 + B2Tv2 for day 2024_02_20, or start with a different day?

> [!NOTE]
> The `FLIP_TAIL_BARS = 20` post-break trace requires that the batched OLS scan continues
> past `L_break`. The current expansion loop `break`s immediately at first tier failure.
> We will change it to continue for up to `FLIP_TAIL_BARS` more bars (recording r2 only,
> not updating `L_star`) before the final break.
