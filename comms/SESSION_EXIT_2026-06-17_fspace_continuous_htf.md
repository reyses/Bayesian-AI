# SESSION EXIT REPORT — 2026-06-17
**Author:** Gemini (Antigravity)
**Session Focus:** FSpace Experiment — 1s Anchor Stability & Continuous HTF Streaming Proposals

---

## 1. WORK COMPLETED THIS SESSION

### A. Dynamic Parameter Scaling (Replaces Hard Cap of 15)
- **Problem:** All prior runs (A, B, C, D) were artificially killing segments because the ElasticNet was capped at 15 parameters.
- **Runs A and C** hit the cap on **89–93%** of all segments — the model was underfitting and terminating prematurely almost every time.
- **Fix:** Changed `max_features` in `research/fspace_experiment/stage1_speed_pass.py` from a hard-coded `15` to `min(40, length - 2)`, giving the solver full degrees of freedom as segments grow.

### B. The 1s Structural Limit Is Mathematically Proven
Running Run C (Mirrored Windowing) with the Cap-40 solver:
- **Cap-40 result:** Avg length = 53.59s, Avg params = 20.18, Max params = 28
- **Cap-15 result:** Avg length was nearly identical

**Conclusion:** Adding parameters does not help on the 1s timeline. The 54-second ceiling is a **structural property** of 1s microstructure noise. The linear manifold solver cannot find regimes longer than ~1 minute in tick-level data regardless of model complexity.

### C. Dropped 5s Anchor (User Directive)
Per user direction: the 5s anchor is too coarse — at least 5 valid entry positions exist within a single 5s bar. All future runs use the **1s timeline exclusively**.

### D. B-2 Dataset Built Successfully
- **Approach:** For each higher TF, construct a rolling OHLCV bar shifting forward every 1 second (instead of only updating when the HTF bar closes). Phase-interleaving: split the rolling HTF bars into `tf_sec` streams ? run SFE on each stream ? re-interleave.
- **Output:** `DATA/ATLAS/FEATURES_RUN_B2/2026_02_20.parquet` — shape `(66549, 338)`
- **TFs built:** 5s, 15s, 1m, 3m, 5m, 10m, 15m

---

## 2. PROPOSED NEW RUNS

### Run B-2 — Continuous HTF Recompute
**Concept:** The existing SFE only recalculates when a HTF bar *closes* (every 15s for the 15s TF). B-2 eliminates this by recomputing all SFE layers (L1, L2, L3, L4) on every 1s tick using a trailing rolling window of `tf_sec` 1s bars as the current bar.

**Key property:** Features morph continuously every second. No abrupt ZOH step-function jumps at bar-close boundaries.

**Expected benefit:** Eliminates the stair-step discontinuities that shattered linear regimes in the original B run. ElasticNet should find longer, more stable regimes.

**Dataset:** Already built — `DATA/ATLAS/FEATURES_RUN_B2/2026_02_20.parquet`
**Status:** Ready to run through Stage 1 + Stage 2 pipeline.

---

### Run C-2 — Rolling Sample Window (Continuous Stream)
**Concept:** Pure N-sample rolling window that slides forward 1 second at a time. Every 1s bar triggers a full recomputation of all SFE layers using the trailing N samples. Window sizes are design parameters independent of TF labeling — a "continuous stream of higher-TF context."

**Key difference from C:** Run C matched windows to HTF wall-clock sizes. C-2 treats N as a free hyperparameter.

**Status:** Dataset build script to be written.

---

### Run B-3 — Continuous HTF Recompute + Kalman Smoothing
**Concept:** Same as B-2 but each rolling feature is passed through a Kalman filter instead of a raw rolling statistic. The Kalman filter provides an exponentially-weighted smooth estimate that handles the "window drop-off cliff" (where an old spike exits the trailing window and causes a discontinuous jump).

**The Kalman cure:**
```
x^_t = x^_{t-1} + K * (z_t - x^_{t-1})
Where z_t = raw feature at time t, K = Kalman gain
```
Old data loses influence *gradually* instead of cliff-dropping to zero.

---

### Run C-3 — Rolling Sample Window + Kalman Smoothing
**Concept:** Same as C-2 but with Kalman smoothing applied per-feature.

---

## 3. THE ZOH CLIFF PROBLEM

Both B-2 and C-2 still suffer from the "zero data cutoff cliff" — when an old data point exits the trailing window, the feature value abruptly jumps. The Kalman cure (B-3/C-3) provides exponential decay of old data rather than a hard cutoff, giving all features physical inertia.

---

## 4. SUMMARY TABLE

| Run | Anchor | Feature Update | Window Type | Kalman | Avg Segment | Status |
|-----|--------|---------------|-------------|--------|------------|--------|
| A | 1s | On bar close | 1s only | No | 40.4s | Done |
| B (original) | 5s | On HTF close | Multi-TF | No | 164.0s | Done (dropped) |
| C | 1s | On bar close | Mirrored wall-clock | No | 53.8s | Done |
| D | 1s | On bar close | Same as C | No (dynamic) | 33.1s | Done |
| C Cap-40 | 1s | On bar close | Mirrored wall-clock | No (cap-40) | 53.6s | Done |
| **B-2** | **1s** | **Every 1s** | **HTF rolling** | **No** | **TBD** | **Dataset built, pipeline pending** |
| **C-2** | **1s** | **Every 1s** | **Rolling N-sample** | **No** | **TBD** | **Proposed** |
| **B-3** | **1s** | **Every 1s** | **HTF rolling** | **Yes** | **TBD** | **Proposed** |
| **C-3** | **1s** | **Every 1s** | **Rolling N-sample** | **Yes** | **TBD** | **Proposed** |

---

## 5. KEY FILES

| File | Description |
|------|------------|
| `research/fspace_experiment/stage1_speed_pass.py` | Modified: max_features = min(40, length - 2) |
| `DATA/ATLAS/FEATURES_RUN_B2/2026_02_20.parquet` | B-2 dataset (66,549 rows x 338 cols) |
| `comms/fspace_findings.md` | Updated findings and architecture decisions |
| `scratch/build_run_b2.py` | B-2 dataset builder script |

---

## 6. OPEN QUESTIONS FOR NEXT SESSION

1. **B-2 Pipeline Run:** Stage 1 + Stage 2 for B-2 not yet executed — run to get avg segment length.
2. **C-2 Dataset Builder:** Need to write `build_run_c2.py` with configurable N-sample rolling windows.
3. **Kalman Parameters:** What process noise (Q) and measurement noise (R) to target for B-3/C-3? Options: (a) tune empirically per feature, (b) fixed conservative K=0.1.
4. **TF Inclusion:** Should B-2/C-2 include 30m and 1h TFs, or cap at 15m to keep feature count manageable?
