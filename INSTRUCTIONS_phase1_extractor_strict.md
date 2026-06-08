# INSTRUCTION SET — Phase 1 Segment Extractor (Antigravity) — DO NOT DEVIATE

Build `scratch/phase1_strict_extraction.py`. Implement EXACTLY the algorithm below. Where a step says "ONCE," running it more often is a bug. The FORBIDDEN list at the bottom is non-negotiable.

---

## A. HARD RULES
1. **Phase 1 segment extraction ONLY.** Output is a list of segments + their records. Build NO probe, PCA, dendrogram, HDBSCAN, matrix, buckets, or experts.
2. **Screen-then-surface.** Never expand the full 185-feature grid to degree-2. Screen to a small active set first; expand only the survivors.
3. **Global standardization** of the raw 185 features using IS-wide mean/std, computed ONCE up front, applied BEFORE any expansion. Never per-span.
4. **Seed-then-trim** extraction (NOT grow-from-tiny).
5. **Selection (elastic-net/group-lasso) runs at most TWICE per segment** (seed screen + final screen). The one-bar trim scan uses **closed-form OLS on the fixed terms** — never re-select inside the scan.
6. **Block-bootstrap** for stability (never row-bootstrap — targets overlap, row resampling fakes stability).
7. Target `Y` = forward 27-bar return. Acausal, OFFLINE ONLY. Never a routing input.

---

## B. PARAMETERS (exact)
| Name | Value |
|---|---|
| Base bar | 5s |
| `SEED_BARS` | 120 (≈10 min) |
| `MIN_BARS` (trim floor) | 80 (must exceed active-term count; assert) |
| `R2ADJ_THRESHOLD` | 0.85 |
| Trim step | 1 bar |
| Screen groups | 24 = TF(8) × Layer(L1/L2/L3) |
| Group-lasso α | CV via `TimeSeriesSplit(n_splits=3)` (NOT hardcoded) |
| Within-group select | Elastic-Net, `l1_ratio=0.5`, CV `TimeSeriesSplit(3)` |
| Final surface | degree-2, Elastic-Net CV `TimeSeriesSplit(3)` |
| Bootstrap | 100 **block** resamples, block size = 27 bars |
| Stability accept | active set reproduces in ≥ 80% of blocks |

---

## C. ALGORITHM (implement in this exact order)

**Setup (ONCE):**
1. Load IS data. Compute global mean/std of the 185 raw features. Standardize all. Compute `Y` = forward-27-bar return per bar.

**Per-segment loop (march forward):**
2. `seed_start` = end of previous segment (start at 0).
3. Define seed window `W = [seed_start, seed_start + SEED_BARS]`.
4. **SCREEN #1 (ONCE, on full seed W):**
   a. Stage A — group-lasso over the 24 TF×Layer groups → surviving groups.
   b. Stage B — elastic-net within surviving groups → surviving raw features.
   This fixes the **active feature set** for the trim search.
5. **Expand** the active features to degree-2 → the **fixed term set** (record term count `p`; assert `p < MIN_BARS`).
6. **TRIM SCAN (GPU-batched, closed-form OLS — NOT elastic-net):**
   For every trailing length `L` in `[MIN_BARS … SEED_BARS]`:
   - Fit the **fixed-term** surface via OLS on `[seed_start, seed_start+L]`.
   - Compute `R²adj(L) = 1 − (1−R²)(L−1)/(L−p−1)`.
   Batch all `L` in one parallel solve (same term structure, different row counts). Produce the **R²adj-vs-L curve**.
7. **Find boundary:** `L* = the largest L with R²adj(L) ≥ R2ADJ_THRESHOLD`.
   - If NO `L ≥ MIN_BARS` meets threshold → emit `NOFIT` for this seed, advance `seed_start += MIN_BARS`, go to 3.
   - If `L*` is at/near `MIN_BARS` (degenerate tiny segment) → flag `DEGENERATE`.
   - Segment = `[seed_start, seed_start + L*]`.
8. **SCREEN #2 (ONCE, on the clean window `[seed_start, seed_start+L*]`):** re-run Stage A→B to record the **definitive active-set fingerprint**. If it differs materially from Screen #1, set `SEED_CONTAMINATED` flag (the seed straddled a boundary). Record the final-window fingerprint as the segment's fingerprint.
9. **STABILITY (block-bootstrap):** 100 resamples of contiguous 27-bar blocks within the segment; re-run Stage A→B on each; if the active set reproduces in ≥80% → `STABLE`, else `UNSTABLE`.
10. **BREAK GEOMETRY:** from the R²adj-vs-L curve near the `L*` crossing — a cliff (recovers in 1–3 bars of trim) = `SHARP`; a slow ramp (10–15 bars) = `GRADUAL`. Tag via the curve's local slope at the crossing.
11. Set `seed_start = seed_start + L*`; repeat from 3.
12. **Discard segment #1** (warm-up — no clean left edge).

---

## D. FORBIDDEN DEVIATIONS (these are bugs — do not do them)
- ❌ Re-running screen/elastic-net inside the trim scan (step 6 is OLS-only, fixed terms).
- ❌ Row-bootstrap for stability (must be 27-bar block-bootstrap).
- ❌ Expanding the full 185 grid to degree-2 (screen first).
- ❌ Per-span standardization (global only, step 1).
- ❌ Sequential trim loop where each step waits on the last (batch the L scan).
- ❌ Growing from tiny instead of seed-then-trim.
- ❌ Building anything past the segment list (no probe/cluster/bucket/expert).
- ❌ Hardcoding regularization α (all CV'd via TimeSeriesSplit).

---

## E. DELIVERABLES
`phase1_segments.pkl` + human-readable JSON, per segment:
- `start_idx`, `end_idx`, `length` (= L*)
- `active_grid_cells` (surviving TF×Layer), `surviving_terms_and_coeffs`
- `r2_adj` at L*
- `stability_score`, `UNSTABLE` flag
- `break_geometry` ("sharp"/"gradual")
- `SEED_CONTAMINATED`, `DEGENERATE`, `NOFIT` flags as applicable

Plus a summary: N segments, length distribution, STABLE vs UNSTABLE ratio, SHARP vs GRADUAL ratio, NOFIT count.

---

## F. RUN & STOP
Run on 1–2 ATLAS days. Produce the summary. **STOP and request review.** Do not proceed to Phase 2.

**Result read (state it in the summary):** mostly STABLE + mostly SHARP + varied fingerprints → segments are real, Phase 2 justified. Mostly UNSTABLE or mostly GRADUAL → segmentation is a fitting artifact; do not proceed.
