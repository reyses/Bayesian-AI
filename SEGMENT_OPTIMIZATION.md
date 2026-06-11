# Segment Optimization — Implementation Plan

**Goal:** make Phase-1 segment extraction as fast as possible (~50 min/day → target **seconds/day**) **without changing the segmentation semantics**, by loading the full day into VRAM and reducing the entire regression layer to GPU prefix-sum batched OLS, with feature selection decoupled onto a coarse cadence.

**Status:** plan only. Current extractor = `scratch/phase1_fast_extraction.py` (CPU screen + GPU per-segment batched OLS). This plan supersedes its compute approach.

---

## 1. Problem / cost model

Per-segment cost today is wildly unequal:

| Stage | Cost | Why |
|---|---|---|
| CPU **screen** (GroupLasso 185-feat + 2× ElasticNetCV cv=3) | **seconds — dominates** | iterative solvers on the full grid, re-run per seed attempt (incl. every roll-by-1) |
| GPU **OLS path** (fixed ≤15 terms) | microseconds of math | trivial solve |
| per-L `.cpu().numpy()` in the break loop | **overhead** | up to ~2000 host syncs/segment |

Two root causes of slowness: (a) the **screen is re-run far too often** (per roll bar), and (b) the **path does per-L host transfers** instead of staying on device.

**Key insight:** the *regression* (OLS with a fixed term set) batches for free via prefix sums; the *feature selection* does not. So speed = **minimize screen calls + keep all regression on-device**. Everything collapses once selection is decoupled from regression.

---

## 2. Target architecture

```
[load day → VRAM]  (≈13 MB: N×185 f32)
        │
        ├── COARSE SCREEN  (few calls/day) ──►  fix Φ (≤15 degree-2 terms) per block
        │        (CPU sklearn OR batched GPU-FISTA — the only remaining real cost)
        │
        └── PREFIX-SUM OLS CORE  (GPU, whole day in a few kernels)
                 S_xx = cumsum(φφᵀ),  S_xy = cumsum(φ·close),  S_x = cumsum(φ)
                 → every window / anchor / length fit = prefix slices + one batched solve
                 → violations (10% / 5-consec) = vectorized GPU reduction
                 │
                 └── SEQUENTIAL RE-ANCHOR MARCH  (cheap walk over precomputed prefixes)
                          roll-to-clean seed → expand-to-break → re-anchor → repeat
```

Once Φ is fixed for a block, **all rolling-window fits, all anchored expansions, and all break checks are prefix-sum slices + batched solves** — no per-window loop, no per-L transfer, no recompute on expansion.

---

## 3. The math (fixed Φ ⇒ prefix-sum OLS)

Let Φ ∈ ℝ^{N×p} be the fixed degree-2 design (p ≤ 15), `close ∈ ℝ^N`, `TICK = 0.25`.

Precompute three cumulative sums over bars (one pass, GPU):
```
S_xx[k] = Σ_{j<k} φ_j φ_jᵀ        # [N, p, p]
S_xy[k] = Σ_{j<k} φ_j · close_j   # [N, p]
S_x [k] = Σ_{j<k} φ_j             # [N, p]
```
For any segment `[a, a+L)` anchored at `a` (Y in ticks = (close − close[a]) / TICK):
```
ΦᵀΦ      = S_xx[a+L] − S_xx[a]
Φᵀy      = ( (S_xy[a+L] − S_xy[a]) − close[a]·(S_x[a+L] − S_x[a]) ) / TICK
β_L      = solve(ΦᵀΦ + λI, Φᵀy)          # ridge λ=1e-6
```
- **Rolling 30-bar seeds (all starts):** windowed prefix diffs → one batched solve → every seed fit at once → clean-seed scan is vectorized.
- **Anchored expansion (one anchor, all L):** slice prefixes from `a` → batched solve for all L → break check vectorized. No recompute.

Anchoring folds into the prefix sums (the `close[a]·S_x` term), so re-zeroing per segment costs nothing.

---

## 4. Criterion (unchanged semantics)

- Anchor each segment: `Y = (close − close[seg_start]) / TICK` (ticks, 0 at anchor).
- **Violation:** `|actual − pred| ≥ max(REL_TOL·|actual|, MIN_TOL_TICKS)` (REL_TOL=0.10; MIN_TOL_TICKS=1.0 floor for the anchor region — **tunable knob**).
- **Window OK** while `≤ MAX_CONSEC (5)` consecutive violations.
- Roll a 30-bar seed forward to the first clean window; expand until a 5-consec run forms; cut; re-anchor.

Break detection vectorized: max-consecutive-True per length via a GPU cumulative reset-scan; `L* = ` largest L with `max_consec(viol over [0:L]) ≤ 5`.

---

## 5. Implementation phases (each independently testable)

### Phase A — Prefix-sum OLS core (GPU), fixed Φ
- `prefix_sums(Phi, close) → (S_xx, S_xy, S_x)` via `cumsum` of outer products (`einsum('np,nq->npq')`).
- `fit_window(a, L)` and `fit_anchor_all_L(a, L_min, L_max)` from prefix slices + batched `solve`.
- **Parity test:** for random `(a, L)`, prefix-sum β must match a direct `lstsq` to 1e-4. Gate Phase A on this.

### Phase B — Coarse-cadence screen (fix Φ per block)
- Screen once per block of `SCREEN_STRIDE` bars (start: whole-day or per ~N-min block). Output: active raw set → degree-2 expand → `fixed_terms` (cap 15) → Φ for that block.
- Keep the **CPU sklearn** screen (GroupLasso → ElasticNetCV) initially (proven, sidesteps FISTA); leave a hook to swap in batched GPU-FISTA later.
- **Decision knob:** `SCREEN_STRIDE` (how often Φ may change). Default: per block; allow whole-day.

### Phase C — Sequential re-anchor march over precomputed prefixes
- With Φ (and prefixes) for the current block: vectorized clean-seed scan (all 30-bar fits) → first clean seed; `fit_anchor_all_L` → vectorized break → `L*`; emit segment; `seed_start += L*`; repeat.
- Roll-to-clean and expansion both read precomputed prefixes → no re-fit, no re-screen mid-segment.

### Phase D — Criterion, curve-mode, outputs
- `--curve-mode {refit,frozen}`: **refit** (β per L from prefixes) vs **frozen** (β from the seed, extrapolate forward). Frozen drops the per-L solve entirely and is the cheaper, arguably-truer regime-persistence test — A/B it.
- Output `artifacts/phase1_fast_segments_<day>.json`: same schema as current (start/end/length/anchor/active_grid_cells/surviving_polynomial_indices/beta_coefficients/max_consec_violations/max_abs_resid_ticks).

### Phase E — Validation
- **Perf:** wall-clock per day (target: seconds; hard ceiling: << 10 min).
- **Parity:** segment boundaries vs current `phase1_fast_extraction.py` on 1 day must match within tolerance (same criterion, faster compute) — or differences explained (Φ now fixed per block vs per seed).
- **Population sanity:** segment count (tens–hundreds), length distribution, coverage (Σlength / N_TOTAL — large gaps ⇒ roll rejecting too much ⇒ tune MIN_TOL_TICKS / criterion).

---

## 6. Knobs / decisions
| Knob | Default | Effect |
|---|---|---|
| `SCREEN_STRIDE` (Φ cadence) | per block (or whole-day) | the only remaining real cost; bigger = faster + more stable, less adaptive |
| `MIN_TOL_TICKS` | 1.0 | anchor-region floor; raise if segments over-break right after each anchor |
| `REL_TOL` | 0.10 | tolerance = 10% of bar's anchored value |
| `MAX_CONSEC` | 5 | consecutive violations allowed before a break |
| `--curve-mode` | refit | refit-per-L vs frozen-seed extrapolation |
| `MAX_TERMS` | 15 | degree-2 term cap (keeps p small ⇒ prefixes tiny) |

---

## 7. Risks / trade-offs (stated honestly)
1. **Fixed Φ per block loses per-window feature adaptivity.** Mitigation: if the active set genuinely shifts that fast, it wasn't one regime — and per-window selection is what *caused* the active-set churn / instability we measured. So fixed-Φ is **faster AND more stable**; this is a feature, not just a compromise.
2. **No stability score / break-geometry emitted** (the 100×-bootstrap was dropped — it was also the bugged code pinning stability to 0.0). **Phase 2's GATE 0 ("mostly STABLE / SHARP") therefore needs a separate fast stability proxy** before bucketing. Out of scope here; tracked.
3. **VRAM sanity:** day = ~13 MB; prefixes for p≤15 = ~16 MB. Trivial. *Full-grid degree-2 (185→17k terms) is impossible (1.2 GB+ design, 17k² prefix) — confirms selection MUST precede expansion.*
4. **Determinism:** prefix-sum order is fixed; ridge λ stabilizes near-singular ΦᵀΦ on short windows.

---

## 8. Out of scope (do NOT build here)
- Phase 2 all-vs-all bucketing / similarity matrix (separate plan; gated on a Phase-1 stability proxy).
- Stability/break-geometry re-introduction (decide later: fast proxy vs corrected bootstrap).
- Cross-day parallelism (orthogonal; one GPU stream at a time).

---

## 9. Acceptance
Ship when: Phase A parity passes (1e-4 vs lstsq), a full day runs in **seconds**, segment population is sane (count/length/coverage), and boundaries reconcile with the current extractor (or differences are explained by the fixed-Φ cadence). Then re-open the Phase-2 GATE-0 question with a stability proxy.
