# JULES — Stage 1 / Stage 2 Overhaul: build the F-space MAP (not a predictor)
n> **Code relocated 2026-06-22 → `research/fspace_cadence/` (pipeline/ builders/ tools/ reports/). See its README.md.**

**Status:** SPEC / instructions. Author 2026-06-18. Execute against this; do not improvise the schema.
**Companion findings:** `reports/findings/b2c_continuous_2024_02_20.md`, `reports/findings/forward_increment_2024_02_20.md`.
**Builders already in place:** `build_run_b2c.py` (continuous sliding-window), `build_run_b2t.py` (tiled),
`make_brownian_null.py` (`--null brownian|fourier`). Corrected break rule already wired into `stage1_speed_pass.py`.

---

## 0. Objective (read first — it changes what we record)
The R-curve's job is **NOT to predict price.** It is to **MAP the F-space**: how cleanly the feature space
explains the price *path*, and **how that image distorts across scenarios** (real vs null, regime vs regime).
Mapping is the **first step to prediction** — we map, then make assumptions, then build a causal predictor.

The two things we ultimately need to predict (NOT magnitude — we do not care about price size):
1. **Trend DIRECTION** — which way the current regime is going.
2. **The FLIP** — when the current regime ends (the boundary / turning point).

A **segment is a directional regime**; a **boundary is a flip**. So Stage 1/2 must record, per regime, everything
needed to (a) characterize direction + flip-timing, and (b) later train a CAUSAL predictor of both.

### Non-negotiable guardrails
- **Segment firewall:** segment betas/tiers/boundaries are non-causal (fit over their own future). The MAP may
  use them (it is a diagnostic). Any *predictor* built later must use ONLY causal features at the regime start
  → never the in-sample segment quantities as inputs.
- **Null discipline:** every map quantity must be produced for **real + Brownian + a Fourier surrogate ENSEMBLE**
  so each number has a null distribution. "Real has structure" = real distorts beyond the surrogate band, with a
  p-value — never n=1.
- **No magic numbers:** all thresholds are named constants with a comment.
- **Mode-first reporting:** distributions + mode/median, never a bare mean; mean only with 95% CI.

---

## 1. What's broken in current Stage 1/2 for this objective
1. **The R-curve is computed then DISCARDED.** `batched_ols_scan_pytorch` produces `tiers_exp/max_res_exp/betas_exp`
   for every candidate length L, but Stage 1 keeps only `L_star` + final `max_residual` + final betas. **The whole
   explainability-vs-horizon curve is thrown away** — that curve IS the map.
2. **No direction recorded.** Net move / regime slope / sign is never stored.
3. **No flip characterization.** Why the segment broke, and the residual trajectory *approaching* the break (the
   pre-flip signature that a flip-timing predictor would learn), are not stored.
4. **No composite.** There is no aggregation of per-segment R-curves into a composite distortion curve, and no
   real-vs-null comparison.
5. **No causal bridge.** Nothing pairs (causal features at regime start) → (eventual direction, duration) so a
   predictor can be trained later.

---

## 2. STAGE 1 OVERHAUL — the per-regime mapper
Keep the corrected break rule (`off[t] = |actual−pred| > max(BAND_FRAC·|Δclose[t]|, TICK_FLOOR)`; break on
`BREAK_CONSEC` consecutive off). Add the following to **each segment record** (both PRISTINE and CHAOS where it applies):

### 2a. Full R-curve (the core addition)
For every candidate length `L` evaluated during forward expansion, record a row:
```
r_curve: [ {L, r2, max_resid, n_off, consec_off}, ... ]   # L from SEED_BARS .. L_break
```
- `r2` = 1 − SS_res/SS_tot of the anchored target `Y=close−close[start]` over `[start, start+L]` (explained variance).
- `max_resid` = max |actual−pred| at length L.
- `n_off`, `consec_off` = count and max-run of off-bars at length L (per the corrected rule).
- **Record PAST the break** too: continue the curve for `FLIP_TAIL_BARS` (e.g. 20) bars beyond `L_break` so we
  capture how the image *distorts through* the flip (the curve doesn't just stop at the boundary).

### 2b. Direction descriptors
```
slope_pts_per_s : OLS slope of close over the segment (pts / second)
net_move_pts    : close[end] − close[start]
direction       : sign(net_move_pts)  ∈ {−1, 0, +1}
```

### 2c. Flip descriptors (for flip-timing prediction)
```
break_reason       : 'consec_off' | 'end_of_data' | 'chaos_giveup'
consec_off_at_break: the run length that triggered the break
resid_tail         : the last FLIP_LOOKBACK_BARS (e.g. 15) residuals before L_break  (the pre-flip signature)
tier_at_break      : tier just before breaking
```

### 2d. Causal bridge pointer
```
raw_start_idx, raw_end_idx : already present — KEEP (robust). These join to the causal feature parquet so a
                             predictor can pull features AT regime start (causal) and pair them with the
                             direction/duration LABEL from this map.
```

### 2e. New constants (top of file, commented)
```
FLIP_TAIL_BARS    = 20   # bars to keep tracing the R-curve PAST the break (distortion-through-flip)
FLIP_LOOKBACK_BARS= 15   # residual tail kept before the break (pre-flip signature)
# BAND_FRAC=0.10, TICK_FLOOR=0.25, BREAK_CONSEC=5 already defined.
```

---

## 3. STAGE 2 OVERHAUL — chaos re-map + COMPOSITE builder
1. **Re-segment chaos blocks** with the same corrected rule and the **same enriched per-segment schema** as Stage 1
   (full r_curve, direction, flip descriptors). RECOVERED + PURE_CHAOS as today, but enriched.
2. **Build the COMPOSITE R-CURVE (the map artifact)** — aggregate ALL segments (pristine + recovered):
   - **Explainability curve:** for each forward offset `k`, the distribution (median, q25/q75) of `r2` across all
     regimes that reached length ≥ k → "how far forward does the F-space explain the path before it distorts."
   - **Survival / hazard curve:** fraction of regimes still alive (un-flipped) at offset `k` → **the flip-timing map.**
     Hazard(k) = P(flip at k | alive at k). This is the "when it flips" distribution.
   - **Direction map:** histogram of `slope_pts_per_s` and `direction`; serial dependence P(next=same); slope vs
     duration relationship.
   - **Flip signature map:** average `resid_tail` trajectory (does the R-curve degrade GRADUALLY before a flip =
     predictable, or SNAP = unpredictable?).
3. **Output** `artifacts/map_<run>_<day>.json` with all four curves + a per-regime table.

---

## 4. NULL / ENSEMBLE HARNESS (so the map has error bars)
New driver (e.g. `research/fspace_experiment/build_map.py`):
- Inputs: representation (`FEATURES_RUN_B2C` and `FEATURES_RUN_B2T`), day, `K_SURROGATES` (e.g. 30).
- Generate: real, 1 Brownian, **K Fourier surrogates** (vary seed; reuse `make_brownian_null.py`, add `--seed`).
- Run Stage1→Stage2→composite on each → collect K null maps + real map.
- Produce **real-vs-null** overlay for every curve (explainability, hazard, direction, flip-signature) with:
  - the null DISTRIBUTION (median + band across the K surrogates),
  - real plotted against it,
  - a **rank-based p-value** `p = (1 + #{null ≥ real}) / (1 + K)` per summary statistic.
- Write `reports/findings/map_<rep>_<day>.md` — mode-first, with the null bands and p-values.

---

## 5. CAUSAL BRIDGE (map → the two predictors we actually want)
Once the map exists, the predictors (separate, causal, firewall-clean):
- **Direction model:** X = causal F-space at `raw_start_idx` → y = regime `direction`. Score OOS vs Fourier null
  (does causal feature state at a regime's birth predict its direction beyond the surrogate?).
- **Flip / hazard model:** X = causal F-space (rolling) → y = time-to-flip (survival). Score vs null.
- Both reuse the harness in `forward_increment_test.py` (train/test time-disjoint, embargo, null-calibrated) but with
  the NEW targets (direction sign / time-to-flip) instead of price increment.

---

## 6. ACCEPTANCE CRITERIA (what "done" means)
1. Stage 1 + Stage 2 emit the enriched per-segment schema (§2, §3) for any `features_root`, real or null.
2. `build_map.py` produces the four composite curves with **K-surrogate null bands + p-values**, for B2C and B2T.
3. The map answers, mode-first, real-vs-null:
   - **Explainability:** does real explain the path further forward than its surrogate? (distortion magnitude)
   - **Flip-timing:** is real's hazard/survival curve different from null? (is "when" structured?)
   - **Direction:** is real's direction distribution / serial dependence different from null?
   - **Flip signature:** does real's pre-flip R-curve degrade gradually (predictable) vs null's snap?
4. Every claim carries a p-value vs the surrogate ensemble. No n=1.

## 7. GUARDRAILS RESTATED
- Firewall: map uses in-sample segments; predictors use only causal-at-start features.
- Null-anchored: structure = real beyond surrogate band, with p-value.
- mappable ≠ tradeable: the map is the objective here; exploitation is the later, separate gate.
- No magic numbers; mode-first; write all outputs to files (findings + journal).
```
EXECUTION ORDER: §2 (Stage1 dump) → §3 (Stage2 + composite) → §4 (ensemble) → §5 (predictors).
Smallest first runnable slice: §2 r_curve dump + §3 composite on the ALREADY-BUILT B2C/B2T 2024_02_20 + the
3 existing series (real/brown/four) — gives the first distortion map with no new heavy builds.
```

---

## TODO / PARKED IDEAS
- **[TODO] Rolling-VWAP as direction/flip primitive** (2026-06-18). VWAP was excluded from the F-space as an
  aggregation, but for DIRECTION/FLIP it's a legitimate reference level (your VP/level world): direction = side
  of VWAP / VWAP slope sign; flip = VWAP cross / reversion. MUST use a ROLLING/windowed VWAP (session-cumulative
  is anchored, heavy, lags flips late-session). Smallest first look = descriptive: how price orbits rolling-VWAP
  (distance dist, cross-timing, side-persistence), REAL vs Brownian/Fourier null, mode-first. No machinery.
  Guardrail: VWAP is very smooth -> artifact-prone -> must beat the nulls. Revisit after the R-curve map.

## SCALE-UP PLAN (funnel) — 2026-06-18
- **1 day = wide screen** (2024_02_20): all variants (B2C / B2T / Run C / C-2 ladder) × real/Brownian/Fourier.
- **Week = deep validation, SURVIVORS ONLY.** Do NOT re-run the full matrix on a week. Take only the variant+axis
  that beats the null on day 1 (current lead: TILED B2T, flip-timing/survival axis) and run a contiguous 2024 IS
  week. Week buys: (1) regime diversity (does the 8x survival edge reproduce?), (2) day-block bootstrap CIs on the
  real-minus-null gap, (3) kills the n=1-day caveat. Per-day nulls are cheap (generators take --day).
- Order: finish 1-day screen -> name survivor -> small multi-day driver -> week run with CIs.

- **[TODO] Anchor -> wait-N-bars -> direction (confirmation window)** (2026-06-19). Direction is a coin flip AT the
  anchor (P(next regime=same)~50%). Idea: anchor at a flip, WAIT N bars, THEN commit direction = direction of the
  rest of the regime. Synthesis with the flip-timing result: waiting both (a) lets the regime show its hand and
  (b) confirms it's the PERSISTENT kind (real regimes survive ~8-20x the null), so there's move left to ride after
  confirming; the null wouldn't reward this (its regimes die fast).
  - GUARDRAIL: the N-bar-confirmed direction must predict the REST of the regime BETTER ON REAL THAN ON THE FOURIER
    NULL. Else it's just momentum a random walk also has. Null-anchored, always.
  - N is a SWEEP (1..K), not a pick (user floated 2 vs 6). Tradeoff: more bars = better read but later entry + less
    remaining move + giveback ("over-waited, right but late" - Kalman regret). OBJECTIVE = (direction-edge-over-null)
    x (remaining regime length), not accuracy alone.
  - Bar unit = the bar-close cadence that mattered (tiled); keep N small vs ~40s regimes (6x15s eats the regime).
  - This is the DIRECTION-predictor test = where L4/lambda (fade-vs-ride) finally gets exercised (survival never did).
  Build after the week. Reuse forward_increment_test harness with direction target + N-bar-delayed anchor.

- **[TODO] RKHS — two concrete uses ONLY (a test + a nonlinear predictor, NOT the map)** (2026-06-19).
  1. **MMD / kernel two-sample test** -> upgrade the null comparison: instead of one scalar (survival@45s real vs
     Fourier), test whether the FULL real F-space distribution differs from the surrogate's, with a principled
     p-value (uses the whole feature cloud). Directly sharpens the null discipline we rely on.
  2. **Kernel ridge / GP** -> the NONLINEAR extension of the (linear, failed) forward-increment test AND the
     anchor->wait-N direction predictor; GP adds calibrated uncertainty ("how confident is the direction call").
  - DO NOT use RKHS for the segment fit / the MAP: a smooth kernel OVER-SMOOTHS (the B2C failure) and DESTROYS
    interpretability (hides which feature/window carries signal) — wrong tool when the goal is understanding *what*
    the structure is. RKHS is a test + a fitter, not a structure-discovery tool.
  - CAUTIONS: O(n^2-n^3) scaling at ~50k pts/day -> needs Nystrom/sparse/subsample on the 3060; kernels overfit ->
    bandwidth+reg CV mandatory; always null-anchored (kernel ridge on real must beat kernel ridge on the surrogate).

- **[TODO] OU as the FADE regime model (alongside linear-RIDE) in the segmenter** (2026-06-19).
  Already used: L3 reversion_prob = OU first-passage (OU_BOUNDARY=3.0). NEW use: the R-curve fits LINEAR drift =
  RIDE (lambda>0). Mean-reverting/snap-back = FADE (lambda<0) is OU, NOT linear -> the linear segmenter calls fade
  regimes "chaos". Add an OU fit as an alternative segment model; classify each block TREND(linear) vs FADE(OU),
  take the better fit. Directly the NMP fade/ride dichotomy + the direction question (fade=bet snap-back, ride=bet
  continuation). Could reclassify much of the "chaos". theta (mean-reversion speed) -> half-life ln2/theta = a
  parametric flip-timing estimate (complements the empirical survival curve).
  - OU-as-NULL is likely REDUNDANT with Fourier (which already preserves the autocorr encoding OU's decay) - skip
    unless we want a parametric mean-reverting null specifically.
  - CAUTIONS: theta estimation NOISY on ~40-pt segments (needs several reversion cycles); OU is an idealization
    (stationary/Gaussian/symmetric/constant theta) - real snap-backs are asymmetric/level-dependent.

- **[TODO] HMM for direction/flip — ONLY as OU-fade/linear-ride states, null-anchored, causal** (2026-06-19).
  GRAVEYARD WARNING: generic regime-Markov ALREADY DIED on this system - the "Three Laws" were base-rate artifacts,
  causal test dead both framings (MEMORY §3/§4). Do NOT retry a generic regime-HMM.
  Conceptually HMM is the right shape: states=direction/regime, transitions=flips, emissions=feature dists; and the
  forward-filter is CAUSAL (past-only) -> firewall-clean, unlike the non-causal segmenter. Transition probs = the
  flip-timing/hazard; state posterior = the direction call.
  ONLY worth a retry if it DIFFERS from the dead "Three Laws":
    1. states = FADE(OU) vs RIDE(linear drift) vs chaos - physically-grounded emissions (tie to the OU TODO), not
       arbitrary EM clusters.
    2. NULL-ANCHORED from the start - state-separation + transition structure must beat the Fourier surrogate (a
       random walk admits a spurious HMM fit; the Three Laws died exactly because transitions were base-rate).
    3. test CAUSALLY (filter, OOS), not in-sample fit.
  CAUTIONS: HMMs overfit (enough states fit anything), EM local optima, #states is a free knob, Markov(memoryless)
  may be wrong if regimes are history-dependent; prior evidence says the TRANSITION layer specifically was dead.
