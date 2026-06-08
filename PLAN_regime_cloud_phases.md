# Regime Template Cloud — Implementation Plan (3 Phases)

Consolidates the design session. Plan-level, not code-level. Each phase has a hard **exit gate** — do not start the next phase until it passes. The whole program is itself gated (see Prerequisites).

---

## PREREQUISITES (before Phase 1)
1. **Base policy edge** must be plausible — sentinel clean, corrected-reward run validated. A regime cloud on an edgeless base = N edgeless experts.
2. This is **measurement-first and empirical** — every quantity counted from observed history. No OOS in any loop. Regimes are feature-defined and forward-observable, never failure-defined or datetime-indexed.

---

## PHASE 1 — R-Curve in F-space (fit + convergence probe)

**Objective:** fit, per candidate span, a regression curve mapping **feature space → price** (price = Y), and FIRST establish that such curves actually separate into distinguishable shapes. If they don't, stop — nothing downstream is real.

**Steps:**
1. Define candidate spans over the IS timeline (windows to fit curves on).
2. Fit one curve per span: `price = f(features)`. Curve order must be ≥ quadratic (a line has only slope, no *shape*).
3. **Convergence probe (the go/no-go):** fit curves on a handful of spans and test whether they cluster at all vs. smear into mush. Cheap, visual + a few distance measurements. This is the bedrock test the whole program rests on.

**Decisions to make (Phase 1):**
- Regression order / form (quadratic? spline? — needs *shape*, not just slope).
- Span definition: fixed window length? overlapping? how candidate spans are proposed.
- Which feature subset enters the curve (regime features: hurst, reversion_prob, z_se, … vs full set).

**EXIT GATE:** the convergence probe shows curves form **distinguishable, repeating shape groups**. If curves don't separate → **HALT**. Do not build Phase 2. (This is the explicit "we don't yet know if data converges" checkpoint.)

---

## PHASE 2 — Buckets (all-vs-all matrix)

**Objective:** from the fitted curves, build the **all-vs-all similarity matrix** and assign each historical bar a bucket label. This matrix is the durable primary artifact — downstream uses (refit, stratify, etc.) are forks held open.

**Steps:**
1. **Probe sets** — push a fixed probe through every curve to compare curves via shared evaluation points. Use BOTH:
   - synthetic grid (even coverage of feature space — pure geometry), and
   - real historical slice (where the market actually lives — realistic weighting).
   Agreement across both = robust similarity; disagreement is itself diagnostic. **Real-data probe governs the decision; synthetic is diagnostic.**
2. **Two gates per pair:**
   - **Shape gate** — distance between curve *outputs* on the probe (do they predict differently?). Student's t (comparable variance).
   - **Generalization gate (cross-fit)** — run A's own data through B's curve and vice versa; compare R-adjusted / residuals (does each explain the other's reality?). **Welch's t** (residuals are heteroscedastic). Run both directions — asymmetry = one regime is more general than the other.
3. **Cell computation:** each all-vs-all cell = the pairwise t-result. Apply a **multiple-comparison correction** (Holm/Bonferroni — ~780 tests at 40 buckets would manufacture ~39 false splits otherwise), OR treat the t-stat as a continuous distance and read the gradient.
4. **Confidence interval on every cell** (t-interval; **Wilson** for proportions). Cells whose interval is too wide to distinguish from "no difference" are flagged **UNKNOWN**, not taken at point value. Report **effective-N** per cell (swarm/correlation deflates raw counts).
5. **Bucket assignment:** group from the matrix block structure. **Over-segmentation is the safe error** — accept redundant/duplicate seeds; a later refit fork can collapse them. (Merging-down is recoverable; under-splitting is not.)

**Decisions to make (Phase 2):**
- Correction method (Holm vs continuous-distance).
- How block structure → final groups (by eye from the grid vs a rule).
- Effective-N / interval-width thresholds for the UNKNOWN flag.

**EXIT GATE:** a stable bucket set with a populated similarity matrix where a meaningful fraction of cells are statistically trustworthy (not all UNKNOWN). If most cells are UNKNOWN → insufficient data per bucket; widen spans or reduce bucket count before proceeding.

---

## PHASE 3 — Inference & Probabilities

**Objective:** use the bucketed history to (a) identify the current regime live via progressive narrowing, and (b) forecast the next regime via a measured transition matrix.

**Steps:**
1. **Transition matrix** — read the bucket-label sequence in time order; count `P(next = B | now = A)`. Measured, not assumed. Confidence interval (Wilson) on every transition cell; sparse cells = UNKNOWN.
2. **Mutual check** — cross-validate the two matrices: if transition says A→B is frequent but similarity says A≈B, they may be one drifting regime, not a transition. Surface these.
3. **Progressive narrowing (t-10 → t-1):** at each approach bar, ping all regime curves consistent with the current evolving setup; the consistent set shrinks bar by bar. By t-1 a handful survive — the regimes that "rhymed" with the present across the whole approach.
   - Collapse speed = setup distinctiveness.
   - **No collapse / no good match = novelty → abstain / size down** (the cloud's edge over a monolithic net: it can say "I've never seen this").
4. **Forecast:** current-regime posterior (from narrowing) × transition matrix = next-regime distribution. This is a measured **Markov chain over regimes** (sequential Bayesian updating).

**Decisions to make (Phase 3):**
- Hard nearest-bucket vs **soft/Bayesian weighting** by feature-distance (soft = the Bayesian-table form; better if regimes are continuous, not discrete).
- Markov order (first-order vs path-dependent — higher order needs far more data per cell).
- Novelty/abstain threshold.
- Stationarity monitoring (do transition probabilities drift? — the non-stationarity watch).

**EXIT GATE:** narrowing reliably collapses to a small candidate set on known setups, abstains on novel ones, and transition forecasts beat a naive base-rate prior — measured out-of-sample (read once).

---

## INVARIANTS (hold across all phases)
1. Feature-defined, forward-observable regimes — never failure- or datetime-defined.
2. No OOS in any build/segmentation/transition loop; OOS read once for validation only.
3. Every cell (similarity and transition) carries a confidence interval; UNKNOWN cells are not treated as their point estimate.
4. Signature/feature-indexed, not calendar-indexed — regimes recur and experts are reusable.
5. All quantities empirical and auditable (spreadsheet-checkable).

## PARKED (explicitly out, revisit later)
- Refit/collapse mechanism for redundant seeds (a Phase-2 fork, not required to build the matrix).
- Per-regime expert training (the original MoE) — only if buckets prove distinct edge.
- Higher-order Markov, stationarity drift correction.
