# SPEC — Policy & Reward v2 (FINAL, red-teamed, build-ready)

Supersedes v1. Resolves THESIS §10, retrofits aperiodicity, and patches the 7 red-team findings. Invariants locked: path-independence, leak wall, additive components, process-based scoring, vol-normalized units, regret-on-entry-only.

## 0. Aperiodic frame
- **Oracle move** = swing between consecutive significant labeled turns (kσ-filtered; human-verified where available, calibrated auto-labeler elsewhere). No half-cycle/fixed-horizon constructs anywhere.
- **Exit = hazard**: P(swing ends now | causal state). Episode = session (22:00 UTC), force-flat at close, per-session hidden reset. **Position size = 1 contract in beta** (sizing is a separate, later decision per thesis).

## 1. Architecture
Shared Mamba trunk → **entry head** (3-way {long, short, flat}, active when flat), **exit head** (2-way {hold, exit} hazard, active in-position), **value head** (A2C/GAE). State-dependent masking. The "three heads" of the thesis = reward decomposition + funnel diagnostics, not three softmaxes.
- **Auxiliary hazard loss (NEW):** supervised BCE on the exit head — target = turn-imminent from the labeled turns. Config weight `w_aux` (start 0.2). Labels are target-side ⇒ zero leak. This is the sample-efficiency lever: the exit head learns turn-timing from the gold labels, not RL reward alone.

## 2. Reward — trade-terminal scorecard (all vol-normalized, clipped [−1,1], paid at trade close / swing completion)
**Cost first (P0 fix):** every closed trade pays `R_cost = −w_cost · (ticks_cost/σ_ticks)`, ticks_cost = spread+commission+slippage (MNQ ~2–4 ticks, config). Capture is **net of cost** — no cost-free fantasy market.

**Capture (the edge), with late-entry guard (P0 fix):** `w_c=1.00` × capture_rate = captured/(entry→turn **remaining extent**).
- **Denominator floor:** remaining extent ≥ `θ_rem` (vol-normalized, config) — else the trade scores **no capture** (falls to selectivity/wiggle logic). Kills the enter-at-90%-for-free-100% exploit and the divide-by-small instability.
- Quality gate Q = MFE/(MAE+ε) applies to the **remaining** swing from entry, not the whole swing.

**Direction:** `w_d=0.20` (±).

**Cut bonus (P1 fix — must net POSITIVE on fast cuts):** `w_x=0.35` × exp(−t_hold/τ)·exp(−MAE/σ). Ceiling (0.35) > direction penalty (0.20) ⇒ a fast cut on a wrong trade nets **+0.15-ish, small positive** per thesis §5. Decay handles anti-over-cutting.

**Wiggle penalty, coverage-gated (P2 fix):** `w_w=0.15` — fires **only in label-covered regions**. In auto-labeler gaps: no wiggle penalty, no regret (symmetric null) — the agent must never learn label *coverage* instead of market structure. Coverage mask emitted in diagnostics.

**Regret, windowed (P2 fix):** `w_r=0.25` × c_t, per missed qualifying (Q≥θ_Q) swing, **credited only to the flat-action bars where c_t ≥ θ_c during the readable entry window** — not smeared across the whole flat stretch. Capped once per swing, path-independent.

## 3. Clarity gate (P1 fix — frozen + calibrated)
- c_t = **picks classifier confidence** (causal ≤t features, trained on the human labels) = "knowability."
- **Version-pinned per training run** — the classifier NEVER retrains mid-run (non-stationary reward is forbidden). New classifier version ⇒ new run.
- **Calibrated** (Platt/isotonic) before gating reward — raw scores from ~300 labels are not probabilities.
- Enters the **reward only**, never the observation.

## 4. Density & training
Trade-terminal (no mark-to-market; paper-gain shaping teaches holding through turns). Potential-based shaping = config escape hatch, off by default. A2C+GAE; fixed-window TBPTT w/ detached carry; B4 session reset; B5 OOM E-exit. Entropy on both heads: β 0.01 → 0.001 linear over first 50%, floor 0.001. Entropy fights mode collapse; regret fights flat-collapse.

## 5. Starting weights (all config, tuned ONLY via the funnel)
`w_c 1.00 · w_x 0.35 · w_r 0.25 · w_d 0.20 · w_w 0.15 · w_aux 0.2 · w_cost = actual costs (not tunable — real).`
Funnel rules: under-trading → ↑w_r/↓w_w · over-cutting → ↓w_x · too_late cluster → check hazard head + capture anchor.

## 6. Edge cases
Held through turn → capture<1, `too_late`, no extra penalty. Session force-flat → score to close, no timeout penalty. Label-gap regions → no regret/wiggle, capture anchors to nearest labeled turn or realized exit (flagged). Multiple missed swings → each once, capped.

## 7. Gates & critical path
- Beta scaffold builds now (per `SPEC_REWARD_BETA_IMPLEMENTATION.md`; these weights = config).
- **Training conclusions gated on a stable labeled turn set — label quality IS reward quality. The IS labeling grind (cusp_marker + cubic overlay + retrain loop) is the critical path.**
- Fourier-null discipline carries to eval: the hazard head's turn-prediction must beat the phase-randomized null before any live claim.
