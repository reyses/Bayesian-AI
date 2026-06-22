# HANDOVER → Gemini — Regime/Markov "dictionary": verified audit + the definitive causal null test (2026-06-16)

Goal: settle whether the `research/research_entry.md` regime-transition thesis has ANY tradeable
causal content, or is a base-rate illusion. Claude audited it against the real data (below);
this doc merges that with a pre-committed, null-controlled OOS test so we can KILL it or keep it
on evidence, not vibes.

## A. WHAT research_entry.md CLAIMS
- 80,717 segments → 3,029 "Regimes" via O(N) polynomial template-matching (tier 1–4, `<2.5×` bound).
- "Top 30 regimes = 85.6% of price action; not a random walk."
- "Three Universal Laws" from `artifacts/transition_matrix.npy` (3030²): Inertia P(R1→R1)=62.93%;
  Vol-snapping P(R2→R1)=52.01%; Chaos→Order P(Noise→R1)=45.09%.
- §6 proposes integrating `transition_matrix.npy` into the live Bayesian filter as a prior.

## B. CLAUDE'S VERIFIED AUDIT (computed from artifacts/transition_matrix.npy + regime_buckets.json)
**The "laws" are base-rate illusions.** Regime 1 alone = **45,559 / ~76,900 segments = base rate P(→R1)=56.44%**.
| "Law" | Reported | Base rate | Lift | Reality |
|---|---|---|---|---|
| Inertia (R1→R1) | 62.93% | 56.44% | **+6.5 pp** | barely above chance |
| Vol-snap (R2→R1) | 52.01% | 56.44% | **−4 pp** | *below* chance |
| Chaos→Order (Noise→R1) | 45.09% | 56.44% | **−11.3 pp** | **opposite of claim** — chaos is LESS likely than chance to reach R1 |

- R1 is mostly loose matches (tier-1 "flawless" = 26 vs tier-2 = 20,940, tier-4 = 7,874). The "dictionary"
  is really **one mega-shape (56%, generic low-vol drift) + a long tail**, not a rich vocabulary.
- This reproduces the **2026-06-12 audit** (`reports/findings/2026-06-12_segment_regime_audit.md`) which
  already flagged inertia=base-rate+6.5pp and chaos→R1 below base rate. The writeup ignored it.
- Consistent with **today's causal chaos-precursor KILL** (`reports/findings/chaos_precursor_causal.md`):
  fittability/chaos adds **+0.000 AUC** over volatility persistence (IS 2024-03 + OOS 2025-03).
- **FIREWALL:** §6's live-prior plan reintroduces lookahead — a segment's regime is fit over its
  WHOLE span (incl. its future), so it's unknown until the segment completes. Do NOT wire
  `transition_matrix.npy` into the live filter on these results.
- Missing: **no null/random-walk control** behind "not a random walk." That gap is the test below.

## C. THE CAUSAL TESTS — two framings, both null-controlled, pre-committed (BUILD THESE)
Scripts: `research/regime_markov_causal_test.py` (A) + `research/regime_causal_earlypredict.py` (B).
Write results to `reports/findings/`.

### Framing A — does the PRIOR completed regime predict the NEXT? (the transition-matrix claim)

#### Step 1 — reconstruct the chronological regime sequence (causal labeling)
1. Load `artifacts/stage2_year_segments.json`; sort segments by `(day, start_idx)` → ordered list, index = sequence position t. This index space MUST match the segment indices used in `regime_buckets.json` member lists — verify by spot-checking that bucket member indices are valid into this ordered array (FLAG if they don't align; the buckets index the flattened segment array from `phase2_extract_cache.py`).
2. Build seg→label: for each segment, find every regime whose tier-1..4 member list contains it; assign the regime where it sits in the **highest tier** (tier1>tier2>tier3>tier4); ties → larger regime. Segments in no regime → label `NOISE` (matches the ~6,616 noise count). Result: sequence r_1..r_N.

### Step 2 — temporal split (NO leakage)
- TRAIN = chronologically first 70% (e.g., 2025-01..2025-07), TEST = last 30% (2025-08..2025-10). Strict OOS.

### Step 3 — models + baselines (predict r_{t+1} from r_t)
- **Marginal baseline:** always predict argmax π(train) (= R1). Also its full distribution for log-loss.
- **Transition model:** P(r_{t+1}|r_t) estimated on TRAIN (Laplace-smooth, e.g. +1). Predict argmax row + full row dist.

### Step 4 — metrics on TEST (day-block bootstrap CI, ≥2000 resamples, block = day)
- **Top-1 accuracy:** transition vs marginal. Report Δacc + 95% CI. (CI includes 0 → no edge.)
- **Log-loss** (proper score): transition vs marginal. Report Δ + 95% CI.
- Report both the overall numbers AND a per-source breakdown for R1 / R2 / NOISE (to see if any source beats its own conditional base rate).

### Step 5 — NULL CONTROLS (the "make sure it's dead" part)
- **Null A (sequence shuffle):** permute the TRAIN sequence, rebuild the transition matrix, re-eval on TEST. Repeat 1,000×. The REAL Δacc/Δlogloss must exceed the **95th percentile** of this null to claim genuine temporal structure.
- **Null B (circular block-shift):** shift the sequence by random offsets in blocks (preserve marginals + short-run autocorr, break long-range). Same comparison. Distinguishes real Markov content from trivial persistence.
- **Null C (random-walk surrogate, optional/heavier — the definitive answer to "not a random walk"):** generate a synthetic Gaussian random-walk price series matched to MNQ 5s vol; run the SAME segmentation + template-matching pipeline (`phase2_*`/`phase3_*`); check whether it ALSO yields ~one-regime-56% concentration and ~"85.6% in top-30". If a random walk reproduces the concentration, the anti-random-walk claim is fully refuted. Only run if A/B somehow pass.

### Step 6 — PRE-COMMITTED decision rule
- **KEEP** only if: OOS transition beats marginal on BOTH accuracy AND log-loss, 95% day-block CI excludes 0, AND real Δ > 95th pct of Null A. 
- **DEAD** otherwise → record in MEMORY §4 graveyard; do NOT integrate into the live filter.
- Prior expectation (Claude): DEAD — in-sample transitions already sit at/below base rate (2 of 3 laws are below), so OOS+null almost certainly confirms base-rate illusion. The test exists to make the kill rigorous, not to rescue.

### Framing B — can you know the regime CAUSALLY before the segment completes? (SMEP-informed; deployment-relevant)
Framing A only asks if regime_t predicts regime_{t+1}. §6's live prior needs MORE: predict a *forming*
segment's regime from its first K bars — WITHOUT the whole-segment fit (which is the firewall violation).
The full-year SMEP (`reports/findings/smep/`) told us which CAUSAL features carry the structure:
reversion (`z_se/z_high/z_low/hurst/reversion_prob`), kinematics (`price_velocity/price_accel`),
bar-shape (`wicks/body`), vol (`sigma/vol_accel`). This framing tests whether that vocabulary can
actually call the regime early. Script: `research/regime_causal_earlypredict.py`.

1. **Target — COARSE & actionable, NOT 3029 classes:** predict the segment's `volatility_tier`-band
   (low 1-3 / mid 4-6 / high 7-9) and/or `status` (PRISTINE/RECOVERED/CHAOS). "Which of 3029" is
   meaningless (R1=56%); the tradeable question is the regime CLASS/tier.
2. **Causal features:** over the segment's FIRST K bars (K ∈ {6,12,30} = 30s/1m/2.5m) + trailing context,
   pull the SMEP families straight from the already-materialized causal `FEATURES_5s_v2` parquets — no
   segment fit. Strictly `[seg_start, seg_start+K)`; nothing past it.
3. **The null to beat = TRAILING-VOL-ONLY** (mirror the chaos-precursor test): tier ≈ vol, so a vol-only
   model already scores high — the SMEP set must **add over it** (increment in AUC/log-loss) or it's
   just vol in disguise (which is the chaos-precursor finding).
4. **Temporal 70/30 split**; multinomial-logistic or gradient-boost; OOS accuracy / macro-F1 / log-loss
   vs (a) marginal baseline, (b) vol-only baseline. Day-block CI.
5. **Null control:** shuffle labels across segments, retrain, re-eval → real must beat the 95th pct.
6. **Decision:** KEEP only if SMEP features beat BOTH the marginal AND the vol-only baselines OOS, CI
   excludes 0, and beats the shuffle null. Else the regime is NOT causally knowable early beyond vol
   → §6's live-prior is dead.
- **Prior (Claude):** tier is partly predictable (it ≈ vol, which persists), but **beating vol-only is the
  real bar** — and the chaos-precursor kill (+0.000 over vol) is a strong prior it WON'T. This is the
  honest, deployment-relevant version; run it AFTER Framing A.

## D. HARD RULES (same as the rest of the program)
- Causal only / no hindsight; temporal split, never random split (sequence is time-ordered).
- Day-block bootstrap CI; state significance explicitly (CI incl 0 → not significant).
- No magic numbers; write results + a plot to `reports/findings/`; update `docs/daily/`.
- FIREWALL: regime labels are non-causal post-hoc fits — even a "predictive" transition is a
  next-SEGMENT (coarse) signal, not a bar-level live prior. Flag this if the test somehow passes.

## E. POINTERS
- Audit math: recompute via `artifacts/transition_matrix.npy` (int64 counts, 3030²) + `artifacts/regime_buckets.json`.
- Related: `reports/findings/2026-06-12_segment_regime_audit.md`, `reports/findings/chaos_precursor_causal.md`,
  `reports/findings/smep/` (hierarchical SMEP: common families = reversion + kinematics + bar-shape, firewall-bound),
  `research/chaos_precursor_causal.py`, `research/smep_hierarchical_segments.py`.
- Pipeline (archived): `archive/research/Regression segments/` + repo-root `artifacts/` (full 2025 + 2026-Q1 stage2).

## F. THE RESEARCH LINE'S REAL CONTRIBUTION (keep this — it pointed the right way)
Even though the "Three Laws" are base-rate artifacts and the live-prior plan is firewall-blocked, this
research line **pointed in the right direction.** Its segmentation + the full-year (223-day) SMEP
**validated the structural feature vocabulary** — reversion (`z`/`hurst`/`reversion_prob`) + kinematics
(`velocity`/`accel`) + bar-shape (`wicks`) — which is exactly what the live NMP uses, what the
orange-line/Kalman thread is built on (velocity/accel = the Kalman states), and exactly what Framing B
feeds on. The value was never the "dictionary of 3029 regimes"; it was **confirming WHICH families carry
the structure.** Preserve that takeaway; discard the laws and the transition-matrix-as-live-prior plan.
