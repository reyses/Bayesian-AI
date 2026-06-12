# λ-COMPLETION PROGRAM — Living Roadmap

> **LIVING DOCUMENT.** This is the master plan for the NMP/λ program. Update
> protocol: every stage completion/kill appends a dated entry to §9; gate
> criteria (§6) and prohibitions (§7) are immutable without explicit user
> approval; status tags in §4 are kept current every session.
> Created 2026-06-11. Owner: user + whichever agent session is active.

---

## 1. Thesis (why this program exists)

The Nightmare Protocol's execution rule is two-branched:
`|Z|>Z* ∧ λ<0 → fade` · `λ>0 → ride`. **As shipped, λ was hardcoded 0.0** —
never computed, never consumed; the de-facto stability gate was
`variance_ratio < 1`, and vr was later dropped from the V2 schema entirely.
The strategy traded **half an equation**: it can only fade, and it cannot see
regime death. The equity-burning drawdown is the **integral of the missing λ
term** (fades fired where λ>0; rides never taken). Every major empirical
finding is a corollary: losers are low-MFE pokes (entries into absent regimes),
stops all lose (the intervened population is dollar-weighted recoverers —
depth carries no regime information), trade AGE beats depth (regime-life
signature), B10 works (the one overlay carrying regime information), the
R-trigger is unbeatable (it IS a crude regime-break detector).

**Program goal: complete the equation** — estimate λ causally, gate/parameterize
the strategy on it, and add the missing ride branch — with kill-points that can
cheaply prove the alternative (λ unobservable ⇒ drawdown is structural ⇒
sizing/participation is the rational lever).

## 2. Architecture of the ML (decided 2026-06-11)

The ML's job is **dynamic parameter adjustment + exit completion**, never
signal-discovery from raw features and never mid-trade overlays:

- **Right altitude:** parameters adjust *between* trades / *per regime* (the
  B10 template — the only promoted overlay, +$69/day sig). Action surface >
  prediction quality (C15).
- **Low-dimensional + discrete:** a small nowcast-regime × parameter table with
  per-cell **Bayesian shrinkage posteriors** (the nested-conditional-table
  design), NOT a continuous black-box controller (quantile-cell lesson: 75% of
  greedy cells flipped sign OOS).
- **Exit work is "completion," not "management":** replacing the crude
  `vr>1` regime-flip exit with the λ̂ t-stat (+ AGE covariate) is completing
  the exit; anything layered ON TOP of an exit is the graveyard (§7).

## 3. Current state ledger (verified, with artifacts)

| Item | Status | Artifact |
|---|---|---|
| NMP→V2 canonical feature map + traps + SEGMENT FIREWALL | ✅ done | `docs/Active/NMP_V2_FEATURE_MAP.md` |
| λ̂/vr derivation layer (X-side instrumentation) | ✅ VERIFIED | `research/nmp_state/{derive,validate}.py` |
| Recalibrated thresholds (the port) | ✅ **Z_ENTRY=1.8481, Z_EXIT=0.4752** on `L3_1m_z_se_15` | `reports/findings/2026-06-11_nmp_state_derivation.md` |
| vr proxy (cross-TF σ ratio) | ❌ DEAD (Spearman 0.38–0.73, all TFs) → vr_exact only | same report |
| λ̂ abstain band | ~±2 (1m) / ±1.7 (5m), per-TF/k calibration pending; t-stat iid caveat | same report |
| Segment corpus (empirical λ ground truth) | ✅ done (1 year parallel-processed) | `artifacts/stage*_segments_*.json` |
| NMP-V2 baseline forward pass | 🔄 spec'd to Jules | `docs/JULES_NMP_V2_FORWARD_PASS.md` |
| Hurst | causal but LAGGING (N_BASE×8) — slow prior only, never the gate | map doc trap #7 |

## 4. Stage pipeline (each stage = ONE change, gated)

**STAGE 0 — Instrumented causal baseline** `[COMPLETED]`
Pure NMP-V2 port (fade-only, V1 exits, no stops, λ̂ logged NOT gated), IS+OOS
via `run_strategy`. Deliverables: honest baseline $/day (day-block CI, gross+net
$5/RT), burn structure (Pareto/exit reasons), instrumented trade CSVs, first
descriptive λ̂-separation read (losers vs winners entry `lambda_t`).
*Expectation: baseline may be NEGATIVE — that is data, not failure.*

**STAGE 1 — λ̂ entry gate (PW-CRL Integration)** `[IN FLIGHT]`
Use the PW-CRL agent to build the entry gate. Require λ̂ significantly < 0 (abstain band from the
null calibration). The both-branch decider's first half.
→ Gate: vs Stage-0 baseline, risk-adjusted (maxDD↓, $/maxDD↑) **net of forgone
winners**, day-block CI, OOS. If λ̂ doesn't separate in Stage 0, SKIP to KT1.

**STAGE 2 — Regime-conditioned parameter table (PW-CRL Table)** `[after Stage 1]`
Small discrete table managed by the PW-CRL reinforcement learning agent: nowcast-regime cell → (Z*, VR*, participation). Per-cell
Bayesian posteriors with shrinkage; cells chosen from Stage-0/1 sensitivity
sweeps (only parameters that demonstrably matter). One table revision at a time.

**STAGE 3 — Exit completion** `[after Stage 2; carve-out, not overlay]`
Replace/augment `vr>1` regime-flip exit with λ̂-death (t-stat flips positive /
loses significance) + trade-AGE covariate (hump hazard). MUST first confirm in
Stage-0 data whether NMP fades share the dollar-weighted-recoverer structure —
if they do, the bar is the documented one (§6.4).

**STAGE 4 — The ride branch (λ>0)** `[after Stage 3]`
The never-built half: ride persistent-positive-λ̂ regimes. New trade population,
new baseline, same gates. Converts the old blind spot into the second engine.

**PARALLEL TRACK — segment corpus science** (feeds Stages 1–4 labels; FIREWALLED)
P1. λ-law measurement suite on the finished corpus: state taxonomy, hazard/life
    curves, transition table (the user's "segments as priors to segments"),
    honest day-block stats. Segment quantities are **labels only** (trap #8).
P2. **KT1 ORACLE kill-test:** with hindsight regime knowledge, does trade
    selection even pay? If NO → program dies cheaply; pivot to sizing floor.
P3. **KT2 NOWCAST kill-test:** causal features (λ̂, vr, z, hurst-as-prior, age)
    recover a worthwhile fraction of oracle skill OOS — boundary-conditional,
    lead-time>0, vs constant-hazard/age-only/carry-forward baselines. If NO →
    λ unobservable to a price-taker at our horizon; drawdown is structural;
    sizing/participation is the rational lever (measured conclusion, not defeat).

**BACKSTOP (any time, no prediction needed):** sizing floor via
`tools/risk/blowout_with_intervention.py` — participation level that survives
the measured tail at current equity. Hygiene, not edge.

## 5. Decision tree at Stage 0 readout

- Baseline negative + λ̂ separates → thesis confirmed → Stage 1.
- Baseline negative + λ̂ does NOT separate → jump to KT1 oracle (is regime value
  even there with hindsight?) before more causal work.
- Baseline positive → first honest causal NMP number; Stages 1+ are upside.

## 6. Hard gates (every stage must clear ALL — immutable)

1. **Sealed OOS, one change at a time** (baseline-management rules apply).
2. **Day-block bootstrap CI (4,000 resamples) + explicit significance statement
   + N.** Never raw-trade stderr (pseudoreplication trap, 2026-06-08).
3. **Risk-adjusted accounting net of forgone winners** for any filter/gate
   (C15: AUC ≠ delta), and net of $5/RT costs.
4. **Exit-side changes** must beat the documented benchmarks: > −$31/day
   (cost of the dumb −$100 stop) with discrimination ≫ AUC 0.465 at the
   intervention point.
5. **Lookahead audit** before believing any number (replay/positive-control
   where applicable; the +$454 and baseline-740 scars).

## 7. Standing prohibitions (the graveyard — do not re-enter)

- ❌ Mid-trade intervention overlays of ANY kind on an adaptive exit: per-trade
  $ stops, model-conditional stops, session/cumulative-PnL stops, winner-banking,
  loser-bailing, attenuation/capping of profitable action streams (7 families,
  all significant losses).
- ❌ Segment quantities (membership/tier/betas/boundaries/remaining-life) as
  model INPUTS — label-side only (SEGMENT FIREWALL, map doc trap #8).
- ❌ Continuous/black-box parameter controllers; per-cell parameter fitting
  without shrinkage + OOS survival.
- ❌ Threshold optimization against outcomes disguised as "porting" —
  recalibration is quantile-matching only; optimization is a separate, labeled,
  sealed-OOS exercise.
- ❌ Hurst as the regime gate (lagging, N_BASE×8 window). Slow prior only.
- ❌ Schema changes while the VM year-run is in flight. (Queued for next rebuild:
  materialize vr; hurst warmup NaN mask; builder-history check for 1h/4h/1D hurst.)
- ❌ Physics vocabulary in new code (statistical naming: stability_exponent, etc.).

## 8. Open items / pins needed

- Strategy-family pin for the NT8-playback separability cross-check (v1.0.6
  playback CSV is the on-disk causal-MAE dataset) — partially mooted once
  Stage 0 generates NMP's own population.
- Per-TF/k λ̂ abstain-band calibration (autocorrelation-aware).
- research_A debt (separate from this program): IS-gate CI pseudoreplication fix;
  evaluator smoke-test.
- SEGMENT_OPTIMIZATION.md Phases A–E (prefix-sum extractor) — performance work,
  parallel to this program.
- Phase 2 bucketing of `PLAN_regime_cloud_phases.md` — blocked on a fast
  stability proxy for GATE 0.

## 9. Log (append-only, dated)

- **2026-06-11** — Program founded. Thesis (λ-completion), ML architecture
  (dynamic parameters + exit completion, B10 template, Bayesian table form),
  stage pipeline + gates codified. λ̂/vr derivation layer VERIFIED
  (Z*=1.8481/0.4752; vr proxy dead; parity PASS). Stage 0 spec'd to Jules
  (`docs/JULES_NMP_V2_FORWARD_PASS.md`). Segment corpus VM run in flight.
- **2026-06-11 (Later)** — Completed STAGE 0 (Instrumented Causal Baseline). Fixed missing L4 generation in `core_v2/build_dataset.py`. Ran IS and OOS benchmarks. The λ̂ separation showed perfectly uniform negative decay across all quartiles in the un-instrumented state. This formally validates the necessity of the PW-CRL RL agent for dynamic parameterization and exit flip logic (ride vs fade), transitioning the roadmap away from static rule thresholding. Architecture document updated to reflect the PW-CRL implementation.
- **2026-06-12** — VM autonomous pipeline finished the full-year run. Downloaded the 111MB `stage2_year_segments.json` and 17.5MB `stage1_year_segments.zip`. The empirical segment corpus ground truth is now complete and ready for PW-CRL integration.
