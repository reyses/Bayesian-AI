---
name: distilled-regime_clustering
description: Segment-regime buckets + transition matrix are DEAD as a causal/live signal (base-rate illusion); regime labels retained ONLY as a hindsight/oracle diagnostic.
metadata: {type: distilled, topic: regime_clustering, status: dead}
---
## Verdict
Asked whether the "112,289-segment" regime bucket dictionary and its transition
matrix ("Three Universal Laws of Transition") hold genuine predictive power over
a base-rate illusion. A 2026-06-12 audit found the mega-bucket is a band-tolerance
artifact, not a market state; a formal 2026-06-16 causal/OOS test then confirmed
the transition matrix and early-prediction models are worse than trivial
baselines. DECISION: DEAD as a live/causal input. Retained only as a non-causal
label-side oracle-ceiling diagnostic.

## Key numbers (with CIs where they exist)
- Segments: 112,289 total; PRISTINE 58,655 / PURE_CHAOS 31,572 / RECOVERED 22,062; valid universe 80,717 (`reports/2026-06-12_segment_regime_audit.md`).
- Buckets: 3,029; Regime 1 = 45,559 members = 56.4% of all segments (the base rate).
- "Law of Inertia" P(next=R1|cur=R1) = 62.93% (n=45,558) vs base rate 56.44% → lift +6.48pp (1.11x); honest contiguous-only recount = 61.42% (n=21,176), lift +4.98pp.
- "Chaos Resolution" P(next=R1|cur=NOISE) = 45.09% (n=6,616) vs base rate 56.44% → lift **−11.35pp** (0.80x, worse than chance); contiguous-only = 43.92% (n=3,165), lift −12.52pp.
- Timeline integrity: 80,716 consecutive valid pairs; 341 spurious cross-day pairs; of same-day pairs, 51.8% span a gap (median 15 bars, p90 60, max 11,230).
- Self-match diagonal: 21.64% (17,468/80,717) of the valid universe is tier-8 (broken) on its own data.
- Bucket-root correlation: Spearman(root degree, root error-band) = +0.142, p=3.8e-15 (n=3,029 roots) — degree driven by loose tolerance, not similarity.
- **Framing A (transition matrix, OOS)**: ΔAcc = −0.0008 (also reported −0.0009), 95% CI [−0.0018, −0.0001]; Null-shuffle 95th pct = −0.0007 → real delta does NOT clear the null (`reports/regime_markov_causal_test.md`, `regime_markov_test_summary.txt`).
- **Framing B (early SMEP prediction vs trailing-vol baseline, OOS)**: Full SMEP Acc 0.5230 vs Vol-Only 0.5272; ΔAcc = −0.0042, 95% CI [−0.00765, −0.00076]; Null shuffle 95th pct = 0.5275 (`regime_earlypredict_summary.txt`).
- SMEP hierarchical betas (in-sample, non-causal): top common terms L0_time_of_day median|β| 8.863 (n=9,372), L1_1m_body 2.639 (n=8,703), L1_1m_price_velocity_1b 2.496 (n=11,131); most tier-shifting term L3_1m_SE_low_15 Δavg|β| across tiers = 27.472 (`reports/smep_hierarchical.md`).

## Graveyard / never-retry (if any)
- Regime transition matrix as a live Kalman prior: DEAD — fails OOS vs marginal baseline AND fails the shuffle null (both framings, `regime_markov_causal_test.md`).
- Early causal prediction of forming-segment volatility tier from rich SMEP kinematics at seg_start+30 bars: DEAD — loses to a simple trailing-volatility baseline OOS.
- "Law of Inertia" / "Chaos Resolution" as named universal regime laws: not supported — the +6.5pp / −11.4pp lifts are explainable by known day-level volatility autocorrelation, not new structure, and one of the two claimed effects is actually a *negative* lift.

## Reusable assets
- `tools/audit_regime_findings.py` — reproduces the full phase2-4 composition/timeline/bucket/self-match audit.
- `tools/phase4_transition_matrix.py`, `phase2_analyze_adjacency.py`, `phase3_analyze_buckets.py`, `phase3_analyze_tier_breakdown.py` — bucket/transition construction and analysis pipeline.
- `tools/smep_hierarchical_segments.py` — hierarchical Standardized Main-Effects Plot generator over regression segments (diagnostic only).
- `tools/phase3_animate_regime.py`, `phase3_plot_top_regimes.py` — visualization of top regimes.
- Sibling topic `research/regime_markov_causal_test/` holds the actual causal-test scripts (`regime_markov_causal_test.py`, `regime_causal_earlypredict.py`) that produced Framing A/B results cited above.

## Data locations
- `artifacts/stage2_year_segments.json` — 112,289 structural segments (source of the audit).
- `artifacts/regime_buckets.json` — bucket/tier dictionary.
- `DATA/ATLAS/FEATURES_5s_v2/` — causal features used for the Framing B early-prediction test.
- `reports/smep_L1.png`, `smep_L2_tier.png`, `smep_L3_band_tod.png` — SMEP plots.

## Open threads
- none — task explicitly closed as DEAD/causal-firewalled; constructive-next-steps list in the audit (symmetric band-independent similarity, mutual-coherence bucketing, PURE_CHAOS as explicit Markov state, day-block CIs) was NOT executed as a follow-up per the files present.

## Sources
- research/regime_clustering/reports/2026-06-12_segment_regime_audit.md
- research/regime_clustering/reports/regime_markov_causal_test.md
- research/regime_clustering/reports/regime_markov_test_summary.txt
- research/regime_clustering/reports/regime_earlypredict_summary.txt
- research/regime_clustering/reports/smep_hierarchical.md
- research/regime_clustering/tools/audit_regime_findings.py
- research/regime_clustering/README.md, project.md (both effectively empty)

## Archive recommendation
ARCHIVE (reason: causal path is formally DEAD per two independent OOS+null tests;
regime labels survive only as a label-side oracle-ceiling diagnostic under the
SEGMENT FIREWALL, not as an active research line — no open threads remain).
