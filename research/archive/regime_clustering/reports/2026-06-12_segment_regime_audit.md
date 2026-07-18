# Segment-Regime Findings Audit (2026-06-12)

## 1. Composition
- total segments in stage2_year_segments.json: 112,289
- status counts: {'PRISTINE': 58655, 'PURE_CHAOS': 31572, 'RECOVERED': 22062}
- valid (PRISTINE|RECOVERED) = phase2/3/4 universe: 80,717

## 2. Phase-4 timeline integrity (transitions = consecutive valid entries)
- valid list chronologically day-ordered: True
- consecutive pairs: 80,716 total = 80,375 same-day + 341 CROSS-DAY (spurious overnight "transitions")
- same-day pairs physically contiguous (end_idx==next start_idx): 38,747 (48.2%)
- same-day pairs with a GAP (removed/failed segment between them): 41,628 (51.8%)
- gap size (5s bars): median 15, p90 60, max 11230

## 3. Bucket structure
- buckets: 3,029; classified segments: 76,900/80,717 (95.3%); NOISE(unbucketed): 3,817
- top-10 bucket sizes: [45559, 6176, 5054, 3338, 2911, 1844, 1794, 645, 314, 212]
- bucket-size median: 2; buckets with <10 members: 2,938 (97.0%)
- Regime 1: total 45,559 members of which tier1+2 20,966 (tier3+4 = loose membership: 24,593)
- share of ALL segments held by Regime 1: 56.4%   <- the base-rate that the "inertia" claim must beat

## 4. Claim verification (as-built: ALL consecutive pairs, incl. cross-day/gapped)
- Regime 1 -> Regime 1 ("Law of Inertia"): claimed 63%
    measured P(next=R1 | cur=1) = 62.93%  (P(stay)=62.93%, n=45,558)
    base rate P(any segment = R1) = 56.44%  ->  LIFT = +6.48 pp (1.11x)
- NOISE -> Regime 1 ("Chaos Resolution"): claimed 45%
    measured P(next=R1 | cur=0) = 45.09%  (P(stay)=12.33%, n=6,616)
    base rate P(any segment = R1) = 56.44%  ->  LIFT = -11.35 pp (0.80x)
- transition matrix sparsity: 3030x3030 = 9,180,900 cells, nonzero 7,526 (0.08%); total transitions 80,716

## 5. Claim verification (HONEST timeline: same-day, contiguous pairs only)
- Regime 1 -> Regime 1: P(next=R1) = 61.42%  (P(stay)=61.42%, n=21,176)  vs base rate 56.44%  ->  LIFT +4.98 pp
- NOISE -> Regime 1: P(next=R1) = 43.92%  (P(stay)=12.45%, n=3,165)  vs base rate 56.44%  ->  LIFT -12.52 pp

## 6. Adjacency self-match (diagonal = curve i on its own segment)
- diagonal tier distribution: {1: 2636, 2: 25552, 3: 32191, 4: 2870, 8: 17468}
- self-match in tier 1-2: 28,188/80,717 (34.92%)
- self-match BROKEN (tier 8 on own data): 17,468 (21.64%)

## 7. Diagonal explained: pipeline FAITHFUL, universe metric-inconsistent
Recorded stage1 self-fit tiers (max_residual/error_band from stage2_year_segments.json)
match the measured sweep diagonal EXACTLY (2,636 / 25,552 / 32,193→32,191 / 2,868→2,870 /
17,468 — two segments of float jitter at tier boundaries). Conclusion:
- The phase2 reconstruction (scaler, column order, NaN-reindex, poly mapping) faithfully
  reproduces stage1. The debug_segment_self_match "seg 6" failure was evidently resolved.
- BUT 21.6% of the "valid" (PRISTINE/RECOVERED) universe is tier-8 ON ITS OWN DATA:
  the segmentation rule (break only on >5 CONSECUTIVE out-of-band bars) and the matching
  rule (max-residual ratio) are different metrics. 17,468 segments can never reach tier 4
  on themselves yet participate as both targets and curves.

## 8. ROOT CAUSE of the mega-bucket: error band in the tier denominator
adjacency[i,j] = max_residual(curve j on segment i) / segment i's OWN error band.
A loose-band target is "fit" by everyone -> max degree -> becomes a bucket root.
- Regime 1 root: error_band = 11.075 = 99.9th percentile (13.4x the population median 0.825).
- ALL top-6 bucket roots sit at the 91.8-99.9th percentile of error band.
- Spearman(root degree, root band) = +0.142, p = 3.8e-15 (n = 3,029 roots).
"Regime 1" is therefore NOT a market state — it is the loosest tolerance tube in the
dataset; 45,559 curves pass through it trivially. Member-to-member similarity within a
bucket is never established (membership ties members to the ROOT's data only).

## VERDICT
- 3,029 buckets = 1 band-artifact mega-bucket (56.4%) + 6 mid buckets + ~3,000 near-
  singletons (median size 2; 97% under 10 members).
- "Law of Inertia" (63%) = base rate 56.4% + ~5-6.5pp lift (1.11x). Binomial SE on
  n=45,558 is ±0.5pp so the lift is nominally significant, BUT segments within a day are
  autocorrelated (no day-block correction) and band ∝ local volatility — the residual
  lift is plausibly the KNOWN day-level volatility clustering (vol autocorr IS +0.275 /
  OOS +0.485, 2026-05-21), not new structure. Does NOT shatter any random walk.
- "Chaos Resolution" (45%) is BELOW the 56.4% base rate (lift −11.4pp, 0.80x): noise is
  LESS likely than chance to be followed by R1. Also phase4's "NOISE" = 3,817 unbucketed
  valid segments — the 31,572 PURE_CHAOS blocks are absent from the timeline entirely.
- Timeline: 51.8% of same-day "transitions" span removed segments (median gap 15 bars,
  p90 60) + 341 cross-day pairs. Honest contiguous-only recount: R1->R1 61.4%, NOISE->R1
  43.9% (conclusions unchanged).
- FIREWALL: using transition_matrix.npy as a live Kalman prior ("know the shape before
  the first tick") violates the SEGMENT FIREWALL — membership requires the completed
  segment + whole-day scaler. Label-side use only until a causal nowcast (KT2) passes.

## Constructive next steps
1. Replace the asymmetric band-relative tier with a symmetric, band-independent
   similarity (common normalization, fixed tick tolerance, or beta-space distance);
   or match within volatility_tier strata (field already exists in stage2 records).
2. Require within-bucket MUTUAL coherence (member x member), not root-coverage.
3. Rebuild the Markov chain on the contiguous timeline with PURE_CHAOS as an explicit
   state; always report transition probabilities as LIFT over the destination base rate,
   with day-block bootstrap CIs.
4. Reconcile the segmentation rule and the tier rule (one metric), or exclude the 21.6%
   tier-8-on-self segments from the matching universe.
5. Any live-prior ambition goes through the ROADMAP_LAMBDA_COMPLETION KT2 gate:
   nowcast the CURRENT segment's bucket causally, score oracle-vs-nowcast recovery.
