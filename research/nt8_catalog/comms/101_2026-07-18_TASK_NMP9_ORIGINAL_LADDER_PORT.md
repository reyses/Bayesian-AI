# TASK 101 — NMP9: the ORIGINAL 9-tier ladder, ported plain (no V1 names)
**Doc:** 101 · **Date:** 2026-07-18 · **Author:** Claude (reviewer) · **Status:** TASK (Opus drone)
Moises: "let's make the good 9 tier mapped without the extra spicy V1 names."
Port the ORIGINAL 2026-04-08 waterfall (recovered at
`research/exnmp_lineage_recovered/nine_tier_2026-04-08/nightmare_blended_9tier.py`
— THE source of truth for conditions/constants, verbatim) into the dossier
league as 9 streams under their ORIGINAL names. Do NOT include the V1-era
additions (MTFEXH/MTFBRK) and do NOT reuse the NMPT-* re-derivations.

## Streams (exact names)
NMP9-CASCADE, NMP9-KILLSHOT, NMP9-FREIGHT, NMP9-FADEAGAINST, NMP9-RIDEAGAINST,
NMP9-RIDEMOM, NMP9-RIDECALM, NMP9-FADEMOM, NMP9-FADECALM.

## Waterfall (from the recovered file — port VERBATIM, priority order)
Entry universe: NMP base (|z21|>ROCHE=2.0 ∧ vr<VR_ENTRY=1.0) at 1m boundaries;
default direction = fade the z. Then:
1. CASCADE:      wick_5m>WICK_5M_MIN ∧ wick_15m>WICK_15M_MIN ∧ |1h_z|≥1.0 aligned
2. KILLSHOT:     wick pair, no 1h alignment
3. FREIGHT:      |1m_vel| ≥ 100 → direction = sign(vel) (ride)
4. FADEAGAINST:  |1h_z| ≥ 1.5 against the fade → direction = follow 1h z
5. RIDEAGAINST:  |1h_vel| ≥ 1.5 against the fade → direction = follow 1h vel
   -- the head seat (originally CNN FADE/RIDE/SKIP; ran CNN-free after 04-10) --
6. RIDEMOM:      HEAD says RIDE ∧ |vel| ≥ 50 → direction flipped (with momentum)
7. RIDECALM:     HEAD says RIDE ∧ |vel| < 50 → direction flipped
8. FADEMOM:      |vel| ≥ 50 → fade
9. FADECALM:     default → fade
Constants verbatim from the recovered file (incl. the WICK_*_MIN values — read
them there; do not guess). Velocity in the file's units — reconcile with the
NMPT port's ticks convention (`_tf_state` vel is ticks; the original 79D
velocity was also ticks-based; VERIFY against the file and document).

## The head for tiers 6-7 (labeled as COMPLETION, not original)
λ̂ (trailing OLS slope of log(|z_se|+0.1), k=21 — the exact NMP-LAMBDA
derivation already in the pipeline): λ̂>0 → RIDE, else fall through to 8/9.
No SKIP (the CNN had one; λ̂ doesn't — document the omission). Tag these two
streams' league doc-lines "λ̂-completed (head seat)".

## Build rules
- Append-only to `research/nt8_catalog/tools/dossier_signal_pipeline.py`
  (same pattern as the NMPT block: reuse `_tf_state` clock-aligned buckets,
  `_z21`, the λ̂ machinery from NMP-LAMBDA; edge-trigger fires on (tier,
  direction) change at 1m boundaries — same adaptation as NMPT, documented).
- One waterfall evaluation shared by all 9 generators (compute once per ctx,
  cache like `ctx._nmpt`); each stream emits only its own tier's fires.
- Standard harness: shared features, train 2024 / test 2025+26, day-block CIs,
  save signal_rows_NMP9<TIER>.parquet BEFORE gating.
- RUN the league for the 9 streams (this is CPU pandas — run it yourself,
  SYNCHRONOUSLY) + rebuild the combiner preview WITH the 9 added (report
  delta vs the current 0.689 AUC).
- Deliverables: league table for the 9 (AUC, base, terciles, N, fires/day,
  CIs) + a comparison table vs the corresponding NMPT-* results where a
  counterpart exists (CASCADE↔CASCADE, KILLSHOT↔KILLSHOT, FREIGHT↔FREIGHT,
  FADEAGAINST↔FADEAGN, RIDEAGAINST↔RIDEAGN, FADECALM↔FADECALM; FADEMOM/
  RIDEMOM/RIDECALM have NO counterpart = the recovered tiers) + report
  `research/nt8_catalog/reports/nmp9_ladder.md`.
- The interesting questions the report must answer: (a) does FADEMOM separate
  from FADECALM (it was absorbed in V1 — was that a real loss?); (b) do the
  λ̂-completed RIDE tiers land in the aligned family like the rest of the
  ride side; (c) does any of the 9 add combiner lift?
- Commit NOTHING. Reviewer verifies (reproduce one stream's numbers from its
  rows parquet) then commits.

## Ladder discipline
RUN SYNCHRONOUSLY (never background-and-stop). python3.11 from repo root
(bare `python` hangs). Claim=evidence; skip-rather-than-fabricate. NOTE: a
Sonnet fleet is running concurrently (wrong-dir dojo) — do not touch
research/exit_dojo/ or its gate_state; your work is confined to the
nt8_catalog pipeline + reports.
