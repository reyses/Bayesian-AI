---
name: distilled-exnmp_lineage_recovered
description: Git-history recovery of the earliest ExNMP (2026-04-05/06 tiered-exit trio → 9-tier ladder) — direct ancestor of today's V1 dossier league; already a fully-written archive.
metadata: {type: distilled, topic: exnmp_lineage_recovered, status: concluded}
---
## Verdict
Recovered 2026-07-17/18 answering Moises's ask for the earliest ExNMP version.
README + `NINE_TIER_EXTRACTION.md` give the full lineage: base NMP engine
(2026-04-04) → the Trio (Killshot/Wick-Overshoot/Cascade, 2026-04-06) →
`nightmare_blended.py` (BlendedEngine) → 9-tier ExNMP ladder + FADE/RIDE/SKIP
CNN head (2026-04-08). Corrected 2026-07-18: the 9-tier ran **CNN-free** in
practice (optional `use_cnn` guard) — an effective 7-tier physics-only ladder.
3 of 9 tiers were lost in the V1 port to today's dossier league (doc 085); one
of those (FADE_MOMENTUM) was reachable CNN-free and dropped by accident, not
by CNN-dependency. Nothing here is live code — kept as documented origin.

## Key numbers (with CIs where they exist)
- 04-04 tree fracture: 29 tradeable + 77 skip branches; TRADE $32,208 vs SKIP
  −$29,426; leaf 150: 944 tr 81% WR $6,730; leaf 149: 341 tr 91%; leaf 101: 62 tr 100%.
- 04-06 KILL_SHOT: 5m_wick>0.83 ∧ 15m_wick>0.77 → 96% win-days IS AND OOS, $16/trade.
  CASCADE amplification: base 486 tr $16 → +1h|z|>1.0 70 tr $19 → +1h|z|>1.5 29 tr $24/trade.
- 04-08 velocity split: winners hold 5× longer (254 vs 52 bars); FADE_MOMENTUM
  $16.7/trade (112 tr) vs FADE_CALM $0.3/trade (8,868 tr) — 50× per-trade gap,
  later erased when V1 absorbed FADEMOM into FADECALM.
- 04-09 stratified 74-day OOS verdict table: FADE_CALM $21,612 OOS (76% WR,
  SOLID); RIDE_AGAINST $18,770 OOS (40% WR, surprise winner); KILL_SHOT $960
  OOS (88% WR); FADE_MOMENTUM $894 OOS (67% WR); CASCADE $276 OOS (85% WR);
  FADE_AGAINST −$5,032 OOS (24% WR, POISON); RIDE_CALM −$352 OOS (weak);
  RIDE_MOMENTUM −$330 OOS (dead).
- `BaseNmpRunner_v1.0-RC.cs` (native NT8 port, recovered): $19,997 / $16.7-per-
  trade Python sim Jan–Mar 2026, ~$50/day NT8-equivalent.
- League cross-check (2026-07): FADE anti-aligned 0.27–0.42 (matches
  FADE_AGAINST poison); RIDEAGN aligned 0.61 (matches RIDE_AGAINST surprise winner).

## Graveyard / never-retry (if any)
- FADE_AGAINST tier: −$14,356 IS / −$5,032 OOS — POISON, confirmed dead again
  by 2026-07 league (FADE family anti-aligned 0.27–0.42).
- RIDE_MOMENTUM: −$235 IS / −$330 OOS — dead, and unreachable without the CNN.
- RIDE_CALM: −$1,760 IS / −$352 OOS — weak, same CNN dependency.

## Reusable assets
None to re-run directly (recovered snapshots, not maintained code). The
RIDE exit-physics vocabulary (velocity-exhausted, vr>1.0 regime shift,
reversion_prob>0.95, wick_ratio>0.60) is flagged in NINE_TIER_EXTRACTION.md as
a conceptual ancestor of today's dojo exit grammar, independently rediscovered.

## Data locations
None owned — pure code/doc recovery, no data artifacts in this folder.

## Open threads
NINE_TIER_EXTRACTION.md proposes reconstituting the 3 lost tiers as
`NMPT-RIDEMOM` / `NMPT-RIDECALM` / `NMPT-FADEMOM` league streams using λ̂>0 as
a stand-in for "CNN says RIDE" — not yet done, flagged as future work for
whoever owns the dossier league next.

## Sources
- research/exnmp_lineage_recovered/README.md
- research/exnmp_lineage_recovered/nine_tier_2026-04-08/NINE_TIER_EXTRACTION.md
- research/exnmp_lineage_recovered/base_nmp_2026-04-04/nightmare.py
- research/exnmp_lineage_recovered/earliest_exnmp_2026-04-06/nightmare_blended.py
- research/exnmp_lineage_recovered/nine_tier_2026-04-08/nightmare_blended_9tier.py
- research/exnmp_lineage_recovered/nine_tier_2026-04-08/BaseNmpRunner_v1.0-RC.cs

## Archive recommendation
This folder IS already an archive (title + README explicitly frame it as a
recovered/superseded lineage). Recommend KEEP under
`research/exnmp_lineage_recovered/` as-is — self-contained with full commit
provenance, nothing left to organize. Fold under a unified `research/archive/`
only as part of a project-wide sweep (alongside exit_lineage_recovered and
misc_archive), not in isolation.
