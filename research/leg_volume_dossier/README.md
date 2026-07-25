# Leg-Volume Dossier

Owner-driven research thread (2026-07-25): volume behavior across winning
legs in leg-native time, and whether normalization windows aligned to leg
scale sharpen the signal.

## How to run (from repo root)
- `python research/leg_volume_dossier/tools/window_scale_test.py` — volume
  z across leg phase under window scales {30-bar, 2×leg, leg, half-leg,
  LEG-PURE (expanding, this-leg-only baseline)}. Raw ATLAS 1m volume.
- `python research/leg_volume_dossier/tools/climax_cohort.py` — split legs
  by presence of a leg-pure mid-leg climax (z≥2, phase 0.25–0.75); compare
  topping behavior.

Companion (dojo-side, packet features): `research/dojo_forge/tools/
fspace_gt_volume.py` (clock-aligned, superseded), `fspace_gt_volume_leg.py`
(leg-phase), `plot_leg_volume.py` (phone chart).

## Findings so far (94 episodes with usable legs; NO CIs yet — directional)
1. **Leg-phase beats clock-phase**: median leg 6-7 min; ±5-min clock windows
   smeared ~66% of a leg per side (owner's critique, confirmed).
2. **Packet-feature view (within-episode z)**: volume climax at MID-LEG
   (phase 0.4–0.5, vol_accel +0.32z), quiet exhaustion at the top
   (velocity −0.30z on silent volume), reversal thrust at 1.25 legs.
3. **Window-scale test (owner's hypothesis)**: leg-aligned/half-leg windows
   do NOT raise the robust (median/trimmed) z — but the MEAN explodes
   (+1.68 leg-pure): the climax is a MINORITY-of-legs event with huge
   leg-relative spikes, not a universal level shift. The signal is
   CONCENTRATED, not smoothed-small. Robust signal lives in volume
   DYNAMICS (velocity/accel), not raw levels.
4. **Climax cohort (n=31 vs 63)**: the leg-pure climax LEADS the peak by
   ~3-4 min (median). INVERSION vs intuition: climax legs give back LESS
   after their top (med 19 pts at +0.5 leg) than quiet-grind legs (med 30) —
   the dangerous tops are the QUIET ones.

## Data
Raw bars: `DATA/ATLAS/1m/<YYYY_MM_DD>.parquet` (repo-root, gitignored).
Episodes: dojo packets (`research/dojo_forge/reports/gen0/packets/`).

## Next
Day-block CIs on cohort deltas; volume-dynamics (velocity/accel) under
leg-pure normalization; climax-as-feature for the exit head (phase-of-climax,
effort-since-climax); extend beyond the 25-day dojo set.

## SYNTHESIS (2026-07-25, "bring it all together")
The composite gauge — VIGOR (conviction fade) × SICKNESS (dynamics anomaly
count) — is the dossier's product. Money contrast: [FADED & sick≥2] vs
[ALIVE & clean] = **−7.96 pts / 3 bars, 95% CI [−13.86, −2.64], SIGNIFICANT**
(4497 frame-obs, 25 days). Hold while alive-and-clean (+5.0); the gauge arms
as vigor fades; 2+ sickness = the tape has turned. All screening → fresh-day
confirmation is the ship-gate for: exit-head features, KNOWLEDGE_PACK v2
lines, control-plane strike inputs. Tool: tools/composite_gauge.py.
