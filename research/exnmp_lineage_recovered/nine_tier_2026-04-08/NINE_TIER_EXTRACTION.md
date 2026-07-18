# The original 9-tier ExNMP ladder — extracted (2026-07-18)

Source: `nn_v2/nightmare_blended.py` @ commit **06d14190** (2026-04-08, "feat:
9 ExNMP tiers + FADE/RIDE/SKIP CNN"). Full file saved alongside as
`nightmare_blended_9tier.py`. Moises: "in the past we broke the 9 tiers because
we couldn't map it to V1 and we gave up" — this documents exactly what the 9
were, what survived the V1 mapping, and what was lost.

## The 9-tier waterfall (priority order, verbatim conditions)
Entry universe: NMP base (|z21| extreme + vr) at 1m boundaries. Default
direction = fade the z (short if z>0). 79D feature indices in the file.

| # | tier | condition | direction | exit physics |
|---|---|---|---|---|
| 1 | **CASCADE** | 5m wick>min ∧ 15m wick>min ∧ \|1h_z\|≥1.0 aligned | fade z | p_center>0.60 ×3 bars |
| 2 | **KILL_SHOT** | wick rejection, no 1h alignment | fade z | p_center>0.60 ×3 bars |
| 3 | **FREIGHT_TRAIN** | \|1m_vel\| ≥ 100 | RIDE the velocity | ride physics |
| 4 | **FADE_AGAINST** | \|1h_z\| ≥ 1.5 AGAINST the fade | follow the 1h z | fade physics |
| 5 | **RIDE_AGAINST** | \|1h_vel\| ≥ 1.5 opposes fade | follow the 1h vel | ride physics |
| 6 | **RIDE_MOMENTUM** | CNN says RIDE ∧ \|vel\| ≥ 50 | flip (with z) | ride physics |
| 7 | **RIDE_CALM** | CNN says RIDE ∧ \|vel\| < 50 | flip (with z) | ride physics |
| 8 | **FADE_MOMENTUM** | \|vel\| ≥ 50 (CNN didn't fire) | fade z | fade physics |
| 9 | **FADE_CALM** | default | fade z | fade physics |
| — | SKIP | CNN says SKIP | no trade | — |

Constants: VELOCITY_THRESHOLD=50, FREIGHT=100 (ticks-based 1m velocity),
H1_Z_MIN=1.0, H1_AGAINST_Z_MIN=1.5, P_CENTER_EXIT=0.60×3.

## The per-tier EXIT physics (never ported anywhere — see "what was lost")
- **FADE mode** (entered against z): exit when \|z\|<0.5 ×3 bars OR
  p_center>0.60 ×3 bars OR oscillation amplitude < 40% of peak amplitude.
- **RIDE mode** (entered with momentum): exit ×3-bar confirmed on ANY of:
  velocity exhausted (\|vel\|<0.3), vr>1.0 (regime shift against), 
  reversion_prob>0.95 (snap-back pressure), wick_ratio>0.60 (indecision).
- Circuit breakers present but disabled for training (hard stop, giveback).

## Mapping to the ported V1 ladder (doc 085 NMPT-* streams)
Ported list (dossier pipeline): FREIGHT, KILLSHOT, CASCADE, RIDEAGN, FADEAGN,
MTFEXH, MTFBRK, FADECALM.

| original 9 | V1 ported? | note |
|---|---|---|
| CASCADE | ✓ CASCADE | survived |
| KILL_SHOT | ✓ KILLSHOT | survived |
| FREIGHT_TRAIN | ✓ FREIGHT | survived (V1 added acc-agreement + vr<0.85 gates) |
| FADE_AGAINST | ✓ FADEAGN | survived |
| RIDE_AGAINST | ✓ RIDEAGN | survived |
| **RIDE_MOMENTUM** | ✗ LOST | required the CNN FADE/RIDE/SKIP head — unportable without it |
| **RIDE_CALM** | ✗ LOST | same CNN dependency |
| **FADE_MOMENTUM** | ✗ LOST | sat below the CNN in the waterfall; absorbed into FADECALM |
| FADE_CALM | ✓ FADECALM | survived (as the default-fade catch-all) |
| — | + MTFEXH, MTFBRK | ADDED later (blended_engine_2026_04_18, 10 days after) |

**So the break was exactly 3 tiers — the CNN-coupled ones** (RIDE_MOMENTUM,
RIDE_CALM, FADE_MOMENTUM). The CNN flip head was the unmappable dependency:
tiers 6-7 don't EXIST without its prediction, and tier 8 was only reachable
when the CNN declined to fire. The supervised CNN stack was deleted 2026-05-28
(RL pivot), which orphaned them permanently.

## Why this matters NOW (2026-07 lens)
1. **The lost tiers are recoverable without the CNN**: today's calibrated
   combiner P() + λ̂ IS the FADE/RIDE/SKIP head, done rigorously (doc 084:
   λ̂ ride/fade flips 59.6% of fires, 0.26→0.54 alignment). RIDE_MOMENTUM /
   RIDE_CALM / FADE_MOMENTUM could be reconstituted as `NMPT-RIDEMOM` /
   `NMPT-RIDECALM` / `NMPT-FADEMOM` league streams with λ̂>0 standing in for
   "CNN says RIDE" — completing the 9-tier ladder for the first time.
2. **The RIDE exit physics is a 2026-04 ancestor of the dojo exit grammar**:
   velocity-exhausted ≈ ER10 collapse; vr>1.0 regime shift ≈ chop/regime gate;
   reversion_prob high + wick high ≈ against-fires clustering. The blind dojo
   independently rediscovered this exit vocabulary 3 months later.
3. The league already proved the RIDE family is the aligned one (MTFBRK 0.80 /
   MTFEXH 0.79 / FREIGHT 0.75 / RIDEAGN 0.61 vs FADE 0.27-0.42) — the three
   LOST tiers are all in or adjacent to the winning family.

## Provenance
`git show 06d14190:nn_v2/nightmare_blended.py` (2026-04-08). Ported V1
reference: `blended_engine_2026_04_18` per the pipeline docstring; league port
= doc 085. CNN stack deletion = 4b658e2a (2026-05-28).
