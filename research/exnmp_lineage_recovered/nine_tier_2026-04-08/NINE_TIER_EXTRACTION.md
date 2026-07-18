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

**CORRECTION (Moises, 2026-07-18): the 9-tier ran WITHOUT the CNN — that was
the whole point.** The CNN branch was optional (`if self.use_cnn` guard); in
CNN-free operation tiers 6-7 (RIDE_MOMENTUM/RIDE_CALM) never fire and the
waterfall falls through to FADE_MOMENTUM/FADE_CALM — an effective **7-tier
physics-only ladder** (CASCADE, KILL_SHOT, FREIGHT, FADE_AGAINST, RIDE_AGAINST,
FADE_MOMENTUM, FADE_CALM). Commit b0deb95b (2026-04-10, "physics OOS baseline +
CNN vs physics comparison") is the physics-only pivot. So the V1-mapping loss:
- RIDE_MOMENTUM / RIDE_CALM — never reachable CNN-free (definitionally lost).
- **FADE_MOMENTUM — reachable CNN-free and STILL lost in the V1 port** (the
  ported list has no FADEMOM; it was absorbed into FADECALM). The one genuinely
  droppable-by-accident tier.

## The CNN-free ladder ran LIVE — and its story founded the exit program
- **2026-04-16 live session** (docs/daily/2026-04-16.md): after fixing the
  frozen-SFE cache bug (live features stale since mid-Feb), the physics tier
  engine caught its first real signal since February: **"$900 peak PnL on a
  single trade... gave most of it back — exits are still weak, but
  architecture works."** Moises live: "we are up to 700" → "OMG its possitive"
  → "peaked at 900 then gaveback evething." (No giveback protection by design
  during parity testing.)
- That $900-peak-full-giveback trade is the founding trauma of the ENTIRE
  exit/capture program — giveback protection, R-trigger analysis, B9, the
  capture-ratio budget, and now the dojos all descend from it.
- **2026-04-17**: per-tier live-aligned table (FADE_CALM 49% WR, +$939 …) —
  the CNN-free ladder's measured performance.
- **2026-04-30**: **`BaseNmpRunner_v1.0-RC.cs`** (recovered alongside, from
  3d765e62) — standalone NATIVE NT8 port of the BASE_NMP tier (the trending-
  regime specialist; $19,997 / $16.7-per-trade Python sim Jan-Mar 2026,
  ~$50/day NT8-equivalent; z_se ROCHE=2.0 entry, |z|<0.5 / vr>1.0 exits).
  This is the "ran it natively on NT8" artifact.

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
