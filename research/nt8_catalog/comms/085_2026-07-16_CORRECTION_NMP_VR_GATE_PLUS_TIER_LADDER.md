# CORRECTION (doc 084) + the REAL extended NMP: the V1 tier ladder in the league
**Doc:** 085 · **Date:** 2026-07-16 · **Author:** Claude · **Status:** FINAL
**Corrects:** doc 084 §1 labeling. Trigger: Moises — "the extended is a bunch of
augmented NMP, check the journals."

## 1. Two corrections to my doc 084 (owning them)
1. **My "NMP" ran the bare z-trigger without the vr gate.** The V1 trigger is
   `|z|>Z* AND vr<1.0` (NMP_V2_FEATURE_MAP §3; trap #6: "any NMP claim that doesn't
   reconstruct vr (or λ̂) is not running the NMP trigger"). FIXED: vr = rolling
   std(10)/std(60) ddof=1 on clock-aligned 1m closes (exact V1 formula — thresholds
   transfer, no window drift). Result barely moves: agreement 0.26 → **0.27**,
   N 10,993 → 10,388. The vr gate trims ~5% of fires and adds no alignment —
   consistent with vr being a weak stability stand-in (the dead-proxy finding).
   Doc 084's λ conclusion is UNCHANGED by the correction.
2. **My "NMP-EXT" was mislabeled.** It is the λ-COMPLETE trigger (map §3's
   never-built branch), not what Moises calls extended NMP. Renamed **NMP-LAMBDA**
   (rows parquet renamed; old NMPEXT parquet deleted to avoid combiner
   double-count). Its result stands: 0.54, +28pp over the fade.

## 2. The real "extended NMP" = the V1 tier ladder, now ported
Source: `docs/reference/legacy_tiers/blended_engine_2026_04_18.py::_classify_full_tier`
(:663-770) — verbatim conditions, all V1 quantities recomputed EXACTLY from raw
bars on clock-aligned TF buckets (21-bar OLS endpoint z ddof=2; vr std10/std60;
velocity = 1-bar close delta in TICKS; wick_ratio = 1−|c−o|/range; Wilder-14
dmi_diff; vol_rel = vol/mean30). Original thresholds therefore transfer
(ROCHE 2.0, WICK 0.83/0.77, H1_Z 1.0/1.5, FREIGHT 100 ticks, MTF gates...).
Evaluated at 1m boundaries; **edge-triggered** on (tier, direction) change —
documented adaptation: legacy per-day frequencies arose from position occupancy
(trade management), not signal definition. Excluded: REGIME_FLIP (manual-injection
only), PEAK (disabled in legacy). Parity tell: FADECALM 37.5/day and RIDEAGN
37.6/day vs the legacy docstring's 40/36 per day.

## 3. Results (train 2024 / test 2025+26; baseline 0.50)
```
NMPT-FADECALM N=21034 AUC 0.676 base 0.42 || 0.25 [0.23,0.26] / 0.43 / 0.59
NMPT-RIDEAGN  N=20690 AUC 0.656 base 0.61 || 0.45 / 0.62 / 0.75 [0.73,0.77]
NMPT-FADEAGN  N=  892 AUC 0.638 base 0.41 || 0.24 / 0.50 / 0.50
NMPT-MTFEXH   N=  840 AUC 0.635 base 0.79 || 0.71 / 0.81 / 0.86 [0.80,0.92]
NMPT-MTFBRK   N= 2167 AUC 0.632 base 0.80 || 0.69 / 0.82 / 0.87 [0.83,0.91]
NMPT-FREIGHT  N= 4575 AUC 0.582 base 0.75 || 0.69 / 0.76 / 0.81 [0.79,0.84]
NMPT-KILLSHOT N= 2931 AUC 0.552 base 0.40 || 0.33 / 0.44 / 0.42
NMPT-CASCADE  N=  669 AUC 0.514 base 0.43 || flat (thin)
NMP (fixed)   N=10388 AUC 0.639 base 0.27 || 0.14 [0.12,0.15] / 0.32 / 0.36
NMP-LAMBDA    N=10793 AUC 0.574 base 0.54 || 0.46 / 0.56 / 0.61
```
### The structural read
- **The ladder splits EXACTLY along ride/fade lines.** RIDE tiers are aligned:
  MTFBRK 0.80, MTFEXH 0.79, FREIGHT 0.75, RIDEAGN 0.61. FADE tiers are
  anti-aligned: NMP 0.27, KILLSHOT 0.40, FADEAGN 0.41, FADECALM 0.42. The λ
  finding (doc 084) generalizes: on MNQ 5s vs these labels, ride > fade, always.
- **Independent corroboration**: the legacy engine's own docstring ranked FREIGHT
  (86% WR) and MTF_EXHAUSTION (76% WR) as its best tiers — a P&L-era ranking the
  label-alignment league reproduces from a different measurement.
- **NMPT-FADECALM is the surprise workhorse**: base anti-aligned (0.42) but AUC
  0.676 with a 0.25→0.59 ladder — its low tercile INVERTED = 75% right on 3,749
  OOS fires. The default fade is wrong in a predictable way.
- FREIGHT regime shift: 941 train vs 3,634 test fires — high-velocity minutes
  ~4× more common in 2025-26.
- vr gate verdict: label-alignment unchanged (0.26→0.27) — V1's de-facto
  stability term added nothing the labels care about; λ̂ is the term that matters.

## 4. State
League: 37 streams (`reports/dossier_signal_league.md`). Combiner rerun with the
family included — results in `reports/combiner_preview.md`. Next: economic
conversion (Opus worker spec), overfit-decay (Sonnet worker spec).
