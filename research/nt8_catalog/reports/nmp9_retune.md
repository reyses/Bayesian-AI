# NMP9 — quantile-match RETUNE of the distribution-dependent thresholds

**Doc 102 · 2026-07-18 · Opus drone (reviewer = Fable).** The verbatim 2026-04-08 nine-tier
ladder (doc 101) was ported with the *original* constants (ROCHE 2.0, wick 0.83/0.77,
H1 1.0/1.5). But the rolling windows changed since 2026-04 → the `z21` / `wick` / `h1z`
**distributions** shifted, so those thresholds sit on the *wrong quantiles* now (symptom:
NMP9-CASCADE fired only 71×/2y). This retune re-solves the six distribution-dependent
thresholds by **quantile-matching on 2024 marginal pass-rates** to the era occupancy anchors
(the validated method — Z_ENTRY 2.0→1.8481, 2026-06-11). **No AI-label and no PnL touch the
tuning loop** (the quantile-cell-overfit trap, MEMORY §3). HELD absolute: `vr<1.0` (regime
boundary), velocity 50/100 (ticks, verbatim formula).

Tools: `nmp9_probe_2024.py` (dumps every 2024 RTH 1m boundary's raw estimators, no gate) →
`nmp9_quantile_match.py` (solves + freezes `reports/nmp9_retuned_constants.json`) →
`nmp9_retune_run.py` (retuned league + combiner delta, restores the verbatim parquets after).
The verbatim run stays byte-reproducible: `dossier_signal_pipeline.NMP9_USE_RETUNED=False`
by default; the runner flips it and a **separate constants dict** `_TIER9_C_RETUNED` is used —
`_TIER9_C` (verbatim) is untouched.

---

## 1. Constants — old → new, with the anchor used per threshold

Solved on 2024 (258 label days, 102,814 RTH 1m boundaries, 398.5/day). Each RETUNE threshold
is set to the value on the **current** estimator whose 2024 marginal pass-rate reproduces the
era occupancy target, solved in **waterfall order** (base → wick → 1h). Bisection, monotone.

| constant | verbatim | retuned | era anchor (source) | 2024 pass-rate before → after |
|---|---:|---:|---|---|
| **ROCHE** (base \|z21\|) | 2.0 | **1.9131** | entry-universe ≈ 9,277/277d ≈ **33.5 bnd/day** (jrnl 04-08) | 27.1 → 33.5 /day |
| **WICK_5M_MIN** | 0.83 | **0.7475** | has_wick (KILL_SHOT+CASCADE) ≈ **2.5/day** (jrnl 04-06) | 1.17 → 2.50 /day |
| **WICK_15M_MIN** | 0.77 | **0.6875** | ″ (single joint anchor; additive shift −0.0825 keeps the era 0.06 gap) | ″ |
| **H1_Z_MIN** (cascade split) | 1.0 | **1.3835** | cascade/has_wick ratio ≈ **70/486 = 0.144** (jrnl 04-06 ladder) | 0.225 → 0.143 |
| **H1_AGAINST_Z_MIN** | 1.5 | **2.213** | \|h1z\|>1.5 aligned-tail ≈ **29/486 = 0.0597** (jrnl 04-06 ladder) | 0.123 → 0.061 |
| VR_ENTRY | 1.0 | 1.0 | HELD — regime boundary (semantically absolute) | — |
| VELOCITY_THRESHOLD | 50 | 50 | HELD — ticks, verbatim formula | — |
| FREIGHT_TRAIN_THRESHOLD | 100 | 100 | HELD — ticks, verbatim formula | — |

**Direction of each shift (why):**
- **ROCHE ↓ slightly** — `z21`'s tail is marginally heavier than the era's `z_se`; 2.0 selected
  only 27/day vs the era's ~33, so it drops to 1.913 (same order as the 2.0→1.848 z-recal).
- **Wick ↓ hard** — the port's `1−|c−o|/range` doji ratio is far more stringent at 0.83/0.77
  than the era's 79D `wick_ratio`: has_wick ran **3× too thin** (1.17 vs 2.5/day). This is the
  real CASCADE/KILLSHOT-starvation driver. Shift both down 0.0825 (era gap preserved).
- **H1_Z_MIN ↑ and H1_AGAINST ↑** — the port's `h1z` (z21 on 1h buckets) runs *larger* than the
  era's 1h `z_se`, so 1.0 over-selected aligned (22.5% of has_wick vs era 14.4%). Faithful match
  raises the bar to 1.38 / 2.21 to restore the era tail fractions.

> **Documented anchor caveat.** The base anchor (33/day, jrnl 04-08 phase-1) and the wick anchor
> (2.5/day, jrnl 04-06) come from *different* era snapshots and imply slightly inconsistent
> has_wick/universe ratios (7.6% vs the 04-08 phase-1's 3.2%). Each is the best available anchor
> for *its* threshold and is matched independently, per the spec's per-threshold instruction.
> `H1_AGAINST_Z_MIN` is calibrated on the `h1z` aligned-tail (symmetric) and, verbatim to the
> era's single constant, is *also* applied to the `h1_vel` AGAINST gate — negligible effect there
> (\|h1vel\| median ≈ 11 ticks ≫ 2.2, so the RIDEAGAINST vel gate stays a sign filter either way).
> Matching unit = **raw in-universe boundary rate** (monotone in the threshold, the literal
> reading of "entry-universe rate"); contiguous-run and edge-fire rates were checked as sanity.

---

## 2. League table — BEFORE (verbatim) vs AFTER (retuned)

Same eval path (train 2024 / test 2025+26, day-block bootstrap CIs, baseline 0.50).
`base` = P(fire direction == AI-label direction) — a *direction-agreement* score, **not $/day**.
`N` is total fires (all label days); `fires/day` is per **test** calendar day.

| tier | N before→after | base before→after | base CI (after) | AUC before→after | fires/day (te, after) |
|---|---|---|---|---|---|
| CASCADE | 71 → **109** | 0.17 → **0.19** (raw) | *N<200, thin* | — → — | ~1.2 |
| KILLSHOT | 329 → **874** | 0.172 → **0.178** | [0.144, 0.215] | 0.635 → 0.623 | 2.12 |
| FREIGHT | 1472 → **1560** | 0.854 → **0.854** | [0.831, 0.876] | 0.638 → 0.644 | 4.96 |
| FADEAGAINST | 1133 → **567** | 0.758 → **0.742** | [0.667, 0.818] | 0.547 → 0.532 | 2.09 |
| RIDEAGAINST | 3969 → **4670** | 0.789 → **0.788** | [0.769, 0.806] | 0.641 → 0.654 | 8.34 |
| RIDEMOM † | 1142 → **1413** | 0.810 → **0.814** | [0.785, 0.841] | 0.636 → 0.639 | 3.40 |
| RIDECALM † | 1865 → **2601** | 0.781 → **0.773** | [0.745, 0.799] | 0.603 → 0.597 | 4.58 |
| FADEMOM | 525 → **668** | 0.206 → **0.222** | [0.178, 0.267] | 0.634 → 0.631 | 1.91 |
| FADECALM | 828 → **1171** | 0.289 → **0.295** | [0.260, 0.332] | 0.561 → 0.585 | 2.69 |

† λ̂-completed head (doc 101). **Occupancy moved as designed** — KILLSHOT +166% (wick loosened),
RIDEAGAINST/RIDECALM/FADECALM fatter (wider universe), FADEAGAINST **halved** (H1_AGAINST
1.5→2.21). **But every AFTER base-agreement CI overlaps the verbatim CI, and every AUC moves
≤0.024** (FADECALM 0.561→0.585 is the largest, still deep inside noise for N≈1k). The
aligned/anti-aligned family split is **identical**: ride/against family FREIGHT 0.85, RIDEAGAINST
0.79, RIDEMOM 0.81, RIDECALM 0.77, FADEAGAINST 0.74 (all 0.74–0.85); pure-fade family KILLSHOT
0.18, FADEMOM 0.22, FADECALM 0.30, CASCADE 0.19-raw (all 0.18–0.30, strongly anti-aligned).

---

## 3. Combiner — same-pool delta

Pooled calibrated P(right) over the identical **55-stream** pool (46 non-NMP9 + 9 NMP9), NMP9
parquets swapped verbatim↔retuned, everything else fixed (`nmp9_retune_run.combiner_auc`,
inlined `combiner_preview` logic so `combiner_preview.md` is not clobbered).

| pool | pooled OOS AUC |
|---|---:|
| 46 non-NMP9 + 9 **verbatim** NMP9 (reproduces doc-101 anchor 0.676) | **0.6759** |
| 46 non-NMP9 + 9 **retuned** NMP9 | **0.6765** |
| **same-pool delta** | **+0.0006** |

**The retune moves the pooled combiner by +0.0006 — utterly immaterial** (well inside noise;
the BEFORE recompute 0.6759 reproduces the 0.676 anchor to rounding). The ride-family identity
weights already carried NMP9's marginal contribution; re-centering the occupancies does not add
independent signal (NMP9 overlaps NMPT-RIDEAGN / NMPT-FREIGHT / NMP-LAMBDA, doc 101 §3c).

---

## 4. The three explicit answers

**(a) Is CASCADE un-thinned? — NO, not meaningfully (and this is FAITHFUL, not a bug).**
71 → 109 fires (+54%), but still **N=109 < 200** — still below the stable-AUC floor, still
"too few signals." The wick loosening *added* has_wick fires, but the faithful h1-alignment
quantile-match (H1_Z_MIN 1.0→1.3835, restoring the era 70/486 = 14.4% cascade fraction) *offset*
most of the gain: in the current port \|h1z\|>1.0 over-selects aligned (22.5% vs era 14.4%), so
matching the era **raises** the bar and keeps CASCADE rare. **CASCADE was rare in the era too**
(70/486 of the wick population; jrnl 04-09 "rare, high WR"). Un-thinning it further would mean
*abandoning* the era ratio — which the quantile-match method explicitly refuses. The 71-fire
symptom was real (the wick gate WAS 3× too thin), but the fix is bounded by the era's own
CASCADE:wick ratio.

**(b) Do any tier verdicts change? — NO.** Every tier's AFTER base-agreement CI overlaps its
verbatim CI; every AUC moves ≤0.024 (noise at these N). No tier flips family; no tier changes
reliability class. FADEAGAINST halved in N but its base (0.758→0.742) and AUC (0.547→0.532) are
unchanged within CI — same verdict (aligned, weakly rankable). The label-side signal each tier
carries is unchanged.

**(c) Verdict: STRUCTURE ALREADY CAPTURED; RETUNE IMMATERIAL.** All AUCs sit within the CIs of
the verbatim run and the combiner delta is +0.0006. The retune **correctly re-centers the tier
occupancies** on the era anchors (has_wick 1.17→2.5/day, KILLSHOT ×2.7, FADEAGAINST halved) —
i.e. it fixes the *fires/day* drift the shifted quantiles caused — **without changing how RIGHT
any tier is or lifting the pooled combiner.** The entry-gate + tier definitions capture the same
population structure at either threshold set; the shift changed *how many* fires each tier gets,
not *how aligned* they are. Per the spec, this is a valid and useful outcome: **Step 2 (full
Shainin re-derivation) is not warranted by this result** — the ladder's label-side signal is
threshold-robust, so a re-derivation would be chasing noise. (Reviewer/Moises call.)

---

## 5. Deviations / notes (flagged for review)
1. **Matching unit = raw in-universe boundary rate.** "entry-universe rate ≈ 33/day" is read as
   the *size of the entry universe* (monotone in ROCHE, clean). Contiguous-run (19/day) and
   edge-fire rates were computed as sanity but are non-monotone/circular under the waterfall.
2. **Wick pair solved as a single additive shift** (one joint 2.5/day anchor, two thresholds →
   1 DOF); the era 5m−15m gap (0.06) is preserved. A per-threshold quantile match would need
   separate era marginals, which the journals do not provide.
3. **H1_AGAINST_Z_MIN calibrated on the h1z aligned-tail** and applied verbatim to *both* the
   FADEAGAINST z-gate and the RIDEAGAINST vel-gate (era used one constant for both); the vel-gate
   effect is negligible (see §1 caveat).
4. **Era anchors are from different 2026-04 snapshots** and are mutually slightly inconsistent;
   documented per §1. Absolute anchors (33, 2.5/day) are exit-driven era *trade* counts, hence
   approximate; the ratio anchors (70/486, 29/486) are cleaner.
5. **Reproducibility preserved.** `_TIER9_C` (verbatim) untouched; retuned values live in a
   separate `_TIER9_C_RETUNED` dict gated by `NMP9_USE_RETUNED` (default False); the runner
   restores the verbatim `signal_rows_NMP9*.parquet` byte-for-byte after computing the AFTER
   pool. **Committed: nothing.**
