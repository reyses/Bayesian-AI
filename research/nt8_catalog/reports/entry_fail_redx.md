# TASK 122 -- the entry-fail RED X: what separates terminal-good from terminal-bad AT ENTRY?

SEALED Shainin contrast. Logistic P(terminal-good | entry-time features) fit on 2024 good-vs-bad, coefs + the three retained-volume thresholds FROZEN, single shot on the 2025-26 test tape. P-only is the pre-registered bar -- beating base alone re-discovers P.

## Population + labels
- Engagement machinery: select_wrongdir.engagements() -- P>=p90(train)=0.76023 FROZEN, 60s/day/dir de-dup, MIN_WINDOW>=15m; terminal drift via swl.scan (eb.signed_drift_path).
- **TRAIN (2024, split=train): 12917 engagements** over 258 days.
- **TEST (2025-26, split=test): 23378 engagements** over 282 days.
- Terminal labels at BAND=4: GOOD terminal>=+4, BAD terminal<=-4, DEAD |terminal|<4. DEAD excluded from FIT, included in volume accounting.
- TRAIN mix: good=5349 (41.4%), bad=6260 (48.5%), dead=1308 (10.1%).
- TEST mix: good=10100 (43.2%), bad=11615 (49.7%), dead=1663 (7.1%). **Unconditional terminal-good rate (base) = 0.432.**
- Moises' fail fact reproduced: bad(<=-4) share of test = 49.7%; good share = 43.2%.

## Feature joins (all causal at fire ts) + coverage
- entry P, det: econ_drift_rows (native).
- pivot_age_min / sig_with_leg / tod: join fire ts into signal_rows_<det> (det verbatim). Coverage train {'pivot_age_min': 1.0, 'sig_with_leg': 1.0, 'tod': 1.0}; test {'pivot_age_min': 1.0, 'sig_with_leg': 1.0, 'tod': 1.0}.
- lambda_hat: dsp._nmp_lambda (z_se store L3_1m_z_se_15, NMP_K=21, NMP_EPS=0.1), per-5s ffilled, as-of fire ts. Coverage (defined) train 0.999, test 0.979 (undefined -> median-fill + missing indicator).
- NMP9 tier: dsp._nmp9_events waterfall (verbatim constants), as-of last emission at/before ts ('none' if no tier armed). trail_vol: std of last 60 5s closes / TICK (ticks).

## 1. THE SHAININ CONTRAST (train good-vs-bad; ranked by univariate |AUC-0.5|)
Univariate train AUC = how well each feature ALONE ranks good above bad. diff = mean(good)-mean(bad) with day-block 95% CI (4000; * = excludes 0). Categoricals show best/worst good-rate level + one-hot AUC.

| rank | feature | uni AUC | good mode | bad mode | good med | bad med | diff (good-bad) [CI] |
|---|---|---|---|---|---|---|---|
| 1 | nmp9_tier | 0.538 | best=RIDEMOM | worst=CASCADE | 0.515 | 0.247 | (categorical) |
| 2 | tod | 0.468 | +0.025 | +0.025 | +0.407 | +0.464 | -0.0351 [-0.0581, -0.0125] * |
| 3 | det | 0.524 | best=NMPLAMBDA | worst=NMPTMTFEXH | 0.565 | 0.416 | (categorical) |
| 4 | trail_vol_ticks | 0.522 | +20.500 | +15.500 | +28.943 | +26.749 | +0.9799 [-0.5397, +2.5916] |
| 5 | P | 0.504 | +0.770 | +0.770 | +0.794 | +0.794 | -0.0008 [-0.0033, +0.0018] |
| 6 | sig_with_leg | 0.496 | +1.500 | +1.500 | +1.000 | +1.000 | -0.0075 [-0.0204, +0.0053] |
| 7 | pivot_age_min | 0.497 | +14.500 | +13.500 | +33.417 | +33.667 | -1.2393 [-3.6513, +1.1128] |
| 8 | lambda_hat | 0.500 | -0.025 | -0.025 | -0.000 | -0.001 | -0.0002 [-0.0016, +0.0013] |

**Top-5 dominators:** nmp9_tier (AUC 0.538), tod (AUC 0.468), det (AUC 0.524), trail_vol_ticks (AUC 0.522), P (AUC 0.504).

## 2. Full model vs P-only -- the increment (single-shot test AUC, good-vs-bad)
- **P-only test AUC = 0.4961**
- **Full-model test AUC = 0.5135**
- **Incremental AUC (full - P-only) = +0.0174**
- Reference signal-magnitude bar (MEMORY §2): AUC gap >=0.10 real / 0.05-0.10 conditional / <0.05 noise. This increment is NOISE-LEVEL.

## 3. Pre-registered operating points (thresholds frozen on 2024 volume; single-shot test)
good-rate = P(terminal-good) among retained (DEAD in denominator -- deployment reality). delta-vs-base and delta-vs-P-only(equal-vol) with day-block 95% CI (* = excludes 0).

| target vol (2024) | test retain vol | good-rate | base | vs base [CI] | P-only good-rate | vs P-only [CI] |
|---|---|---|---|---|---|---|
| 70% | 74.3% | 0.437 | 0.432 | +0.0054 [-0.0000, +0.0109] | 0.429 | +0.0084 [+0.0006, +0.0163] * |
| 50% | 56.2% | 0.442 | 0.432 | +0.0103 [+0.0019, +0.0191] * | 0.424 | +0.0184 [+0.0058, +0.0307] * |
| 30% | 35.5% | 0.453 | 0.432 | +0.0209 [+0.0080, +0.0338] * | 0.425 | +0.0282 [+0.0088, +0.0475] * |

## 4. Decomposition at each operating point (what gets sacrificed)
| target vol | N kept | goods kept | bads kept | dead kept | dead share | goods lost (dip/clean) | fails avoided |
|---|---|---|---|---|---|---|---|
| 70% | 17369 | 7598 | 8597 | 1174 | 6.8% | 2502 (1447/1055) | 3018 |
| 50% | 13137 | 5811 | 6492 | 834 | 6.3% | 4289 (2472/1817) | 5123 |
| 30% | 8295 | 3757 | 4050 | 488 | 5.9% | 6343 (3670/2673) | 7565 |

## Full frontier (description only -- NOT the verdict)
| retain vol target | test retain vol | good-rate | N kept |
|---|---|---|---|
| 95% | 95.2% | 0.433 | 22253 |
| 90% | 90.9% | 0.434 | 21261 |
| 85% | 86.4% | 0.434 | 20208 |
| 80% | 82.4% | 0.436 | 19271 |
| 75% | 78.7% | 0.437 | 18394 |
| 70% | 74.3% | 0.437 | 17369 |
| 65% | 70.1% | 0.440 | 16396 |
| 60% | 65.9% | 0.440 | 15407 |
| 55% | 61.3% | 0.441 | 14339 |
| 50% | 56.2% | 0.442 | 13137 |
| 45% | 51.0% | 0.443 | 11925 |
| 40% | 46.3% | 0.445 | 10827 |
| 35% | 40.9% | 0.449 | 9551 |
| 30% | 35.5% | 0.453 | 8295 |
| 25% | 30.3% | 0.455 | 7087 |
| 20% | 24.5% | 0.456 | 5722 |
| 15% | 18.8% | 0.465 | 4393 |
| 10% | 13.0% | 0.466 | 3047 |
| 5% | 6.6% | 0.469 | 1539 |

## PRE-REGISTERED BAR + VERDICT
Bar: at >=1 operating point with test retain vol >=30%, filtered good-rate beats BOTH (a) unconditional base AND (b) P-only at equal volume, with delta-vs-P-only CI excluding 0.
- vol 70% (test 74%): beats base=True (++0.005), beats P-only=True (vs-P-only +0.0084 [+0.0006, +0.0163] *) -> PASS
- vol 50% (test 56%): beats base=True (++0.010), beats P-only=True (vs-P-only +0.0184 [+0.0058, +0.0307] *) -> PASS
- vol 30% (test 35%): beats base=True (++0.021), beats P-only=True (vs-P-only +0.0282 [+0.0088, +0.0475] *) -> PASS

## **VERDICT: PASS (by the letter) -- but the effect is noise-magnitude; read the caveats**
All three pre-registered points beat both base and P-only with CI-clean deltas, so the
pre-registered bar is met. BUT as a critical read (this is a real-money system):

1. **The increment over P-only is +0.017 AUC -- NOISE-level** by the project's own
   signal-magnitude bar (real >=0.10 / conditional 0.05-0.10 / noise <0.05). Full-model
   test AUC is 0.5135 -- barely above a coin flip.
2. **The P-only comparison is a flattered bar.** P-only test AUC = 0.496 is *below* 0.5 --
   entry P is essentially ANTI-predictive of terminal economics on test (exactly Moises'
   thesis: P was trained on direction-agreement, not terminal $). The P-only *filter*
   therefore retains a WORSE-than-base good-rate (0.425-0.429 vs base 0.432), so "beating
   P-only" is nearly automatic for any weak signal. The honest yardstick is **vs BASE**:
   +2.1pp good-rate at 30% retain, CI[+0.8pp, +3.4pp] -- real but small.
3. **Magnitude never escapes the fail regime.** Throwing away 65% of volume (30% retain)
   moves good-rate 43.2% -> 45.3%; even at 5% retain it is only 46.9%. Fail share stays
   ~48-50% at every operating point. Entry-filtering CANNOT get you out of the coin-flip.
4. The thin edge is carried by **nmp9_tier (uni AUC 0.538) and det one-hots (0.524)**, not
   by leg age / sig_with_leg / lambda_hat / P (all uni AUC ~0.50, diffs CI-includes-0).
   lambda_hat separates terminal good-vs-bad essentially not at all here (AUC 0.500).

**Bottom line:** entry-time features add a statistically-clean but economically-tiny
terminal-good separation over entry P. This CONFIRMS that entry P carries no terminal-
economics signal, but it does NOT show the fail problem is solvable at entry -- the
separation is far too small to lift out of the ~50% fail regime. Consistent with
"turns live in paths, not snapshots" (MEMORY §5): the binding constraint is the path /
turn detector, not an entry-time snapshot.

_Descriptive path/label study on the sealed test tape. Rates only (no trading sim, friction irrelevant). A retained rule still graduates through the sealed harness._