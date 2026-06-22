# RUN_B2C — Continuous sliding-window F-space, first IS result (2024_02_20)

**Date:** 2026-06-17 · **Day:** 2024_02_20 (Databento IS) · **Builder:** `research/fspace_experiment/build_run_b2c.py`
**Dataset:** `DATA/ATLAS/FEATURES_RUN_B2C/2024_02_20.parquet` (65,097×127) · **Segments:** `artifacts/stage1_RUN_B2C_segments_2024_02_20.json`

## What this is
Continuous sliding-window representation from the 1s base (thesis: window length in seconds = the
cardinal reframe of "timeframe"). Features = custom rolling-L1 + production SFE `compute_L2/L3` at
`N = horizon-seconds` over 5 windows (5s/15s/1m/3m/5m, W = n_base×tf_sec = 45/180/900/2160/2700s).
**Not** Gemini's RUN_B2 (which phase-shifted *tiled* bars onto 1s → the ZOH step-fill problem).
**Scope v1:** L1–L3 only. DEFERRED: L4-NMP, L5-dist, 10m/15m windows (Hurst warmup > 1 day).

## Results (mode-first)
- **Segment length:** mode **30–40s** (302/1033); median 39s (all) / 49s (pristine); mean 55.2/69.1 (ref).
- **Coverage:** **100%** of 56,998 valid rows (vs Gemini tiled runs: 30–45% left uncovered).
- **Pristine:** **73%** (754/1033); tier mix **98% T1** (tightest) / 2% T2; max_residual median **0.44 pts**.
- **Surviving terms:** median **27**, p75 **28** — at/near the cap `min(40, length−2)=28` for 30-bar seeds.

## Verdict: encouraging, NOT conclusive
**Positives:** 100% coverage + 73% pristine on a clean continuous ruler; length comparable to the best
tiled 1s variant (Run C ~54s), well above tiled ZOH (B2 31s).

**Confounds that block an "it wins" claim:**
1. **Parameter saturation** — median 27 terms on 30–85s segments (~0.5 params/sample). The 98%-T1 /
   0.44-pt residual is plausibly OVERFIT, not structural cleanliness. Same cap-saturation Gemini hit on 5s.
2. **L1–L3 only, single day, IS.** No L4/L5, no OOS, no multi-day.
3. **ERROR_BAND_FRACTION artifact** still governs absolute segment length (not a structural fact).
4. **Not a clean A/B** — Gemini's tiled numbers were L1–L4 on 2026 OOS; cross-run length is directional only.

## Next controlled step (one change at a time)
Re-run with a **parameter budget normalized to segment length** (fix params/sample, e.g. cap = length//k)
so fit quality isn't bought with free parameters — AND a **same-day tiled vs continuous A/B** (build a tiled
F-space for 2024_02_20 and run both through identical stage-1). Only then can "continuous explains better"
be tested. Judge on residual/tier at fixed params/sample, effective-N-honest — not on segment length.

## SAME-DAY A/B — continuous (B2C) vs tiled (B2T), 2024_02_20 (the controlled test)
Matched day/TFs/layers/pipeline; ONLY variable = tiled-step-fill vs continuous-sliding (horizons identical).

| metric | B2C continuous | B2T tiled |
|---|---|---|
| pristine fraction | **73%** | **5%** |
| length mode | 30–40s | 500–510s (degenerate; = chaos-hunt ceiling) |
| tier T1 | 98% | 14% |
| pristine residual (median pts) | 0.439 | 0.450 |
| surviving terms (median) | 27 (saturated) | 14 |

**Finding:** 133/140 tiled segments are UNPROCESSED_CHAOS give-up blocks (`max_hunt=500`). Tiled step-fill is
almost entirely un-regress-able; continuous is not. Confirms Gemini's ZOH finding under controlled conditions.

**Skeptical read (likely correct):** the continuous edge is plausibly ARTIFACT, not signal — (1) autocorrelation:
sliding features barely move per second so a line fits trivially (tiny effective-N); (2) param saturation: B2C
used 27 terms vs 14. Residuals are nearly identical (0.44 vs 0.45); only the qualifying FRACTION differs — exactly
what smoothness + more params produce. "More regress-able in-sample" is NECESSARY-NOT-SUFFICIENT for "predicts better."

**Decisive next test:** forward-predictive, param-controlled, effective-N-honest — does a model fit on continuous
features beat tiled at predicting HELD-OUT next-step price, not just fitting the window it sits in?

Artifacts: `artifacts/stage1_RUN_B2{C,T}_segments_2024_02_20.json` · builders `build_run_b2{c,t}.py`.

## Reference — Gemini's prior runs (caveat: 2026 OOS, L1–L4, tiled)
Run A (1s-only) 40.4s · Run C (1s mirrored wall-clock) 53.8s · Run B2 (tiled step-fill) 31s · Run B (5s multi-TF) 164s.

## NULL CONTROLS — Brownian + Fourier (the deflation, 2024_02_20)
Continuous (B2C) pristine run on no-signal nulls matched to the real day (σ=0.78pts/s).

| series | pristine% | med residual | med band | band-normalized (Resid/Band) |
|---|---|---|---|---|
| Real | 73% | 0.439 | 0.575 | 0.99 |
| Brownian (iid walk) | 67% | 0.728 | 0.750 | 1.14 |
| Fourier (phase-randomized, matched spectrum) | 66% | 0.999 | 1.075 | 1.10 |

**Verdict:** pristine% is ~90% ARTIFACT — a random walk scores 67%, matched-spectrum surrogate 66%, vs real 73%.
The 73-vs-5 tiled gap was overwhelmingly REPRESENTATION (sliding-window smoothing), NOT signal. Only a thin
real-over-null edge survives (band-normalized tightness 0.99 real vs 1.10-1.14 nulls, ~10%), and it's in-sample,
one day, partly band-confounded — NOT exploitable evidence.
- ACTIONABLE: continuous >> tiled in regress-ability (real F-space finding).
- NOT ESTABLISHED: a tradeable signal. Next = forward-predictive test (real vs Fourier null), CIs across days,
  measuring the real-MINUS-null edge — never raw pristine%.
Generator: `research/fspace_experiment/make_brownian_null.py` (--null brownian|fourier).

## NEW BREAK RULE (per-bar 10%-of-delta, 1-tick floor, 5-consecutive) — re-run real+nulls
Replaced the lagged prior-range band + spike-sensitive max-cap with: off[t] = |actual-pred| > max(0.10*|Δclose[t]|, 0.25); break on 5 consecutive off.

| series | pristine% | segs | med pristine len |
|---|---|---|---|
| REAL | 69.8% | 1092 | 51s |
| BROWN | 65.5% | 1309 | 49s |
| FOUR | 65.5% | 1361 | 46s |

REAL−null gap = +4.2/+4.3 pp (OLD band gave +6/+7). Gap NARROWED, did not widen.
PREDICTION (Claude's) that the rule would widen separation was WRONG: the threshold is proportional to the
local per-bar move, so a random walk earns a proportionally bigger tolerance — the self-scaling scales up for
noise too, neutralizing the expected separation.

**Conclusion (robust across BOTH band designs):** the in-sample pristine-segment metric CANNOT distinguish real
price from a matched-spectrum/random-walk null (gap ~4-7 pp, ~90-95% artifact). Segment regress-ability ≠ signal.
The new rule is methodologically better (self-scaling, spike-proof, no lag) — KEEP it — but it confirms the same
answer more cleanly. ONLY remaining honest discriminator: forward-predictive test (fit window -> predict next
unseen bar, real vs Fourier null, disjoint data). Artifacts: stage1_B2Cv2_{REAL,BROWN,FOUR}_segments_*.json.

## FULL 2x3 GRID under corrected rule — THE RESULT FLIPS (2024_02_20)
| corrected rule | REAL | BROWN | FOUR | real-Fourier gap |
|---|---|---|---|---|
| continuous (B2C) | 69.8% (1092) | 65.5% (1309) | 65.5% (1361) | +4.3 pp |
| tiled (B2T) | 62.4% (1339) | 52.6% (1122) | 45.7% (470) | +16.7 pp |

**FLIP:** continuous barely beats its null (+4pp = over-smoothing inflates null too); TILED separates real from
the spectrum-matched Fourier null STRONGLY (+16.7pp). Discrete bar-sampling preserves the real-vs-null contrast
that sliding-window smoothing destroys. The old-rule "tiled=5%" was a BAND ARTIFACT (spike-sensitive max-cap on
step-features); the corrected 5-consecutive rule rescued tiled. => earlier "continuous>>tiled" conclusion REVERSED
for signal discrimination; tiling is the better detector, continuous over-smooths.

**GATING CAVEAT:** n=1 Fourier draw — NOT a significance test. NEXT: surrogate ENSEMBLE (30-50 Fourier seeds),
tiled stage-1 on each -> null distribution of pristine%, test if real 62.4% sits significantly above it (p-value).
Then multi-day, then forward-predictive. Artifacts: stage1_B2{C,T}v2_{REAL,BROWN,FOUR}_segments_*.json.
