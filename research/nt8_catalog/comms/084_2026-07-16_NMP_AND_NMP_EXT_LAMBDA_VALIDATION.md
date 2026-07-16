# NMP + NMP-EXT — the master equation enters the league; λ term VALIDATED on labels
**Doc:** 084 · **Date:** 2026-07-16 · **Author:** Claude (autonomous, Moises directive
"add to the dossier the NMP and extended NMP") · **Status:** FINAL

## 1. Implementation (canonical, verified pieces only)
- **z source**: `L3_1m_z_se_15` from `FEATURES_1s_v2` — the ONLY store carrying the
  window the verified thresholds live on (the 5s store is a `_30` build; thresholds
  do NOT transfer across window drift — the 21→15 recalibration lesson).
- **Thresholds**: Z_ENTRY=1.8481 / Z_EXIT=0.4752 (recalibration verified 2026-06-11).
- **Episode semantics** (V1): fire at first RTH bar with |z|>Z_ENTRY while armed;
  re-arm when |z|<Z_EXIT.
- **λ̂**: k=21 trailing OLS slope of log(|z_se|+0.1) on the CLOSED-1m sequence
  (one sample per minute at ts%60==0), vectorized but estimator-identical to
  `research/nmp_state/derive.py:120-157`. k=21 = mid of the verified K_SWEEP
  (12,21,30), matching the V1 z_21 window heritage — declared choice.
- **NMP** = V1 behavior: direction always FADE (λ hardcoded 0). **NMP-EXT** = the
  completed equation: λ̂<0 → fade, λ̂≥0 → ride (fire skipped if λ̂ undefined).
- Smoke sanity: ~21.5 fires/day; median |z| at fire 2.06; λ̂ flips 59.6% of fires.

## 2. Results (train 2024 / test 2025+26, day-block CIs; baseline 0.50)
```
NMP      N=10993  OOS-AUC 0.648  base 0.26 || low 0.13 [0.11,0.14] | mid 0.31 | high 0.36
NMP-EXT  N=10793  OOS-AUC 0.574  base 0.54 || low 0.46 [0.44,0.49] | mid 0.56 | high 0.61
```
### What this says
1. **The V1 equation as it ran live is ANTI-ALIGNED: 0.26.** When |z_se| blows out
   past 1.85, the golden labels are riding that move ~74% of the time — the pure
   fade bets against the label. Its low tercile INVERTED = **87% right** on 1,909
   OOS fires (the strongest inverted cell after PIVOT-16). This is the
   label-alignment face of the honest causal reality (V1 loses money live).
2. **The λ term rescues the equation: 0.26 → 0.54 (+28pp).** λ̂ flips 59.6% of
   fires from fade to ride and the agreement crosses to the aligned side. The
   λ-completion thesis — "the equity-burning drawdown = the integral of the
   missing λ term" — now has a direct, out-of-sample, label-level measurement.
3. NMP-EXT's ladder (0.46→0.61) is real but mid-pack; the pooled combiner uses it
   (is_NMPEXT +0.062) while auto-inverting raw NMP (is_NMP −0.170).

## 3. 27-stream combiner refresh (`reports/combiner_preview.md`)
N=469,219 fires (271,060 test): **OOS AUC 0.678**, calibration still on the diagonal
(0.15→0.16 ... 0.74→0.74, decile CIs ±0.01). Tails: bottom decile 0.16 [0.15,0.17]
→ INVERT = **84% right**; top 0.74 [0.73,0.75] → 74% right; ~54k OOS fires in the
two tails. Consensus still inverts at 6+ co-fires (0.47) — crowding = chop.

## 4. Caveats
1. P(label-right) ≠ P($) — economic conversion is the gate before any of this
   feeds the Mamba.
2. λ̂ k=21 is a declared (not swept-and-frozen) choice; the overfit-decay and a
   k-sensitivity check belong to the shelf-life pass.
3. NMP fires cluster at |z| excursions — episodes are not independent within a
   volatility burst; day-block CIs mitigate.
4. Roadmap log updated (docs/Active/ROADMAP_LAMBDA_COMPLETION.md §9, 2026-07-16).

## 5. League/table state
27 graded+low-freq streams in `reports/dossier_signal_league.md`; all
`signal_rows_*.parquet` saved (NMP, NMPEXT included) for the combiner and the
coming economic layer.
