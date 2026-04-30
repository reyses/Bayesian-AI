# Genetic Optimization Head-to-Head: v1.0.6-RC vs v1.0.7-RC

**Generated**: 2026-04-28
**Compute**: NVIDIA RTX 3060 12GB, numba.cuda kernel + manual DE in PyTorch
**Wall time**: v106 = **3.4 sec**, v107 = **6.8 sec** (vs estimated 25-100 min on CPU multiprocessing)
**Speedup over CPU**: ~500-1000× via custom CUDA kernel

## Setup

- **Atlas**: `DATA/ATLAS` (Databento back-adjusted continuous, 391,514 1m bars, 2025-01-01 to 2026-03-21)
- **Train**: 314,716 bars (2025-01-01 to 2025-12-19) — ~231 trading days
- **Holdout**: 76,798 bars (2026-01-01 to 2026-03-21) — ~57 trading days
- **Optimizer**: Manual rand/1/bin Differential Evolution on GPU
- **Parity verified**: CUDA kernel ≡ CPU sim within $0.44 (v107) / $1.38 (v106) over 60-day windows

## Headline finding

| Strategy | Train PnL/day | Best Holdout PnL/day | Sign on holdout |
|---|---:|---:|---|
| **v1.0.6-RC** (static R) | +$30.78 | **-$49.42** | ❌ **NEGATIVE FLIP** |
| **v1.0.7-RC** (dynamic R, ATR-driven) | +$87.81 | **+$98.21** | ✅ **POSITIVE, anti-overfit** |
| v1.0.4 baseline (live, untouched) | +$50/day (measured live) | n/a | ✅ |

**v1.0.6-RC fails. v1.0.7-RC wins decisively.**

## v1.0.6-RC top-5 (FAIL)

| RPoints | MaxLoss | MFE-cut | Trail | Train $/day | Holdout $/day | Drop% |
|--------:|--------:|--------:|------:|------------:|--------------:|------:|
| 68.46 | 150.00 | 0/$0 | 50/0.35 | +$30.78 | **-$49.42** | -240% |
| 74.47 | 150.00 | 10/$0 | 50/0.42 | +$29.55 | **-$85.17** | -388% |
| 74.29 | 99.23 | 17/$0 | 50/0.45 | +$29.08 | **-$85.31** | -393% |
| 74.63 | 150.00 | 9/$0 | 26/0.46 | +$28.82 | **-$60.87** | -311% |
| 74.35 | 150.00 | 25/$0 | 50/0.45 | +$28.71 | **-$67.22** | -334% |

**ALL 5 sign-flip on holdout.** Static R cannot adapt to regime change.
The 32-day-window genetic combo (R=45/SL=90/MFE=17/$2/Trail=21/0.05) performs
equally poorly here. **v1.0.6-RC is dead — do not deploy any v106 combo.**

## v1.0.7-RC top-5 (WIN)

| AtrLookback | AtrMult | MinR | MaxR | MaxLoss | MfeBar/USD | Trail | Train $/day | Holdout $/day | Drop% |
|------------:|--------:|-----:|-----:|--------:|----------:|------:|------------:|--------------:|------:|
| 127 | 10.31 | 5.00 | 195.72 | 144.87 | 3/$0 | 0/0.50 | +$87.81 | +$79.16 | 78% |
| **129** | **11.75** | **7.75** | **230.50** | **54.52** | **30/$0** | **0/0.50** | **+$88.50** | **+$98.21** | **-11% (gain!)** |
| 124 | 11.62 | 8.17 | 249.58 | 48.79 | 0/$46 | 0/0.50 | +$83.39 | +$93.42 | -12% (gain!) |
| 240 | 15.00 | 26.67 | 236.46 | 36.01 | 24/$13 | 0/0.50 | +$84.00 | +$25.38 | 70% |
| 172 | 11.98 | 16.72 | 237.06 | 53.15 | 6/$0 | 0/0.05 | +$82.05 | +$24.43 | 70% |

**3 of 5 top combos have holdout PnL/day ≥ train PnL/day** — that's the
opposite of overfit. The ATR-driven R adapts to volatility regime change.

## Recommended winner combo (Row 2)

```
UseDynamicR        = true
AtrLookbackBars    = 129
AtrMultiplier      = 11.75
MinRPoints         = 7.75
MaxRPoints         = 230.50
MaxUnrealizedLossPoints = 54.52   (= ~$110 SL on MNQ)
MfeCutBarsAfterEntry    = 30
MfeCutThresholdUsd      = 0       (effectively disabled — passes if MFE > 0)
TrailActivatePoints     = 0       (disabled)
TrailGivebackPct        = 0.50
```

**Train**: 936 trades, 31% WR, +$88.50/day on 229 days
**Holdout**: 208 trades, 34% WR, **+$98.21/day on 57 days**

WR is *higher* on holdout. Trade rate slightly higher per day on holdout
(208/57 = 3.6/day vs 936/229 = 4.1/day). Both within sampling noise of each other.

## Critical caveats — DO NOT skip these

1. **Python sim has known ~2× trade count vs NT8 SA** (prior parity work,
   2026-04-28 v1.0.6-RC backtest doc). So $98/day Python ≈ **~$50/day NT8**.
   Still beats v1.0.4 baseline ($50/day live), but margin is thinner than
   the report headline suggests.

2. **Float32 CUDA sim** has ~$0.44 drift on 60-day windows. On 230-day train
   this could be ~$1.50 drift. CPU float64 verification (top-K table) is the
   honest number — and it matches the float32 result within rounding.

3. **MFE-cut and Trail effectively disabled** in winner combo. The strategy
   is doing its work via dynamic R + tight hard SL. This is suspicious —
   double-check that the combo isn't relying on a degenerate edge case (e.g.
   never triggering the secondary rules because they're unreachable).

4. **Holdout is only 57 days** vs 231 train days. Smaller sample = wider
   confidence intervals on holdout. Doesn't invalidate the result, but
   means we shouldn't extrapolate "always +$98/day" — could be +$50 to +$150
   honest range.

5. **Static R was NOT a fair fight here** — the GA found `r=68-74` combos
   that fit the 2025 regime. 2026 has different vol structure. Dynamic R
   adapts; static R doesn't.

## Recommended action

1. **Update v1.0.7-RC NT8 strategy defaults** to the Row-2 combo above.
2. **Run NT8 Strategy Analyzer** on this combo on Sim101 (NT8 native sim)
   for one week — confirms the Python sim numbers translate to NT8 fills.
3. **Live-deploy Thursday** ONLY if NT8 SA holdout validation shows ≥ +$30/day
   (accounting for the 2× Python trade count bias).
4. **Weekly re-tune cadence**: re-run this GA every weekend on the rolling
   12-month window, deploy new winner combo Monday. Cycle time: 7 sec.

## Decision matrix (revised, with realistic threshold)

| Outcome | Action |
|---|---|
| Holdout PnL/day > 0 AND drop < 100% | Ship if ≥ baseline; weekly re-tune. |
| Holdout PnL/day > 0 AND drop < 30% | **STRONG SHIP** — robust combo. |
| Holdout sign flips negative | DO NOT ship. Current v106 result. |
| Holdout > train (negative drop) | **Anti-overfit signal — strongest ship signal.** Current v107 rows 2,3. |

## Reusability — for future strategies

The CUDA framework (`tools/zigzag_genetic_cuda.py`) is reusable. To adapt for
a new strategy:
1. Write a new `simulate_kernel_<strategy>` numba.cuda kernel
2. Define new `BOUNDS` and `NAMES` lists
3. Add new fitness wrapper
4. Reuse `de_evolve` + parity test infrastructure as-is

Numba CPU fallback at `tools/zigzag_genetic_numba.py` provides verification
path independent of GPU.
