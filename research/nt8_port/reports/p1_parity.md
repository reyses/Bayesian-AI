# P1 C# port parity vs golden vectors (task 131)

- dotnet SDK: **available** (build+run harness path)
- fire-state agreement: **100.000%** (178640/178640 cells); bar = >=99.5%
- max |P_csharp - P_compact_ref|: **2.220e-16**; bar = <=1e-6
- entry-decision agreement: **100.000%** (8120/8120); bar = 100%
- compact entry threshold (frozen, 90pct 2024 compact-P) = 0.713983
- **zigzag R-trigger leg parity: 100.000%** (8120/8120 bars); bar = 100%
- **zigzag pivot-confirm parity: 100.000%** (8120/8120 bars); bar = 100%
- max |zz_pivot_age_min diff|: **0.000e+00** min; max |zz_pivot_price diff|: **0.000e+00** pts (bar <=1e-6)

## Per-stream fire-state parity (all days)
| stream | mismatch | total | agree% |
|---|---|---|---|
| RSI06 | 0 | 8120 | 100.000 |
| MACD07 | 0 | 8120 | 100.000 |
| EXITKMDR | 0 | 8120 | 100.000 |
| TMPL0 | 0 | 8120 | 100.000 |
| ZIGZAG | 0 | 8120 | 100.000 |
| ATR09 | 0 | 8120 | 100.000 |
| NMP | 0 | 8120 | 100.000 |
| DOW19 | 0 | 8120 | 100.000 |
| NMP9RIDEAGAINST | 0 | 8120 | 100.000 |
| ROUND05 | 0 | 8120 | 100.000 |
| NMPTFADECALM | 0 | 8120 | 100.000 |
| RENKO24 | 0 | 8120 | 100.000 |
| ORB02 | 0 | 8120 | 100.000 |
| VWAP03 | 0 | 8120 | 100.000 |
| CTXER | 0 | 8120 | 100.000 |
| PIVOT16 | 0 | 8120 | 100.000 |
| SAR23 | 0 | 8120 | 100.000 |
| PTRNENGULF | 0 | 8120 | 100.000 |
| NMP9RIDECALM | 0 | 8120 | 100.000 |
| NMPTMTFBRK | 0 | 8120 | 100.000 |
| TUNNEL20 | 0 | 8120 | 100.000 |
| NMP9FADEAGAINST | 0 | 8120 | 100.000 |

## Verdict vs pre-registered bar
- fire-state >=99.5%: **PASS** (100.000%)
- P within 1e-6: **PASS** (2.22e-16)
- entry 100%: **PASS** (100.000%)

## Disagreement diagnosis
- **21 of 22 streams: 100.000% bit-exact vs golden** (0 mismatched cells across all 20 days), including the heavy-math streams (RSI/MACD EWM, EXIT-KMDR Wilder-ATR, NMP9/NMPT z21+Wilder-DMI+vr ladders, NMP z_se episodes). This proves the shared math (z21 OLS endpoint z, Wilder DMI, pandas-exact EWM/rolling, clock bucketing) is exact.
- **TMPL0: 0 mismatched cells** (0.000%). The P1 residual (67 cells, 99.175%) was same-minute opposite-direction TMPL0 sub-fires (1m + 5m/15m pattern events) whose "last fire wins" order was undefined for same-ts fires (pandas quicksort vs C# stable sort). **P2 (doc 133) PINNED a deterministic tie rule** in BOTH the golden generator and the C# port: highest-TF wins; tie -> larger conviction |long_frac-0.5|; still tied -> hold prior (0). With the ambiguity removed, TMPL0 is now bit-exact and the Python golden/reference self-check is 0/178640.
- **P2 zigzag**: the native R-trigger (ZigzagStrategy port: extreme+-R, min_bars_5s=36, R=max(4,round(ATR14x4/TICK))) is ported into the C# harness and validated against the golden zz_* columns -- leg + pivot-confirm 100.000%, pivot age/price bit-exact (0.0 diff).

## Declared boundaries / deviations (P1)
- **z_se (L3_1m_z_se_15)** = external V2 field-engine feature -> EXPORTED as harness input (NMP / NMP9 head). Native derivation is out of P1 scope.
- **rth / before9 / tod** = America/Chicago session calendar masks -> EXPORTED (pure time functions, native in NT8 from bar time; not signal logic). Eliminates a DST/timezone parity risk.
- **prior_daily** (H/L/C + volume-profile POC/VAH/VAL, 20 days) -> EXPORTED daily context.
- Everything else (zz_thr 1m ATR14x4, DayCtx streaming zigzag, all 22 generators, 22-stream consensus, compact logistic) is COMPUTED natively in C# from the raw 5s OHLCV.
- **Compact model** = top_k_streams.txt (5 base+consensus coefs + 22 one-hots, frozen mu/sd, NO intercept). Consensus computed over the 22 top-K streams only (fit==deploy). Entry threshold = 90th pct of 2024 compact-P over the reference days = 0.713983 (quantile-match on 2024), applied identically to both sides. The golden `P_topk`/`entry` columns are the FULL 56-stream combiner (P2 reference), a different quantity -- not the P1 target.
- **R-trigger zigzag** columns (zz_leg/zz_confirm/zz_pivot_*) are P2 scope (README); carried from golden into the reference, not a P1 C# parity target.

## Per-day parity
| day | fire mismatch | fire cells | entry mismatch | entry bars | P-defined disagree |
|---|---|---|---|---|---|
| 2024_01_30 | 0 | 8932 | 0 | 406 | 0 |
| 2024_03_06 | 0 | 8932 | 0 | 406 | 0 |
| 2024_04_15 | 0 | 8932 | 0 | 406 | 0 |
| 2024_05_21 | 0 | 8932 | 0 | 406 | 0 |
| 2024_06_26 | 0 | 8932 | 0 | 406 | 0 |
| 2024_08_02 | 0 | 8932 | 0 | 406 | 0 |
| 2024_09_09 | 0 | 8932 | 0 | 406 | 0 |
| 2024_10_15 | 0 | 8932 | 0 | 406 | 0 |
| 2024_11_21 | 0 | 8932 | 0 | 406 | 0 |
| 2024_12_30 | 0 | 8932 | 0 | 406 | 0 |
| 2025_01_02 | 0 | 8932 | 0 | 406 | 0 |
| 2025_02_20 | 0 | 8932 | 0 | 406 | 0 |
| 2025_04_14 | 0 | 8932 | 0 | 406 | 0 |
| 2025_05_27 | 0 | 8932 | 0 | 406 | 0 |
| 2025_07_16 | 0 | 8932 | 0 | 406 | 0 |
| 2025_08_26 | 0 | 8932 | 0 | 406 | 0 |
| 2025_10_16 | 0 | 8932 | 0 | 406 | 0 |
| 2025_12_03 | 0 | 8932 | 0 | 406 | 0 |
| 2026_02_03 | 0 | 8932 | 0 | 406 | 0 |
| 2026_03_19 | 0 | 8932 | 0 | 406 | 0 |