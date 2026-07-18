# nt8_port/csharp — P1 platform-neutral C# entry engine (task 131)

Native C# port of the 22 top-K stream generators + compact entry logistic, validated
bar-by-bar against the P0 golden vectors. **Platform-neutral** (no NinjaScript refs;
built on .NET 10, only `System.Text.Json` + `System.IO.Compression` from the BCL).
NinjaScript packaging is P2.

## Files
| file | contents |
|---|---|
| `Pandas.cs` | pandas/numpy-exact helpers: `ewm(adjust=False,ignore_na=False)`, rolling mean/std/max/min/sum/median (min_periods), diff/shift, clock-aligned OHLCV `Buckets` (groupby ts//period + close-row / last-closed mapping). |
| `Model.cs` | `Ctx` (DayCtx equiv: zz_thr = 1m ATR14×4 index-bucketed, streaming zigzag, `Emit` features); `MathX` (z21 OLS endpoint z ddof=2, Wilder-14 DMI diff); `TfState` (multi-TF z/vel/acc/wick/vr/volr/dmi + last-closed row map). |
| `Gens.cs` | 21 generators: ZIGZAG, ORB02, ROUND05, VWAP03, DOW19, TUNNEL20, ATR09, PIVOT16, RENKO24, SAR23, RSI06, MACD07, CTXER, EXITKMDR, PTRNENGULF, NMP, NMP9 waterfall (RIDEAGAINST/RIDECALM/FADEAGAINST), NMPT waterfall (FADECALM/MTFBRK). |
| `Tmpl0.cs` | TMPL0 frozen-codebook K-means template stream (6-D features incl. Wilder ADX + R/S Hurst, nearest-centroid over `_tmpl0.json`). |
| `Program.cs` | IO (gz-JSON inputs), consensus (22-stream ±180s), compact logistic P + entry, per-1m-bar aggregation → `out/<day>.json`. |
| `harness_data/` | per-day exported inputs (`<day>.json.gz`), `_model.json` (compact weights/threshold), `_tmpl0.json` (frozen codebook). Produced by `tools/parity_check.py export`. |
| `out/` | per-day C# output consumed by `tools/parity_check.py compare`. |

## Run (repo root; python3.11 — bare python hangs)
```
python3.11 research/nt8_port/tools/parity_check.py export      # rebuild inputs + python reference
cd research/nt8_port/csharp && dotnet run -c Release .          # run the port -> out/<day>.json
cd ../../.. && python3.11 research/nt8_port/tools/parity_check.py compare   # -> reports/p1_parity.md
```
`TMPL_DEBUG=1 dotnet run -c Release .` also dumps per-event `out/<day>.tmpl0.json` (features/tid) for diagnosis.

## Result (see reports/p1_parity.md)
- Fire-state vs golden **99.962%** (≥99.5% bar): **21/22 streams bit-exact**; TMPL0 residual = golden
  pandas-quicksort tie-order on same-minute opposite-direction sub-fires (event-level bit-exact).
- P vs compact reference **2.2e-16** (≤1e-6 bar). Entry **100.000%** (913 entries).
