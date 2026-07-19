# P2b — EnsembleRunner v0.2-RC: decision core ported in + parity re-proven

**Date:** 2026-07-18 · **Executor:** Opus build drone · **Status:** DONE (bench-verified; NOT NT8-compiled, NOTHING deployed)

v0.1 was a NinjaScript shell with a stubbed `Decide()` and 14 numbered TODOs. v0.2
ports the **validated decision core in** and re-proves it at **100.000% parity vs the
golden 20 days** through a dedicated shim that compiles the *same source* that ships
in the strategy. Deploy gate honored — nothing written to any NinjaTrader folder.

Deliverables:
- `docs/nt8/7-EnsembleRunner_v0.2-RC.cs` — class `EnsembleRunner_v02`, `Name="EnsembleRunner_v0.2-RC"`, `VERSION="0.2.0-RC"`.
- `research/nt8_port/csharp/v02/` — single-source core, generator, shim, verifiers.
- this report.

---

## 1. Parity result (v0.2 shared core vs golden, 20 reference days)

Verifier: `research/nt8_port/csharp/v02/p2b_verify.py` (scores `out_v02/` vs the golden
parquets + the compact reference, the SAME scoring the harness passed in P2).

| metric | v0.2 shared core | bar |
|---|---|---|
| 22 top-K fire-states | **100.000%** (178,640 / 178,640 cells; 0 mismatch) | ≥99.5% |
| governing entry decision | **100.000%** (8,120 / 8,120 bars; **913 entries**) | 100% |
| gov direction on entries | **100.000%** (913 / 913) | 100% |
| compact combiner P | max \|dP\| = **2.220e-16** (0 P-defined disagreements) | ≤1e-6 |
| `zz_leg` | **100.000%** (8,120 / 8,120) | 100% |
| `zz_confirm` | **100.000%** (8,120 / 8,120) | 100% |
| `zz_pivot_age_min` | max \|d\| = **0.000e+00** (bit-exact) | ≤1e-6 |
| `zz_pivot_price` | max \|d\| = **0.000e+00** (bit-exact) | ≤1e-6 |
| shim output vs harness `out_baseline/` | **byte-identical**, all 20 days | — |

Worst-case stream = 0 mismatches. The `P = 2.22e-16` residual is one IEEE ULP (same
as P1/P2) — the model logistic is arithmetically identical, not merely close.

**Interpretation.** The v0.2 core is not "close to" the harness — the shim writes the
`out_v02/*.json` and it is **byte-for-byte equal** to the harness `out/*.json`, which
the P2 build already proved is 100.000% vs golden. So the down-level (C#7.3 / .NET4.8)
is arithmetic-neutral by construction, then double-checked against golden directly.

---

## 2. Sharing / diff mechanism (parity, not vibes)

NinjaScript ships as ONE self-contained `.cs` and cannot `#include`. To guarantee the
shim proves *exactly* the code that runs in NT8, there is **one source of truth** and
everything else is generated + hash-checked:

```
v02/core_logic.cs.inc      (hand-ported, down-levelled logic; types only)
v02/_generated_data.cs.inc (gen_data.py: _model.json + _tmpl0.json -> C# constants)
        │  assemble.py
        ▼
v02/EnsembleCoreV02.region.cs   <-- THE canonical shared region (usings + types)
        ├─► v02/shim/EnsembleCoreV02.gen.cs   (region wrapped in `namespace EnsembleV02Core {}`)
        └─► docs/nt8/7-EnsembleRunner_v0.2-RC.cs (region injected VERBATIM between its markers)
```

- `assemble.py` writes the canonical region, wraps it for the shim, and injects it
  **byte-for-byte** into the strategy (C# is whitespace-insensitive, so the block sits
  at column 0 inside the strategy's nested `namespace EnsembleV02Core` unchanged).
- `verify_region.py` extracts the region from all three files and asserts an identical
  **SHA-256**:

  ```
  canonical region sha256: 48e84368…818d40 (148,312 bytes)
  strategy  region sha256: 48e84368…818d40 MATCH
  shim      region sha256: 48e84368…818d40 MATCH
  VERDICT: PASS -- one identical region in canon + strategy + shim
  ```

So the shim (which passes 100.000% vs golden) and the NinjaScript strategy contain the
**same bytes** for the entire generator + model + TMPL0 + consensus + zigzag core. Edit
the core only via the source `.inc` files + `assemble.py`; never hand-edit the region.

Embedded frozen data (P2-2 / P2-4): `ModelData` (27 cols, coef, mu, sd, top-K,
threshold `0.7139834155227371`, consensus 180s) and `Tmpl0Data` (1020 templates ×6 +
scaler + long_frac + member_count) are emitted from `_model.json` / `_tmpl0.json` as
shortest-round-trip decimal literals — the SAME IEEE doubles the harness parsed. No
runtime file IO in NT8; the codebook is compiled in.

---

## 3. LangVersion 7.3 down-level check (P2-13) — PASS

Two independent green builds at `LangVersion=7.3`:

| project | what it compiles | result |
|---|---|---|
| `v02/shim/V02ParityShim.csproj` | shared region + shim IO (JSON/gz) | **Build succeeded, 0 warn / 0 err** |
| `v02/nt8stub/Nt8Stub.csproj` | the ACTUAL `7-EnsembleRunner_v0.2-RC.cs` + a stubbed NT8 API | **Build succeeded, 0 warn / 0 err** |

The shim proves the **shared core** is 7.3-clean; the stub project proves the
**strategy wrapper** (OnStateChange / OnBarUpdate / RunDecision / helpers / properties)
is 7.3-clean and type-plausible against the NinjaScript surface it calls. LangVersion is
independent of the target framework, so a 7.3 green build is a valid down-level proof for
NT8's older Roslyn. No records, target-typed `new`, switch-expressions, ranges/indices,
init-only, or `using`-declarations remain.

---

## 4. Deviations from the harness (forced by C#7.3/.NET4.8) — each parity-neutral

All are proven neutral by the byte-identical shim output; none change arithmetic.

| # | deviation | why | proof it's neutral |
|---|---|---|---|
| D1 | named `ValueTuple`s → plain `struct`s (`NmpFire`, `Nmp9Ev`, `TmplSub`, `ZzResult`, `DmiAdx`-as-out-params) | NT8's old Roslyn / `System.ValueTuple` risk; keep it a pure language-feature-free surface | structs carry the same fields; same values flow |
| D2 | `Math.Log2(x)` → `Pd.Log2(x) = Math.Log(x)/Math.Log(2)` | `.NET4.8` has no `Math.Log2` | only feeds TMPL0 `log2_tf_secs` (3 fixed values, then standardized + nearest-centroid); shim still 100.000% |
| D3 | JSON codebook load → embedded `ModelData`/`Tmpl0Data` constants | NT8 self-contained `.cs`, no fragile runtime file paths | literals are shortest-round-trip = same doubles the harness parsed |
| D4 | LINQ `OrderBy(Ts).ThenBy(idx)` → explicit `Array.Sort` with `(Ts, then original index)` comparer | avoid a LINQ dependency in the ported core; make the stable order explicit | identical total order → identical consensus windows |
| D5 | TMPL0 write-only `Debug` capture dropped | it was diagnostic-only (`TMPL_DEBUG`), never read into output | removing a write-only side-channel cannot change output |

No deviation was forced by the **streaming** form itself — the generators are carried
verbatim; streaming is handled by the wrapper (§5), not by editing the core.

---

## 5. Streaming model + the 180s consensus settle (P2-1 / P2-7 resolution)

The proven core is a **per-day batch** (`Core.ProcessDay(Ctx)`). The strategy buffers
the session's 5s bars and re-runs the core once per **completed 1-minute bar** over the
day-so-far. Every generator is **causal** (rolling look-back, cooldowns, cumulative
state), so:

- a minute's `zz_confirm`/`zz_leg` is **final at minute close** → the **R-trigger exit**
  fires immediately, bit-identical to the golden per-minute zigzag sampling;
- a minute's **entry** decision only becomes final once the **±180s consensus window**
  around its fires has elapsed. The strategy therefore acts on minute `M`'s entry when
  the clock reaches `M + 180s` (the settle). At that point batch-over-day-so-far equals
  full-day batch for minute `M` → the **decision is bit-identical to golden**.

The only cost is a **3-bar (1m) action latency** vs the golden timestamp (P2-8, fill
semantics). This is *inherent to the validated feature*: the combiner was trained on the
forward ±180s consensus, so reproducing the validated edge REQUIRES feeding the same
consensus — an immediate causal-only variant is a *different, untested* feature, not a
port. Chosen: preserve the proven decision; flag the latency.

`P2-perf`: `ProcessDay` is O(N) per minute (~390 calls/day). Fine for backtest/live; an
optional incremental per-generator port would drop it to O(1)/bar without touching the
proven arithmetic.

---

## 6. Remaining TODOs — genuinely need the live NT8 loop (NOT faked)

Kept as numbered TODOs in the strategy header. These are **live plumbing** around the
proven core, not core logic. The parity proof feeds the core the harness-exported
`rth`/`before9`/`tod`/`zse`/`prior`/`start`; the strategy derives them live and each
derivation is flagged:

- **P2-3** native `z_se` bit-parity vs `core_v2 _ols_fit_kernel` (the harness EXPORTED
  `z_se`; strategy computes a 1m endpoint-OLS window-15 ddof-2, ffilled to 5s rows).
- **P2-5** DST-correct America/Chicago RTH / `before9` / `tod` / epoch-`ts` basis
  (NT8 exchange-local).
- **P2-8** entry fill semantics — acts ~180s (3×1m) after the signal minute (§5);
  confirm vs the harness "act at bar close `T+60`" convention.
- **P2-10** catastrophic stop as a real `ExitLongStopMarket` for live (not an intrabar
  poll).
- **P2-12** warmup / prior-day 5s TAIL (harness `Start≈2500`) + prior-daily profile
  equivalence vs the exported `prior_daily`.
- **P2-perf** optional incremental per-generator port (§5).

Resolved in v0.2 (were open in v0.1): P2-1, P2-2, P2-4, P2-6, P2-7, P2-9, P2-13.

---

## 7. Files + how to reproduce

```
research/nt8_port/csharp/v02/
  gen_data.py                 # _model.json + _tmpl0.json -> _generated_data.cs.inc
  core_logic.cs.inc           # hand-ported, down-levelled decision-core logic
  _generated_data.cs.inc      # generated frozen constants (ModelData + Tmpl0Data)
  assemble.py                 # -> EnsembleCoreV02.region.cs, shim gen, strategy inject
  EnsembleCoreV02.region.cs   # canonical shared region (single source of truth)
  verify_region.py            # SHA-256 identity: canon == strategy == shim
  p2b_verify.py               # golden parity scorer for out_v02/
  shim/  V02ParityShim.csproj + ShimMain.cs + EnsembleCoreV02.gen.cs (generated)
  nt8stub/ Nt8Stub.csproj + Nt8Stub.cs   # LangVersion 7.3 compile of the strategy

# reproduce (dotnet 10; python3.11):
python3.11 research/nt8_port/csharp/v02/gen_data.py
python3.11 research/nt8_port/csharp/v02/assemble.py
python3.11 research/nt8_port/csharp/v02/verify_region.py                       # region identity PASS
dotnet build -c Release research/nt8_port/csharp/v02/shim                       # 7.3, 0/0
dotnet build -c Release research/nt8_port/csharp/v02/nt8stub                    # 7.3, 0/0 (strategy)
research/nt8_port/csharp/v02/shim/bin/Release/net10.0/v02shim.exe research/nt8_port/csharp
python3.11 research/nt8_port/csharp/v02/p2b_verify.py                            # VERDICT PASS 100.000%
```

## 8. Deploy status
`-RC`. NOT NT8-compiled (no NinjaTrader assemblies in this environment). NOTHING copied
to `Documents/NinjaTrader 8/bin/Custom/Strategies/`. Promotion requires explicit
per-revision user approval per the house deploy gate.
