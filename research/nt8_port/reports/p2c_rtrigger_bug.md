# P2c — Root-cause: v0.2-RC R-trigger reversal exit fired ZERO times

**Date:** 2026-07-18 · **Executor:** Opus drone · **Status:** DONE (bench-verified; NOT NT8-compiled)
**Scope:** WRAPPER-only. SHARED-CORE-V02 region byte-identical (verify_region.py PASS).
**Artifacts:** `docs/nt8/7-EnsembleRunner_v0.3-RC.cs` (fix),
`research/nt8_port/tools/p2c_rtrigger_replay.py` (driver),
`research/nt8_port/reports/p2c_replay_{2024_06_26,2025_10_16}.txt` (empirical logs).

---

## 1. Symptom

The v0.2-RC NT8 backtest (2026-06-22..07-17, 44 trades, multi-hour holds —
`backtest_v02_trades_2026-06-22_07-17.csv`) fired the `RTriggerReversal` exit
**ZERO times**. Every exit was `X_CatastrophicStop` (~20), `Exit on session close`
(NT8 template, ~14:00 CT), or `X_SessionFlatten`. The golden vectors carry
**~19.5 confirmed reversals/day** (measured: 3–40, mean ≈ 19.5 over the 20 golden
days), and several trades gave back their entire MFE while riding to the 14:00
close (e.g. trade 44: MFE **$592** → exit **+$35**, a ~$557 giveback), which a
wired reversal exit would have prevented. So the exit was structurally dead.

## 2. Method

Built `p2c_rtrigger_replay.py`: it loads the raw 5s ctx used to build the golden
(`research/nt8_port/csharp/harness_data/<day>.json.gz`), ports `Ctx.BuildZzThr`
and `zigzag_rtrigger` **verbatim**, and replays each RTH day **minute-by-minute
day-so-far exactly as `RunDecision` does** (re-run the batch core over `bars[0:end
of minute M]`, read `zz_confirm` at `curMin`). It logs, for each minute M, the
`zz_confirm` of the **M-truncated** run vs the **full-day** run, under two warmup
regimes, then simulates the exit gate `zz_confirm == -openDir`.

- **HARNESS regime** — prior-day tail present (`start≈2500`): the exact input that
  yields 100.000% golden parity. This is what v0.3 (fixed) reproduces.
- **COLD regime** — RTH-only, `start=0`, no tail: what the v0.2 wrapper actually
  runs at NT8 runtime (buffer cleared to empty each session).

## 3. What is NOT the bug (each suspect tested, not assumed)

| Suspect | Test | Verdict |
|---|---|---|
| **#2 temporal re-assignment** (confirm assigned to an earlier bar; per-minute re-run misses it) | Truncated vs full-day `zz_confirm`, warm regime | **NOT the bug.** The causal core is **truncation-invariant: 406/406 minutes identical, 0 confirms missed**. `min_rev` is fixed at `zz_thr[first_rth]` and the forward pass is causal, so `flip[i]` is identical in every truncation. Re-running the batch each minute reproduces every confirm at its true minute. |
| **#1b sign convention** | Code inspection of `ZigzagRTrigger` vs the exit gate | **Correct.** `flip=-1` ⇒ new **down**-leg confirmed ⇒ "against a long"; exit wants `zz_confirm == -openDir = -1`. Match. (`flip=+1` ⇒ up-leg ⇒ against a short.) |
| **#1a key-space alignment** | `curMin` vs `BarRec.BarTs` | **Aligned.** Both are `(ToEpoch(t)//60)*60` on the same buffer; `curMin` (the just-completed minute) is always an emitted RTH key holding that minute's 12 bars. Entry uses `curMin−180`, exit uses `curMin`; both are valid keys. |
| **#3 openDir bookkeeping** | Code path + backtest behavior | **Works both ways.** `OnPositionUpdate` sets `openDir=±1` on fill and `0` on flat; same-day re-entries after stops prove it resets, so it is nonzero during a hold. |

## 4. Root cause — the wrapper starts the R-trigger COLD (P2-12)

The parity harness feeds a **prior-day 5s tail** (`build_ctx`:
`full = concat([tail, df]); start = len(tail)`, TAIL ≈ Start ≈ 2500) so that
`ATR(14, 1m)` — and therefore the R-trigger reversal threshold
`min_rev = round(zz_thr[first_rth] / TICK)` — is **warm at RTH open**.

`v0.2 RollSession` **cleared the entire 5s buffer to empty each session** and
**hardwired `dayStartIdx = 0`**. `WarmupTailBars5s` (set to 2500) existed but was
**dead code — never consumed**. So `min_rev` was computed COLD on the volatile
RTH-open bars. Empirically (driver output):

```
VERDICT 2024_06_26
  v0.2 (cold, no tail)  runtime min_rev=[4, 272]  confirms/day=9   (full-day cold: 6)
  v0.3 (warm tail)      runtime min_rev=119        confirms/day=19
  v0.3 runtime==full-day zz_confirm at 406/406 minutes (truncation-invariant)

VERDICT 2025_10_16
  v0.2 (cold, no tail)  runtime min_rev=[4, 428]  confirms/day=9   (full-day cold: 7)
  v0.3 (warm tail)      runtime min_rev=167        confirms/day=40
  v0.3 runtime==full-day zz_confirm at 406/406 minutes (truncation-invariant)
```

Cold-start inflates `min_rev` **2.3–2.6×** (119→272 ticks; 167→428 ticks) and is
**unstable across the re-runs** (`[4, 272]` — before ATR warms, `min_rev`
degenerates to the 4-tick floor, so the state machine diverges between minutes).
A 2.6× threshold means a reversal must travel ~68 pts (vs the intended ~30 pts) and
hold 36 bars to confirm, so **confirmed reversals collapse to ~1/3** and land at
the wrong bars.

The few survivors are then **pre-empted**:
1. **Stop-race** — the catastrophic stop is polled on **every 5s bar**, but the
   R-trigger exit is evaluated **only once per completed minute** in `RunDecision`.
   The backtest ran the stop **ON** (against the SIM default OFF — every
   `X_CatastrophicStop` proves it), so any adverse move that reaches the stop inside
   a minute fires the stop before the minute's reversal check runs.
2. **14:00 session close** — NT8's trading-hours template auto-flattens at 14:00 CT
   (the "Exit on session close" fills), **before** the strategy's 15:55 flatten and
   before slow cold-threshold reversals can confirm.

Net over a trending 4-week window: **zero** `RTriggerReversal` exits. The fault is
entirely the wrapper's missing warmup — the SHARED-CORE is correct and
truncation-invariant.

## 5. The fix (v0.3-RC, wrapper-only)

`docs/nt8/7-EnsembleRunner_v0.3-RC.cs` — class `EnsembleRunner_v03`,
Name `"EnsembleRunner_v0.3-RC"`, VERSION `"0.3.0-RC"`. SHARED-CORE byte-identical.

- **Seed the warmup tail.** `RollSession` now calls a new `TrimBuffersToTail(
  WarmupTailBars5s)` that **retains the last `WarmupTailBars5s` buffered 5s bars**
  as the new session's tail (all parallel buffers trimmed in lockstep), and sets
  `dayStartIdx = bTs.Count` so `Ctx.Start` points at the first current-session bar.
  `ProcessDay` still emits/decides only on `i >= Start` (current RTH), so the tail
  warms `ATR` / the generators / the R-trigger **without being traded** — exactly
  the harness `concat([tail, df]); start=len(tail)` semantics. This restores
  `min_rev` to the parity value.
- **Documented (no behavior change):** the stop-race (keep the stop OFF in SIM to
  observe ride-only behavior), the stop residual-slip expectation (poll is already
  every 5s and uses the intrabar Low/High; the market flatten fills next 5s bar, so
  residual slip = trigger→next-bar travel — the v0.2 10–230% overage; a native
  `ExitLongStopMarket` remains **P2-10**, deferred to live), and the 14:00-vs-15:55
  session-close mismatch (**P2-5/P2-11**, left AS-IS).

## 6. Verification

- **Region SHA:** `verify_region.py` extended to check v0.3 —
  `strat v0.3 region sha256: 48e84368…818d40 MATCH` → **VERDICT: PASS** (canon +
  v0.2 + v0.3 + shim all identical, 148,312 bytes).
- **Replay (fix behavior):** the driver's HARNESS regime **is** the v0.3 fixed
  wrapper (tail seeded, `Start≈2500`). It reproduces the full-day `zz_confirm` at
  **406/406 minutes** (0 mismatches, 0 missed) on both test days, with `min_rev` =
  the parity value (119 / 167). The exit gate therefore fires at **every minute the
  full-day run confirms a reversal against the held direction** — 19 (2024-06-26)
  and 40 (2025-10-16) reachable reversals/day, vs the cold ~6–9 that v0.2 further
  drowned under the stop-race + 14:00 close.

## 7. Must wait for the NT8 compile/verify loop

- Exact **bit-parity of the warmup tail** vs the Databento export (the tail now
  comes from the NT8 tape, not the golden parquet): **P2-12 remains open** — v0.3
  restores *functional* warmth (real prior-day ATR), not byte-parity.
- **P2-5** session calendar (DST-correct CT) and **P2-11** flatten/close reconciliation
  (the 14:00 template close) — needed to stop NT8 force-closing before 15:55.
- **P2-10** native resting stop (bounded slip) for live.
- **P2-3** native z_se bit-parity. None of these block observing the R-trigger fix in
  a SIM backtest with the stop OFF.
