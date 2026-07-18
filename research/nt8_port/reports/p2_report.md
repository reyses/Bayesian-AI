# P2 — NinjaScript strategy draft + native zigzag + TMPL0 tie-rule pin (task 133)

**Date:** 2026-07-18 · **Executor:** Opus reviewer drone · **Status:** DONE (bench-verified; NOT NT8-compiled)

Three deliverables: (1) pin a deterministic TMPL0 same-ts tie rule in BOTH the
golden generator and the C# port → regenerate golden → rerun parity; (2) port the
R-trigger zigzag natively into the C# harness with 100% pivot parity vs the golden
`zz_*` columns; (3) draft `docs/nt8/7-EnsembleRunner_v0.1-RC.cs`. Deploy gate
honored — **nothing copied to any NinjaTrader folder.**

---

## 1. TMPL0 tie-rule pin — result

### The rule (pinned identically on both sides)
For a bar with ≥1 TMPL0 sub-fire, the `f_TMPL0` fire-state is:
1. **highest-TF event wins** (900 > 300 > 60 s);
2. tie on TF → the event whose **conviction `|long_frac − 0.5|` is larger**;
3. still exactly tied (same TF, same conviction, opposite dirs) → **hold prior state = 0**
   (fire-states are per-bar and don't carry, so the within-bar default is 0).

Order-independent and reproducible: `tf` is an integer (60/300/900); conviction is
computed from the SAME frozen 4-dp codebook `long_frac` on both sides, so the equality
tie-break compares bit-identical IEEE doubles.

**Where implemented:**
- `research/nt8_port/tools/golden_vector_gen.py` — new `resolve_tmpl0_state(...)`; `gen_tmpl0`
  now carries `tf_secs`; `aggregate_day` applies the rule to TMPL0 only (the other 21 streams
  keep last-fire-wins, already 100%).
- `research/nt8_port/tools/parity_check.py` — `compact_bars` mirrors the rule via
  `gvg.resolve_tmpl0_state`.
- `research/nt8_port/csharp/{Model,Tmpl0,Program}.cs` — `Fire.Tf` added; `Tmpl0.Run` emits it;
  `Program.ResolveTmpl0(...)` applies it in the per-bar aggregation.

### Parity table — BEFORE → AFTER
| metric | P1 (before) | P2 (after pin) | bar |
|---|---|---|---|
| **TMPL0 fire-state** | 67 / 8120 mismatch → **99.175%** | **0 / 8120 → 100.000%** | ≥99.5% |
| overall fire-state (22 streams) | 178,573 / 178,640 → 99.962% | **178,640 / 178,640 → 100.000%** | ≥99.5% |
| Python golden↔reference self-check | 67 / 178,640 | **0 / 178,640** | — |
| compact P (max abs diff) | 2.22e-16 | 2.22e-16 | ≤1e-6 |
| entry decision | 8120 / 8120 → 100.000% | 8120 / 8120 → 100.000% | 100% |

All 22 streams now **0 mismatched cells** across the 20 reference days. The P1 residual
(same-minute opposite-direction 1m + 5m/15m TMPL0 sub-fires whose "last-fire-wins" order was
undefined for same-ts fires under pandas quicksort) is fully removed; the "rounding boundary
drift" mentioned in P1 was the SAME ambiguity and also vanished (self-check 0/178,640).

---

## 2. Native R-trigger zigzag in C# — result

Ported `training/strategies/zigzag.py::ZigzagStrategy` (via `golden_vector_gen.zigzag_rtrigger`)
verbatim into the C# harness as `Program.ZigzagRTrigger(Ctx)`:
- streams the full 5s close series (incl. prior-day tail);
- `R = max(4, round(zz_thr[first_rth] / TICK))` ticks; `zz_thr = ATR(14 1m) × 4` points,
  causally open-anchored at the first RTH 5s bar;
- `extreme ± R` flip with `min_bars_5s = 36`;
- sampled per RTH 1m bar at the last 5s row of the minute → `zz_leg`, `zz_confirm`,
  `zz_pivot_age_min`, `zz_pivot_price`. `parity_check.compare` now scores these vs golden.

### Pivot parity table (C# vs golden zz_* columns, 20 days)
| metric | agreement | bar |
|---|---|---|
| `zz_leg` (leg direction) | **100.000%** (8120 / 8120 bars) | 100% |
| `zz_confirm` (pivot flip) | **100.000%** (8120 / 8120 bars) | 100% |
| `zz_pivot_age_min` (max abs diff) | **0.000e+00** min (bit-exact) | ≤1e-6 |
| `zz_pivot_price` (max abs diff) | **0.000e+00** pts (bit-exact) | ≤1e-6 |

The scalar-R reconciliation note from P0/P1 (causal open-anchored ATR vs the archived offline
whole-day median-TR) is a **knob**, not a state-machine difference — the golden and the C# port
use the identical open-anchored R, so parity is exact. If a future revision wants the offline-R
convention, it is a one-line change to how `min_rev_ticks` is computed, and both sides move together.

---

## 3. NinjaScript strategy draft — `docs/nt8/7-EnsembleRunner_v0.1-RC.cs`

Class `EnsembleRunner_v01`, `Name = "EnsembleRunner_v0.1-RC"`, `VERSION = "0.1-RC"`, header
banner + CHANGELOG per `docs/nt8/VERSIONING.md`. **Not NT8-compiled; nothing deployed.**

### Structure
- **Series**: primary = 5s (the substrate); secondary `AddDataSeries(Minute,1)` for z_se OLS +
  `zz_thr` ATR(14)×4.
- **`OnBarUpdate` dispatch**: on 1m close → update ATR-R + push native z_se; on 5s close →
  lock per-day R at first RTH bar, advance the R-trigger, EXIT check, catastrophic stop, session
  guard, then ENTRY.
- **Entry**: pooled 22-stream logistic combiner `P ≥ 0.713983`; side = governing (max-P) stream.
  1 contract, `EntriesPerDirection = 1`, `Calculate.OnBarClose` (closed-bar, no lookahead).
- **Exit**: R-trigger REVERSAL only (ride-only, doc 107) — a confirmed pivot against the open
  leg flattens it. No TP/MFE-cut/trail.
- **Guards**: `EnableCatastrophicStop` (default **false** = OFF in SIM) + `CatastrophicStopPoints`
  (present for live); session flatten 15:55 CT (`SessionFlattenHH/MM`), blocks re-entry.
- **z_se**: implemented NATIVELY as a 1m endpoint-OLS z (window 15, residual std ddof=2) —
  confirmed the SAME formula family as core_v2 `_ols_fit_kernel` / harness `MathX.Z21`, so it is
  portable. A `ZSeFeedMode` property keeps a file-feed fallback. z_se native derivation is flagged
  for bit-parity verification before live (P1 had EXPORTED it as an external input).
- **Engine boundary**: the 22 generators + consensus + compact logistic + TMPL0 (with the P2 tie
  rule) are the VALIDATED harness code (`research/nt8_port/csharp/{Gens,Tmpl0,Program}.cs`),
  represented as `EnsembleEngine` / `RTriggerZigzag`. The R-trigger is fully written (exact port);
  the generator bodies are the streaming port boundary.

### Numbered TODO — needs the live NT8 compile/verify loop
1. **P2-1** Port the 22 generator bodies (`Gens.cs`) into incremental streaming (per-bar) form.
2. **P2-2** Embed + byte-verify the frozen combiner model (`_model.json`: 27 cols, coef, mu, sd,
   topk) and the entry threshold.
3. **P2-3** Native z_se **bit-parity** vs `core_v2` `_ols_fit_kernel` (P1 exported z_se).
4. **P2-3b** File-feed z_se fallback path.
5. **P2-4** TMPL0 codebook (`_tmpl0.json`) load + nearest-centroid + the P2 same-bar tie rule.
6. **P2-5** DST-correct America/Chicago RTH / `tod` / `before9` session gate (NT8 exchange-local).
7. **P2-6** `zz_thr` ATR(14 1m)×4 basis identical to the harness (index `//12` buckets, `tr` def).
8. **P2-7** Consensus rolling ±180s same-direction fire buffer (streaming form of `day_consensus`).
9. **P2-8** Entry fill semantics vs the harness "act at bar close `T+60`" convention.
10. **P2-9** R-trigger exit wiring: confirmed pivot against the open leg closes it (ride-only).
11. **P2-10** Catastrophic stop as a real `ExitLongStopMarket` for live (not an intrabar poll).
12. **P2-11** Session flatten 15:55 CT + block re-entry until next session.
13. **P2-12** Warmup / prior-day-tail equivalence vs the harness `TAIL` context.
14. **P2-13** Down-level net10 / C#12 syntax to NT8 (.NET 4.8 / older Roslyn).

---

## 4. Deviations & decisions (with reasons)
- **"Hold prior state" = 0 within the bar** (not carried across bars). Fire-states reset per bar and
  `f=0` means "no directional decision"; carrying a nonzero state across bars would break that
  invariant. The dead-tie branch (same TF + same conviction + opposite dirs, requiring two mirror
  templates firing the same bar) does **not** occur in the 20 reference days — it is a safety net,
  and both sides return 0 identically.
- **Tie rule applied to TMPL0 only.** The other 21 streams are 100% under last-fire-wins and have no
  multi-TF concept; touching them would risk a regression for zero gain.
- **z_se implemented native in the .cs draft** despite P1 exporting it. Rationale: the formula is
  provably the same endpoint-OLS z the harness already computes bit-exact (`MathX.Z21`), just window
  15 on 1m bars. Kept a file-feed fallback + a hard TODO for the pre-live parity check.
- **R = open-anchored causal ATR** (not offline median-TR). Matches P0/P1 golden; the reconciliation
  is a scalar knob both sides share, so pivot parity is exact.
- **.cs is a skeleton, not a full generator port.** The 22 generator bodies live validated in the
  harness; re-authoring them blind in NinjaScript would risk drift from the 100%-parity code. The
  draft makes the engine boundary explicit and enumerates the port as TODO(P2-1) so the compile loop
  ports FROM the validated source, not from scratch.

## 5. How to reproduce
```
python3.11 research/nt8_port/tools/golden_vector_gen.py          # regenerate golden (tie rule)
python3.11 research/nt8_port/tools/parity_check.py export        # reference + harness inputs (self-check 0/178640)
cd research/nt8_port/csharp && dotnet build -c Release && ./bin/Release/net10.0/harness.exe .
python3.11 research/nt8_port/tools/parity_check.py compare       # -> reports/p1_parity.md (all 100%)
```
